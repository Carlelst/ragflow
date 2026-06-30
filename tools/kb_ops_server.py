#!/usr/bin/env python3
"""
KB 运维 HTTP 服务 — 供 Dify Workflow 调用

端点:
  POST /import     {"source": "wiki", "doc_id": 0, "wait": true}
  POST /ingest     {"content_type": "url|text", "content": "...", "kb_name": "enflame-wiki"}
  POST /cleanup    {"kb": "enflame-wiki", "docs_only": false}
  GET  /status     {"kb": "enflame-wiki"}
  GET  /ingest/{task_id}  查询 ingest 任务状态

启动:
  python3 kb_ops_server.py --port 8100
"""

import argparse, json, os, re, subprocess, threading, time, uuid
from flask import Flask, request, jsonify

import requests

app = Flask(__name__)

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
IMPORT_SCRIPT = os.path.join(TOOLS_DIR, "batch_import.py")
CONFIG = os.path.join(TOOLS_DIR, "batch_config.yaml")
CLEANUP_SCRIPT = os.path.join(TOOLS_DIR, "kb_cleanup.py")

# Docker exec 模式: 脚本必须在容器内运行 (依赖 RAGFlow 内部模块)
DOCKER_CONTAINER = os.environ.get("RAGFLOW_CONTAINER", "ragflow-server")
VLM_ENABLED = os.environ.get("VLM_ENABLED", "1")
# 容器内路径 (docker cp 或 volume mount 同步进去的)
CONTAINER_TOOLS = "/ragflow/tools"

_tasks: dict[str, dict] = {}
_lock = threading.Lock()

# MySQL 直连 (用于 /status 查询 document 进度)
MYSQL_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "infini_rag_flow")
MYSQL_DB = os.environ.get("MYSQL_DB", "rag_flow")
# ---- RAGFlow 配置 ----
RAGFLOW_URL = os.environ.get("RAGFLOW_URL", "http://127.0.0.1:8088")
RAGFLOW_API_KEY = os.environ.get("RAGFLOW_API_KEY", "ragflow-307044760fae4f548209426ba6191d9e")
RAGFLOW_HEADERS = {"Authorization": f"Bearer {RAGFLOW_API_KEY}", "Content-Type": "application/json"}

# ---- LLM 配置 (元数据提取) ----
LLM_URL = os.environ.get("LLM_URL", "http://172.16.90.45:8082/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "scs_qwen3.5-397b")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))

# ---- KB 名称 → ID 缓存 ----
KB_NAME_CACHE: dict[str, str] = {}

# ---- ingest 异步任务存储 ----
_ingest_tasks: dict[str, dict] = {}


@app.route("/import", methods=["POST"])
def api_import():
    body = request.get_json(silent=True) or {}
    source = body.get("source", "wiki")
    doc_id = body.get("doc_id", 0)
    wait = body.get("wait", True)

    task_id = f"import_{source}_{doc_id}"
    with _lock:
        if task_id in _tasks and _tasks[task_id]["status"] == "running":
            return jsonify({"task_id": task_id, "status": "already_running"}), 409
        _tasks[task_id] = {"status": "running", "progress": ""}

    def run():
        cmd = [
            "docker", "exec", "-e", f"VLM_ENABLED={VLM_ENABLED}",
            DOCKER_CONTAINER,
            "python3", f"{CONTAINER_TOOLS}/batch_import.py",
            "--config", f"{CONTAINER_TOOLS}/batch_config.yaml",
            "--source", source,
        ]
        if doc_id:
            cmd += ["--doc-id", str(doc_id)]
        if wait:
            cmd.append("--wait")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            with _lock:
                _tasks[task_id]["status"] = "done" if result.returncode == 0 else "failed"
                _tasks[task_id]["output"] = result.stdout[-500:] + "\n" + result.stderr[-200:]
        except subprocess.TimeoutExpired:
            with _lock:
                _tasks[task_id]["status"] = "timeout"
        except Exception as e:
            with _lock:
                _tasks[task_id]["status"] = "failed"
                _tasks[task_id]["output"] = str(e)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"task_id": task_id, "status": "started"})


@app.route("/retry-failed", methods=["POST"])
def api_retry_failed():
    """重试所有失败的文档 (progress < 0)"""
    body = request.get_json(silent=True) or {}
    kb_name = body.get("kb", "enflame-wiki")

    script = (
        'import sys; sys.path.insert(0,"/ragflow")\n'
        'from common.settings import init_settings\n'
        'from common.config_utils import read_config\n'
        'read_config(); init_settings()\n'
        'from api.db.db_models import Document, Knowledgebase\n'
        'from api.db.services.file2document_service import File2DocumentService\n'
        'from api.db.services.task_service import queue_tasks\n'
        'kb=Knowledgebase.select().where(Knowledgebase.name=="' + kb_name + '").first()\n'
        'count=0\n'
        'for doc in Document.select().where(Document.kb_id==kb.id, Document.progress<0):\n'
        '    Document.update(progress=0, status="1", progress_msg="").where(Document.id==doc.id).execute()\n'
        '    d=doc.to_dict()\n'
        '    b,n=File2DocumentService.get_storage_address(doc_id=doc.id)\n'
        '    queue_tasks(d,b,n,0)\n'
        '    count+=1\n'
        'print(count)'
    )

    cmd = ["docker", "exec", DOCKER_CONTAINER, "python3", "-c", script]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            count = result.stdout.strip()
            return jsonify({"ok": True, "retried": int(count) if count.isdigit() else 0})
        return jsonify({"ok": False, "error": result.stderr[-500:]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/requeue", methods=["POST"])
def api_requeue():
    """重新入队所有 pending 文档 (progress>=0 且 <1)"""
    body = request.get_json(silent=True) or {}
    kb_name = body.get("kb", "enflame-wiki")

    script = (
        'import sys; sys.path.insert(0,"/ragflow")\n'
        'from common.settings import init_settings\n'
        'from common.config_utils import read_config\n'
        'read_config(); init_settings()\n'
        'from api.db.db_models import Document, Knowledgebase\n'
        'from api.db.services.file2document_service import File2DocumentService\n'
        'from api.db.services.task_service import queue_tasks\n'
        'kb=Knowledgebase.select().where(Knowledgebase.name=="' + kb_name + '").first()\n'
        'count=0\n'
        'for doc in Document.select().where(Document.kb_id==kb.id, Document.progress>=0, Document.progress<1):\n'
        '    d=doc.to_dict()\n'
        '    b,n=File2DocumentService.get_storage_address(doc_id=doc.id)\n'
        '    queue_tasks(d,b,n,0)\n'
        '    count+=1\n'
        'print(count)'
    )

    cmd = ["docker", "exec", DOCKER_CONTAINER, "python3", "-c", script]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            count = result.stdout.strip()
            return jsonify({"ok": True, "queued": int(count) if count.isdigit() else 0})
        return jsonify({"ok": False, "error": result.stderr[-500:]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/cleanup", methods=["POST"])
def api_cleanup():
    body = request.get_json(silent=True) or {}
    kb = body.get("kb", "enflame-wiki")
    docs_only = body.get("docs_only", False)

    cmd = [
        "docker", "exec", "-e", "MYSQL_HOST=ragflow-mysql",
        "-e", "MYSQL_PORT=3306",
        "-e", "ES_HOST=http://ragflow-es:9200",
        DOCKER_CONTAINER,
        "python3", f"{CONTAINER_TOOLS}/kb_cleanup.py",
        "--kb", kb,
    ]
    if docs_only:
        cmd.append("--docs-only")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return jsonify({"ok": result.returncode == 0, "output": result.stdout})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/status", methods=["GET", "POST"])
def api_status():
    if request.method == "POST":
        kb = (request.get_json(silent=True) or {}).get("kb", "enflame-wiki")
    else:
        kb = request.args.get("kb", "enflame-wiki")

    import pymysql
    try:
        conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
                               password=MYSQL_PASSWORD, database=MYSQL_DB)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                  IFNULL(SUM(CASE WHEN progress >= 1.0 THEN 1 ELSE 0 END), 0) as done,
                  IFNULL(SUM(CASE WHEN progress < 0 THEN 1 ELSE 0 END), 0) as failed,
                  IFNULL(SUM(CASE WHEN progress >= 0 AND progress < 1 THEN 1 ELSE 0 END), 0) as pending,
                  COUNT(*) as total
                FROM document WHERE kb_id = (SELECT id FROM knowledgebase WHERE name=%s)
            """, (kb,))
            row = cur.fetchone()
        conn.close()
        return jsonify({"done": int(row[0]), "failed": int(row[1]), "pending": int(row[2]), "total": int(row[3])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/tasks", methods=["GET"])
def api_tasks():
    with _lock:
        return jsonify(_tasks)


# ============================================================
# /ingest — 单条元输入即时导入 RAGFlow（供 Dify Workflow 调用）
# ============================================================

def _resolve_kb_id(kb_name: str) -> str:
    """根据 KB 名称查找 KB ID，带内存缓存"""
    if kb_name in KB_NAME_CACHE:
        return KB_NAME_CACHE[kb_name]
    try:
        resp = requests.get(
            f"{RAGFLOW_URL}/api/v1/datasets?name={kb_name}",
            headers=RAGFLOW_HEADERS, timeout=10,
        )
        data = resp.json()
        if data.get("code") == 0 and data.get("data"):
            kb_id = data["data"][0]["id"]
            KB_NAME_CACHE[kb_name] = kb_id
            return kb_id
    except Exception:
        pass
    raise ValueError(f"KB '{kb_name}' not found in RAGFlow")


def _fetch_url_content(url: str) -> dict:
    """抓取 URL 内容"""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; KB-Ingest/1.0)",
    }
    resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    # 粗略判断是否为文本类型
    text_types = ("text/", "application/json", "application/xml", "application/javascript")
    is_text = any(content_type.startswith(t) for t in text_types)
    return {
        "content": resp.text if is_text else f"[binary: {content_type}]",
        "content_type": content_type,
        "status_code": resp.status_code,
        "url": resp.url,  # 最终 URL (跟随重定向后)
    }


def _extract_metadata(content: str, source_url: str = "", content_type: str = "text") -> dict:
    """通过 LLM 从内容中提取元数据"""
    # 截断输入，避免超 token 限制
    truncated = content[:8000] if len(content) > 8000 else content

    prompt = f"""从以下文档内容中提取元数据，返回严格 JSON（不要 markdown 代码块，不要额外说明）。
字段:
- title: 文档标题 (必填，不超过100字)
- summary: 200字以内的摘要
- tags: 3-5个关键词数组
- doc_type: 文档类型 (article/wiki/manual/faq/spec/other)

{'来源URL: ' + source_url if source_url else ''}
输入类型: {content_type}

内容:
{truncated}"""

    try:
        resp = requests.post(
            LLM_URL,
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个文档元数据提取器。只返回 JSON，不要其他内容。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 512,
            },
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        raw = result["choices"][0]["message"]["content"].strip()

        # 容错：移除可能的 markdown 代码块包裹
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        meta = json.loads(raw)
        return {
            "title": str(meta.get("title", "")).strip() or "未命名文档",
            "summary": str(meta.get("summary", "")).strip(),
            "tags": meta.get("tags", []) if isinstance(meta.get("tags"), list) else [],
            "doc_type": meta.get("doc_type", "other"),
        }
    except (json.JSONDecodeError, KeyError, requests.RequestException) as e:
        # 降级：从内容首行提取标题
        first_line = content.strip().split("\n")[0].strip("# ").strip()[:100]
        return {
            "title": first_line or "未命名文档",
            "summary": "",
            "tags": [],
            "doc_type": "other",
            "_fallback": True,
            "_error": str(e),
        }


def _clean_content(content: str, title: str, source_url: str = "", max_chars: int = 100000) -> str:
    """清洗内容：格式化、注入标题、截断"""
    # 去除开头多余空行
    content = content.strip()
    # 截断
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[内容已截断，原始长度 {len(content)} 字符]"
    # 注入标题和来源
    header = f"# {title}\n\n"
    if source_url:
        header += f"> 来源: {source_url}\n\n"
    return header + content


def _upload_to_ragflow(kb_id: str, title: str, content: str) -> dict:
    """上传文档到 RAGFlow，返回 doc_id"""
    resp = requests.post(
        f"{RAGFLOW_URL}/api/v1/datasets/{kb_id}/documents",
        json={
            "name": title,
            "content": content,
            "parser_method": "manual",
            "parser_config": {
                "chunk_token_num": 600,
                "delimiter": "\n\n",
            },
        },
        headers=RAGFLOW_HEADERS,
        timeout=30,
    )
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"RAGFlow upload failed: {data.get('message', resp.text)}")
    doc_data = data["data"]
    doc_id = doc_data.get("id") or doc_data.get("document_id") or doc_data.get("doc_id")
    return {"doc_id": doc_id, "kb_id": kb_id, "response": doc_data}


def _poll_parse_status(kb_id: str, doc_id: str, max_wait: int = 120) -> dict:
    """轮询 RAGFlow 文档解析状态"""
    url = f"{RAGFLOW_URL}/api/v1/datasets/{kb_id}/documents/{doc_id}"
    deadline = time.time() + max_wait

    while time.time() < deadline:
        try:
            resp = requests.get(url, headers=RAGFLOW_HEADERS, timeout=10)
            data = resp.json()
            if data.get("code") != 0:
                return {"status": "api_error", "error": data.get("message", "")}

            doc = data.get("data", {})
            progress = doc.get("progress", 0)

            if progress >= 1.0:
                return {
                    "status": "completed",
                    "progress": 1.0,
                    "chunk_count": doc.get("chunk_count", 0),
                    "token_count": doc.get("token_count", 0),
                    "doc_id": doc_id,
                    "kb_id": kb_id,
                }
            elif progress < 0:
                return {
                    "status": "failed",
                    "progress": progress,
                    "error": doc.get("status", "parse error"),
                    "doc_id": doc_id,
                    "kb_id": kb_id,
                }
        except requests.RequestException:
            pass  # 网络抖动，继续重试

        time.sleep(3)

    return {
        "status": "timeout",
        "doc_id": doc_id,
        "kb_id": kb_id,
        "message": f"文档已上传但解析未在 {max_wait}s 内完成，请稍后通过 GET /status 查询",
    }


@app.route("/ingest", methods=["POST"])
def api_ingest():
    """
    单条元输入导入 RAGFlow

    Request JSON:
      - content_type: "url" | "text" (必填)
      - content: 原始内容或 URL (必填)
      - kb_name: 目标知识库名 (必填，如 "enflame-wiki")
      - metadata: 预设元数据 (可选，不传则 LLM 自动提取)
          { title, summary, tags[], doc_type }
      - source_url: 来源 URL (可选，用于溯源)
      - wait: 是否等待解析完成 (默认 true，Dify 同步调用)

    Response:
      {
        "ok": true,
        "doc_id": "...",
        "kb_id": "...",
        "title": "...",
        "parse_status": "completed|pending|timeout",
        "chunk_count": N,
        ...
      }
    """
    body = request.get_json(silent=True) or {}
    content_type = body.get("content_type", "text")
    content = body.get("content", "")
    kb_name = body.get("kb_name", "")
    preset_meta = body.get("metadata")
    source_url = body.get("source_url", "")
    wait = body.get("wait", True)

    # ---- 参数校验 ----
    if not content:
        return jsonify({"ok": False, "error": "content is required"}), 400
    if not kb_name:
        return jsonify({"ok": False, "error": "kb_name is required"}), 400
    if content_type not in ("url", "text"):
        return jsonify({"ok": False, "error": "content_type must be 'url' or 'text'"}), 400

    # ---- 异步模式 ----
    if not wait:
        task_id = str(uuid.uuid4())[:8]
        with _lock:
            _ingest_tasks[task_id] = {"status": "running", "created_at": time.time()}

        def _async_ingest():
            try:
                result = _do_ingest(content_type, content, kb_name, preset_meta, source_url)
                with _lock:
                    _ingest_tasks[task_id] = {**result, "status": "completed"}
            except Exception as e:
                with _lock:
                    _ingest_tasks[task_id] = {"status": "failed", "error": str(e)}

        threading.Thread(target=_async_ingest, daemon=True).start()
        return jsonify({"ok": True, "task_id": task_id, "status": "started"})

    # ---- 同步模式 ----
    try:
        result = _do_ingest(content_type, content, kb_name, preset_meta, source_url)
        return jsonify({"ok": True, **result})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": f"internal: {e}"}), 500


@app.route("/ingest/<task_id>", methods=["GET"])
def api_ingest_status(task_id):
    """查询异步 ingest 任务状态"""
    with _lock:
        task = _ingest_tasks.get(task_id)
    if not task:
        return jsonify({"ok": False, "error": "task not found"}), 404
    return jsonify({"ok": True, "task_id": task_id, **task})


def _do_ingest(content_type: str, content: str, kb_name: str,
               preset_meta: dict | None, source_url: str) -> dict:
    """执行完整的 ingest 流程"""

    # 1. 解析 KB ID
    kb_id = _resolve_kb_id(kb_name)

    # 2. 获取内容
    if content_type == "url":
        fetched = _fetch_url_content(content)
        raw_content = fetched["content"]
        if not source_url:
            source_url = fetched.get("url", content)
    else:
        raw_content = content

    if not raw_content or not raw_content.strip():
        raise ValueError("extracted content is empty")

    # 3. 提取/使用元数据
    if preset_meta:
        meta = {
            "title": str(preset_meta.get("title", "未命名文档")).strip(),
            "summary": str(preset_meta.get("summary", "")).strip(),
            "tags": preset_meta.get("tags", []),
            "doc_type": preset_meta.get("doc_type", "other"),
        }
    else:
        meta = _extract_metadata(raw_content, source_url, content_type)

    # 4. 清洗内容
    cleaned = _clean_content(raw_content, meta["title"], source_url)

    # 5. 上传到 RAGFlow
    upload_result = _upload_to_ragflow(kb_id, meta["title"], cleaned)
    doc_id = upload_result["doc_id"]

    # 6. 轮询解析状态
    parse_result = _poll_parse_status(kb_id, doc_id)

    return {
        "doc_id": doc_id,
        "kb_id": kb_id,
        "kb_name": kb_name,
        "title": meta["title"],
        "summary": meta.get("summary", ""),
        "tags": meta.get("tags", []),
        "doc_type": meta.get("doc_type", "other"),
        "source_url": source_url,
        "parse_status": parse_result.get("status"),
        "chunk_count": parse_result.get("chunk_count", 0),
        "token_count": parse_result.get("token_count", 0),
        "fallback_meta": meta.get("_fallback", False),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    print(f"kb_ops_server starting on :{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
