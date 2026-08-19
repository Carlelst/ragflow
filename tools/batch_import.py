#!/usr/bin/env python3
"""
RAGFlow 批量导入工具 — 按数据源分别导入

支持三种数据源:
  - wiki:    Confluence wiki 页面 (wiki_metadata)
  - html:    文档站 HTML 内容 (html_metadata)
  - wangpan: 网盘文件 (wangpan_metadata)

用法:
  # 使用配置文件
  python3 batch_import.py --config batch_config.yaml

  # 命令行覆盖配置
  python3 batch_import.py --source wiki --limit 10
  python3 batch_import.py --source html --enable-graphrag --enable-raptor

  # 等待向量化完成 (轮询直到全部 done/fail)
  python3 batch_import.py --source wiki --wait

  # 查看所有KB状态
  python3 batch_import.py --status
"""

import argparse, json, random, string, sys, os, time, base64, io, re
from datetime import datetime

import requests
from minio import Minio


# ══════════════════════════════════════════════════════════════════════
# VLM 图片识别配置
# RAGFlow VisionFigureParser prompt
VLM_PROMPT = "Describe this image in detail. If it contains a table, list ALL rows and columns exactly. If it contains text, transcribe verbatim. Output raw data, no summary."
# ══════════════════════════════════════════════════════════════════════

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://172.16.90.45:8082/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "scs_qwen3.5-397b")
VLLM_MAX_TOKENS = int(os.environ.get("VLLM_MAX_TOKENS", "16384"))
VLLM_TIMEOUT = int(os.environ.get("VLLM_TIMEOUT", "180"))
VLM_ENABLED = os.environ.get("VLM_ENABLED", "1") == "1"


def _resolve_image_path(minio_client, bucket, doc_minio_key, relative_path):
    """将 markdown 中的相对图片路径解析为 MinIO 中的实际路径"""
    import os as _os
    doc_dir = _os.path.dirname(doc_minio_key)
    resolved = _os.path.normpath(_os.path.join(doc_dir, relative_path)).replace("\\", "/")
    try:
        minio_client.stat_object(bucket, resolved)
        return resolved
    except Exception:
        pass
    # 向上查找 images/ 目录
    filename = _os.path.basename(relative_path)
    parts = doc_dir.split("/")
    for i in range(len(parts), 0, -1):
        candidate = "/".join(parts[:i]) + "/images/" + filename
        try:
            minio_client.stat_object(bucket, candidate)
            return candidate
        except Exception:
            continue
    return None


def _describe_image_vllm(image_bytes, alt_text=""):
    """RAGFlow VisionFigureParser prompt"""
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = VLM_PROMPT
    if alt_text:
        prompt = f"Alt text: {alt_text}\n{prompt}"

    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]}],
        "max_tokens": VLLM_MAX_TOKENS,
        "temperature": 0.1,
    }
    resp = requests.post(f"{VLLM_BASE_URL}/chat/completions", json=payload, timeout=VLLM_TIMEOUT)
    resp.raise_for_status()
    result = resp.json()
    desc = result["choices"][0]["message"]["content"].strip()
    return desc if desc else None


def _build_pg_meta(row, source_key):
    """根据数据源类型，从 PG 行数据构建元数据字典"""
    meta = {}
    if source_key == "wiki":
        for pg_col, label in [
            ("title", "title"), ("source_url", "wiki_url"),
            ("page_id", "wiki_page_id"), ("space_key", "wiki_space"),
        ]:
            v = row.get(pg_col)
            if v:
                meta[label] = str(v)
        wp = row.get("wiki_path")
        if isinstance(wp, list):
            meta["wiki_path"] = " > ".join(str(p) for p in wp)
        elif wp:
            meta["wiki_path"] = str(wp)
        img = row.get("image_urls")
        if isinstance(img, list) and img:
            meta["image_urls"] = ", ".join(str(u) for u in img)
    elif source_key == "html":
        for pg_col, label in [
            ("title", "title"), ("source_url", "source_url"),
            ("description", "description"), ("author", "author"),
            ("domain", "domain"), ("keywords", "keywords"),
        ]:
            v = row.get(pg_col)
            if v:
                meta[label] = str(v)
    elif source_key == "wangpan":
        for pg_col, label in [
            ("wangpan_file_name", "title"), ("wangpan_file_path", "file_path"),
            ("wangpan_editor", "editor"), ("wangpan_size", "file_size"),
        ]:
            v = row.get(pg_col)
            if v:
                meta[label] = str(v)
    elif source_key == "git":
        for pg_col, label in [
            ("repo_url", "repo_url"), ("branch", "branch"),
            ("file_path", "file_path"), ("source_url", "source_url"),
            ("commit_hash", "commit_hash"), ("author", "author"),
        ]:
            v = row.get(pg_col)
            if v:
                meta[label] = str(v)
        cd = row.get("commit_date")
        if cd:
            meta["commit_date"] = cd.isoformat() if hasattr(cd, "isoformat") else str(cd)
        # title 用 file_path 的 basename（git 表无 title 列）
        fp = row.get("file_path")
        if fp:
            meta["title"] = fp.rsplit("/", 1)[-1]
    elif source_key == "local":
        # local_metadata: docx→md 转换的本地文档
        for pg_col, label in [
            ("local_file_name", "original_file_name"),
            ("local_file_path", "original_file_path"),
            ("content_type", "content_type"),
            ("task_id", "task_id"),
        ]:
            v = row.get(pg_col)
            if v:
                meta[label] = str(v)
    lu = row.get("last_updated")
    if lu:
        meta["last_updated"] = lu.isoformat() if hasattr(lu, "isoformat") else str(lu)
    return {k: v for k, v in meta.items() if v}

def read_first_line(minio_client, bucket, minio_key, max_bytes=4096):
    """读取 MinIO 对象的第一行文本。md 直接读; 返回第一行(去换行)或 None。"""
    ext = minio_key.rsplit('.', 1)[-1].lower() if '.' in minio_key else ''
    if ext not in ("md", "markdown", "txt"):
        return None  # xlsx 等二进制暂不支持跨 project 标记
    try:
        obj = minio_client.get_object(bucket, minio_key)
        data = obj.read(max_bytes)
        obj.close(); obj.release_conn()
        text = data.decode("utf-8", errors="replace")
        first_line = text.splitlines()[0] if text.splitlines() else text
        return first_line.strip()
    except Exception as e:
        print(f"[EKBTAG] read_first_line fail {minio_key}: {e}")
        return None


EKB_TAG_RE = re.compile(r'【\s*EKB[_\-]?\s*([0-9.]+)\s*】', re.IGNORECASE)


def extract_ekb_tags(minio_client, bucket, minio_key):
    """解析文档第一行的【EKB_X.Y】标记，返回排序后的版本号列表（去重）。
    md 才解析; xlsx 或读失败返回 []（走 PG project_name 兜底）。"""
    fl = read_first_line(minio_client, bucket, minio_key)
    if not fl:
        return []
    tags = []
    seen = set()
    for m in EKB_TAG_RE.findall(fl):
        v = m.strip()
        if v and v not in seen:
            seen.add(v)
            tags.append(v)
    return sorted(tags)


def build_full_meta(row, source_key, pg_project):
    """组装完整 doc metadata：基础 PG 字段 + project 关联。"""
    meta = _build_pg_meta(row, source_key)
    meta["project_name"] = pg_project  # 恒等于 PG project_name
    meta["project_tags"] = [pg_project]  # 无标记默认单一主标记
    meta["related_projects"] = []
    return meta


def enrich_project_meta(meta, tags, pg_project):
    """按解析到的 EKB 标记覆盖 project 关联。
    有标记 → 全量覆盖 project_tags；related = tags - main。
    无标记 → 保持默认 (仅 main)。"""
    if not tags:
        return meta
    meta["project_tags"] = sorted(tags)
    related = sorted(t for t in tags if t != pg_project)
    meta["related_projects"] = related
    return meta


def log_ekb_anomaly(row, pb, tags, log_path, source_key):
    """记录『有 EKB 标记但不含 PG project_name』的文档，供人工找文档所属人修改。"""
    try:
        line = (f"{datetime.now().isoformat()} | source={source_key} | "
                f"name={row.get('title') or row.get('wangpan_file_name') or ''} | "
                f"minio_key={row.get('minio_key')} | pg_project={pb} | tags={tags}\n")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[EKBTAG] log anomaly fail: {e}")


def enrich_markdown_images(minio_client, bucket, minio_key, source_url, image_urls=None):
    """
    将 markdown 中的相对图片路径替换为 MinIO HTTP URL，
    让 RAGFlow 原生 Markdown parser 自行下载图片并调用 VLM。

    返回: (processed_text, uploaded_minio_path) 或 (None, None)
    不修改原始 MinIO 文件。
    """
    try:
        data = minio_client.get_object(bucket, minio_key)
        content = data.read().decode("utf-8")
        data.close()
        data.release_conn()
    except Exception:
        return None, None

    is_markdown = minio_key.endswith(".md")
    if not is_markdown:
        return None, None

    img_refs = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", content)
    if not img_refs:
        return None, None

    modified = False
    for alt_text, rel_path in img_refs:
        if rel_path.startswith(("http://", "https://", "data:")):
            continue
        if not rel_path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")):
            continue

        img_path = _resolve_image_path(minio_client, bucket, minio_key, rel_path)
        if not img_path:
            continue

        # 图片引用不替换为 HTTP URL，保留相对路径。
    # RAGFlow 无法下载本地路径的图片，自动跳过，不影响分块速度。
    # 原始 ![]() 引用保留在正文中，供日后使用。
    return None, None


# ══════════════════════════════════════════════════════════════════════# 数据源配置
# ══════════════════════════════════════════════════════════════════════

SOURCE_CONFIGS = {
    "wiki": {
        "kb_name": "ekb_wiki",
        "source_table": "wiki_metadata",
        "description": "Confluence Wiki 文档",
        "default_chunk_tokens": 256,
    },
    "html": {
        "kb_name": "ekb_docs",
        "source_table": "html_metadata",
        "description": "文档站 HTML 内容",
        "default_chunk_tokens": 512,
    },
    "wangpan": {
        "kb_name": "ekb_pan",
        "source_table": "wangpan_metadata",
        "description": "企业网盘文件",
        "default_chunk_tokens": 512,
    },
    "git": {
        "kb_name": "ekb_git",
        "source_table": "git_metadata",
        "description": "GitLab 仓库文档",
        "default_chunk_tokens": 512,
    },
    "local": {
        "kb_name": "ekb_register_html",
        "source_table": "local_metadata",
        "description": "本地注册文档 (docx→md)",
        "default_chunk_tokens": 512,
    },
}

# 外部数据源连接信息 — 命令行 / YAML 可覆盖
PG_DEFAULTS = {
    "host": "10.9.200.14",
    "port": 5432,
    "user": "postgres",
    "password": "enflame",
    "dbname": "metadata",
}

MINIO_DEFAULTS = {
    "host": "172.16.90.36:9000",
    "user": "minioadmin",
    "password": "minioadmin",
    "bucket": "rag-data",
}


# ══════════════════════════════════════════════════════════════════════
# YAML 配置文件加载
# ══════════════════════════════════════════════════════════════════════

def load_yaml_config(path):
    """从 YAML 配置文件加载参数。

    配置文件格式 (batch_config.yaml)::

        source: wiki          # wiki | html | wangpan | all
        limit: 100            # 导入上限
        chunk_tokens: 200     # chunk token 数

        enable_graphrag: false
        enable_raptor: false
        no_parse: false
        wait_for_parse: true  # 等待向量化完成后再退出

        pg:
          host: "10.9.200.14"
          port: 5432
          user: "postgres"
          password: "enflame"
          dbname: "metadata"

        minio:
          host: "172.16.90.36:9000"
          user: "minioadmin"
          password: "minioadmin"
          bucket: "rag-data"

        sources:
          wiki:
            chunk_tokens: 200
            enable_graphrag: true
          html:
            chunk_tokens: 400
          wangpan:
            chunk_tokens: 400
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("需要 PyYAML: pip install pyyaml")

    with open(path, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    # 顶层 key 映射
    args_map = {
        "source": "source",
        "limit": "limit",
        "chunk_tokens": "chunk_tokens",
        "enable_graphrag": "enable_graphrag",
        "enable_raptor": "enable_raptor",
        "no_parse": "no_parse",
        "wait_for_parse": "wait_for_parse",
        "tenant_id": "tenant_id",
        "embd_id": "embd_id",
    }
    result = {}
    for k, v in args_map.items():
        if k in cfg:
            result[v] = cfg[k]

    # PG
    pg = cfg.get("pg", {})
    if pg:
        for field in ("host", "port", "user", "password", "dbname"):
            result[f"pg_{field}"] = pg.get(field)

    # MinIO
    minio = cfg.get("minio", {})
    if minio:
        for field in ("host", "user", "password", "bucket"):
            result[f"minio_{field}"] = minio.get(field)

    # 数据源级别覆盖
    sources_cfg = cfg.get("sources", {})
    result["_sources_cfg"] = sources_cfg

    return result


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def new_uuid():
    import uuid
    return str(uuid.uuid1()).replace("-", "")


def fetch_rows(pg_config, source_table, limit=0, doc_id=0, project_filter=None, task_id=None):
    import psycopg2, psycopg2.extras
    conn = psycopg2.connect(**pg_config)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    if doc_id > 0:
        query = f"SELECT * FROM {source_table} WHERE id = %s"
        cur.execute(query, (doc_id,))
    else:
        query = f"SELECT * FROM {source_table}"
        where = []
        if task_id:
            where.append(f"task_id = '{re.sub(r'[^a-zA-Z0-9_-]', '', task_id)}'")
        if project_filter and source_table in ("wiki_metadata", "wangpan_metadata", "git_metadata"):
            if project_filter == "other":
                where.append("minio_key NOT LIKE '%4.0%' AND minio_key NOT LIKE '%4.5%' AND minio_key NOT LIKE '%5.0%'")
                if source_table == "wiki_metadata":
                    where.append("wiki_path::text NOT ILIKE '%libra%'")
            elif project_filter == "libra":
                if source_table == "wangpan_metadata":
                    where.append("(minio_key ILIKE '%/libra/%' OR minio_key ILIKE '%/libra.%')")
                    where.append("minio_key NOT ILIKE '%libra-%' AND minio_key NOT ILIKE '%libra_h%'")
                else:
                    where.append("wiki_path::text ILIKE '%libra%' AND wiki_path::text NOT ILIKE '%libra-%' AND wiki_path::text NOT ILIKE '%libra_h%'")
            elif project_filter == "libra_h":
                if source_table == "wangpan_metadata":
                    where.append("(minio_key ILIKE '%libra-%' OR minio_key ILIKE '%libra_h%')")
                else:
                    where.append("(wiki_path::text ILIKE '%libra-%' OR wiki_path::text ILIKE '%libra_h%')")
            elif project_filter == "draco":
                if source_table == "wangpan_metadata":
                    where.append("minio_key ILIKE '%draco%'")
                else:
                    where.append("wiki_path::text ILIKE '%draco%'")
            else:
                ver = project_filter.replace("SIP", "")
                where.append(f"project_name = '{ver}'")
                where.append("status IN ('processed','uploaded')")
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY id"
        if limit > 0:
            query += f" LIMIT {limit}"
        cur.execute(query)
    rows = cur.fetchall()
    cur.close(); conn.close()
    return rows


def log_unmatched_pg(pg_config, source_table, matched_minio_keys, log_file):
    """将未匹配的 PG 行写入 log 文件"""
    import psycopg2
    conn = psycopg2.connect(**pg_config)
    cur = conn.cursor()
    cur.execute(f"SELECT minio_key FROM {source_table} WHERE minio_key NOT IN %s ORDER BY minio_key",
                (tuple(matched_minio_keys) if matched_minio_keys else ('',)))
    with open(log_file, 'w') as f:
        for (mk,) in cur:
            f.write(mk + "\n")
    cur.close(); conn.close()


def infer_suffix(minio_key):
    ext = minio_key.rsplit('.', 1)[-1].lower() if '.' in minio_key else ''
    map_suffix = {'md': 'md', 'markdown': 'md', 'pdf': 'pdf', 'docx': 'docx',
                  'xlsx': 'xlsx', 'txt': 'txt', 'csv': 'csv', 'html': 'html',
                  'json': 'json'}
    return map_suffix.get(ext, 'txt')


def get_minio_size(minio_config, bucket, key):
    from minio import Minio
    client = Minio(minio_config["host"], access_key=minio_config["user"],
                   secret_key=minio_config["password"],
                   secure=minio_config.get("secure", False))
    try:
        return client.stat_object(bucket, key).size
    except Exception:
        try:
            obj = client.get_object(bucket, key)
            data = obj.read(); obj.close(); obj.release_conn()
            return len(data)
        except Exception:
            return 0


# create 路径 (POST /api/v1/datasets) 用 exclude_unset=False 做 model_dump,
# 会把 ParserConfig 里这些带默认值的字段(多为 None/空)也 dump 进 DB。
# 其中 table_column_mode/names/roles 值为 null,撞上前端 zod 的 .optional()
# (只接受 undefined 不接受 null),导致浏览器里 KB 点保存时 form.trigger() 静默失败。
# pan_dev/wiki_dev 无这些字段(前端保存走 update 路径 exclude_unset=True),故能正常保存。
_PARSER_CONFIG_LEAK_KEYS = (
    "table_column_mode",
    "table_column_names",
    "table_column_roles",
    "pages",
    "task_page_size",
    "filename_embd_weight",
    "ext",
    "tag_kb_ids",
    "compilation_template_group_id",
)


def _strip_parser_config_leaks(kb):
    """删除 create 路径泄漏进 parser_config 的 ParserConfig 默认值字段,使 KB 与前端保存的 pan_dev 对齐。"""
    from api.db.services.knowledgebase_service import KnowledgebaseService

    pc = kb.parser_config or {}
    if not pc:
        return kb
    changed = False
    for key in _PARSER_CONFIG_LEAK_KEYS:
        if key in pc:
            pc.pop(key)
            changed = True
    if changed:
        KnowledgebaseService.update_by_id(kb.id, {"parser_config": pc})
        kb.parser_config = pc
    return kb


def build_parser_config(chunk_token_num, graphrag_cfg=None, raptor_cfg=None, extra_cfg=None):
    """构建 parser_config。

    extra_cfg 可包含: delimiter, overlapped_percent, auto_keywords,
    auto_questions, toc_extraction, enable_metadata, etc.
    """
    if extra_cfg is None:
        extra_cfg = {}

    cfg = {
        "chunk_token_num": chunk_token_num,
        "delimiter": extra_cfg.get("delimiter", "\n"),
    }

    # 子文本块
    if extra_cfg.get("enable_children"):
        cfg["enable_children"] = True
        cfg["children_delimiter"] = extra_cfg.get("children_delimiter", "\n")

    # 智能提取
    for k in ("auto_keywords", "auto_questions", "toc_extraction"):
        if extra_cfg.get(k):
            cfg[k] = extra_cfg[k]

    # 自动元数据
    if extra_cfg.get("enable_metadata"):
        cfg["enable_metadata"] = True
        cfg["metadata"] = extra_cfg.get("metadata", [])
        cfg["built_in_metadata"] = extra_cfg.get("built_in_metadata", [])

    # GraphRAG
    if graphrag_cfg:
        cfg["graphrag"] = {
            "userag": True,
            "entity_types": graphrag_cfg.get("entity_types", ["organization", "person", "geo", "event", "category"]),
            "method": graphrag_cfg.get("method", "light"),
            "batch_chunk_token_size": graphrag_cfg.get("batch_chunk_token_size", 4096),
        }
        # 可选高级参数
        for k in ("resolution", "community"):
            cfg["graphrag"][k] = graphrag_cfg.get(k, False)
        for k in ("retry_attempts", "retry_backoff_seconds", "retry_backoff_max_seconds",
                   "build_subgraph_timeout_per_chunk_seconds", "build_subgraph_min_timeout_seconds",
                   "merge_timeout_seconds", "resolution_timeout_seconds",
                   "community_timeout_seconds", "lock_acquire_timeout_seconds"):
            if k in graphrag_cfg:
                cfg["graphrag"][k] = graphrag_cfg[k]

    # RAPTOR
    if raptor_cfg:
        cfg["raptor"] = {
            "use_raptor": True,
            "prompt": raptor_cfg.get("prompt",
                "Please summarize the following paragraphs. Be careful with the numbers, do not make things up. "
                "Paragraphs as following:\n      {cluster_content}\n"
                "The above is the content you need to summarize."),
            "max_token": raptor_cfg.get("max_token", 256),
            "threshold": raptor_cfg.get("threshold", 0.1),
            "max_cluster": raptor_cfg("max_cluster", 64),
            "random_seed": raptor_cfg.get("random_seed", 0),
            "scope": raptor_cfg.get("scope", "file"),
            "clustering_method": raptor_cfg.get("clustering_method", "gmm"),
            "tree_builder": raptor_cfg.get("tree_builder", "raptor"),
        }
    return cfg


# ══════════════════════════════════════════════════════════════════════
# RAGFlow 操作
# ══════════════════════════════════════════════════════════════════════

def ensure_kb(tenant_id, kb_name, embd_id, chunk_token_num, graphrag_cfg=None, raptor_cfg=None):
    """通过 RAGFlow REST API 创建/获取知识库，确保前端兼容"""
    from api.db.db_models import Tenant
    from api.db.services.knowledgebase_service import KnowledgebaseService

    kb = KnowledgebaseService.query(name=kb_name, tenant_id=tenant_id)
    if kb:
        return kb[0]

    tenant = Tenant.get_by_id(tenant_id)
    embd = (embd_id or tenant.embd_id).replace("___", "@")

    payload = {
        "name": kb_name,
        "embedding_model": embd,
        "chunk_method": "naive",
        "parser_config": build_parser_config(chunk_token_num, graphrag_cfg, raptor_cfg),
    }

    import requests
    api_url = "http://127.0.0.1/api/v1/datasets"
    api_key = os.environ.get("RAGFLOW_API_KEY", "ragflow-307044760fae4f548209426ba6191d9e")
    resp = requests.post(api_url, json=payload,
                         headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    data = resp.json()
    if data.get("code") != 0:
        raise Exception(f"API create KB failed: {data.get('message', resp.text)}")
    kb_id = data["data"]["id"]
    kb = KnowledgebaseService.get_by_id(kb_id)[1]
    # create 路径会把 ParserConfig 默认值字段(null)泄漏进 parser_config,
    # 导致前端保存时 zod .optional() 校验失败。创建后清理,与前端保存的 KB 对齐。
    return _strip_parser_config_leaks(kb)


def insert_one_document(kb_id, tenant_id, row, chunk_token_num,
                        graphrag_cfg=None, raptor_cfg=None, minio_config=None,
                        meta=None, write_meta=True):
    from api.db.services.document_service import DocumentService
    from api.db.services.file_service import FileService
    from api.db.services.file2document_service import File2DocumentService

    if minio_config is None:
        minio_config = MINIO_DEFAULTS

    bucket = minio_config.get("bucket", MINIO_DEFAULTS["bucket"])
    minio_key = row['minio_key']
    title = row.get('title') or minio_key.rsplit('/', 1)[-1]
    file_hash = row.get('file_hash', '')
    suffix = infer_suffix(minio_key)
    # 用 PG local_size 避免每篇一次 MinIO HEAD 网络往返(39259 篇时是主要开销)
    size = row.get("local_size") or get_minio_size(minio_config, bucket, minio_key)

    name = title[:250]
    if suffix and not name.endswith('.' + suffix):
        name = name[:250 - len(suffix) - 1] + '.' + suffix

    doc_id = new_uuid()
    doc = {
        "id": doc_id, "kb_id": kb_id,
        "parser_id": "naive", "pipeline_id": "",
        "parser_config": build_parser_config(chunk_token_num, graphrag_cfg, raptor_cfg),
        "source_type": "local", "type": suffix,
        "created_by": tenant_id, "name": name,
        "location": minio_key, "size": size,
        "suffix": suffix, "content_hash": file_hash,
    }
    DocumentService.insert(doc)

    # 创建 File + File2Document，parent_id 指向外部 MinIO 的 bucket 名
    # 这样 get_storage_address 返回 (bucket, minio_key)
    file_id = new_uuid()
    FileService.save(**{
        "id": file_id,
        "parent_id": bucket,
        "tenant_id": tenant_id,
        "created_by": tenant_id,
        "name": name,
        "type": suffix,
        "size": size,
        "location": minio_key,
        "source_type": "",  # LOCAL
    })
    File2DocumentService.save(**{
        "id": new_uuid(),
        "file_id": file_id,
        "document_id": doc_id,
    })

    # 写入文档级元数据 (doc_meta ES 索引)，供检索 enrich 附加到每个 chunk
    # write_meta=False 时跳过(39259 篇批量导入提速用，metadata 可后续补写)
    if write_meta and meta:
        try:
            from api.db.services.doc_metadata_service import DocMetadataService
            DocMetadataService.update_document_metadata(doc_id, meta)
        except Exception as e:
            print(f"  [META] 写元数据失败 {name[:50]}: {e}")

    return doc_id


def queue_parse(tenant_id, kb_id, doc_ids):
    from api.db.db_models import Document
    from api.db.services.file2document_service import File2DocumentService
    from api.db.services.task_service import queue_tasks

    for doc_id in doc_ids:
        doc = Document.get_by_id(doc_id)
        bucket, name = File2DocumentService.get_storage_address(doc_id=doc_id)
        task_doc = {
            "id": doc_id, "kb_id": kb_id, "tenant_id": tenant_id,
            "parser_id": doc.parser_id or "naive", "pipeline_id": doc.pipeline_id or "",
            "parser_config": doc.parser_config or {},
            "name": doc.name, "location": doc.location,
            "size": doc.size, "type": doc.type or "txt",
            "source_type": doc.source_type or "local",
        }
        queue_tasks(task_doc, bucket, name, 0)


def queue_graph_tasks(tenant_id, kb_id, doc_ids, with_graphrag, with_raptor):
    from api.db.services.document_service import queue_raptor_o_graphrag_tasks
    from api.db.services.task_service import GRAPH_RAPTOR_FAKE_DOC_ID

    if not doc_ids:
        return [], []

    sample_doc = {"id": doc_ids[0]}
    graphrag_id = None
    raptor_id = None

    if with_graphrag:
        try:
            graphrag_id = queue_raptor_o_graphrag_tasks(sample_doc, "graphrag", 0, doc_ids=doc_ids,
                                                        fake_doc_id=GRAPH_RAPTOR_FAKE_DOC_ID)
            print(f"  GraphRAG queued ({len(doc_ids)} docs) task_id={graphrag_id}")
        except Exception as e:
            print(f"  GraphRAG: FAIL: {e}")

    if with_raptor:
        try:
            raptor_id = queue_raptor_o_graphrag_tasks(sample_doc, "raptor", 0, doc_ids=doc_ids,
                                                      fake_doc_id=GRAPH_RAPTOR_FAKE_DOC_ID)
            print(f"  RAPTOR queued ({len(doc_ids)} docs) task_id={raptor_id}")
        except Exception as e:
            print(f"  RAPTOR: FAIL: {e}")

    return graphrag_id, raptor_id


def wait_for_graph_tasks(kb_id, graphrag_id, raptor_id, timeout=7200, poll_interval=15):
    """等待 GraphRAG/RAPTOR 任务完成。

    通过检查 KB 的 graphrag_task_finish_at / raptor_task_finish_at 判断完成状态。
    """
    from api.db.db_models import Knowledgebase

    started = time.time()
    waited = set()
    if graphrag_id:
        waited.add("graphrag")
    if raptor_id:
        waited.add("raptor")

    if not waited:
        return

    print(f"  waiting for graph tasks ({', '.join(waited)})...")
    while waited:
        kb = Knowledgebase.get_by_id(kb_id)
        elapsed = time.time() - started

        if "graphrag" in waited and kb.graphrag_task_finish_at:
            print(f"\nGraphRAG done ({elapsed:.0f}s)")
            waited.discard("graphrag")

        if "raptor" in waited and kb.raptor_task_finish_at:
            print(f"\nRAPTOR done ({elapsed:.0f}s)")
            waited.discard("raptor")

        if not waited:
            break

        if elapsed > timeout:
            print(f"\n    timeout ({timeout}s)，{' '.join(waited)} pending")
            break

        remaining = ', '.join(waited)
        print(f"\r  {remaining} generating... ({elapsed:.0f}s elapsed)", end="", flush=True)
        time.sleep(poll_interval)


def wait_for_parse(kb_id, doc_ids, timeout=3600, poll_interval=10):
    """等待所有文档向量化完成。

    轮询 KB 下的文档状态，直到全部为 done(3) 或 fail(4) 或timeout。
    返回 (done_count, fail_count, timed_out)。
    """
    from api.db.db_models import Document

    target_ids = set(doc_ids)
    started = time.time()

    while True:
        docs = list(Document.select().where(
            Document.kb_id == kb_id,
            Document.id.in_(list(target_ids))
        ))
        if not docs:
            time.sleep(poll_interval)
            continue

        total = len(docs)
        done = sum(1 for d in docs if d.progress >= 1.0)
        fail = sum(1 for d in docs if d.progress < 0)
        processing = total - done - fail
        elapsed = time.time() - started

        print(f"\r  progress: done={done} fail={fail} pending={processing}/{total} ({elapsed:.0f}s)", end="", flush=True)

        if processing == 0:
            print()  # newline
            return done, fail, False

        if elapsed > timeout:
            print(f"\n    timeout ({timeout}s)，remaining {processing} 篇pending")
            return done, fail, True

        time.sleep(poll_interval)


def show_all_status(tenant_id):
    from api.db.db_models import Document
    from api.db.services.knowledgebase_service import KnowledgebaseService

    for source_key, cfg in SOURCE_CONFIGS.items():
        kb = KnowledgebaseService.query(name=cfg["kb_name"], tenant_id=tenant_id)
        if not kb:
            print(f"\n{cfg['kb_name']} ({cfg['description']}) — not created")
            continue
        kb = kb[0]
        docs = list(Document.select().where(Document.kb_id == kb.id))
        by_run = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}
        for d in docs:
            by_run[d.run] = by_run.get(d.run, 0) + 1
        run_map = {"0": "pending", "1": "running", "2": "paused", "3": "done", "4": "FAIL"}
        total_chunks = sum(d.chunk_num or 0 for d in docs)
        total_tokens = sum(d.token_num or 0 for d in docs)
        print(f"\n{kb.name} ({cfg['description']})")
        print(f"  docs={len(docs)} chunks={total_chunks} tokens={total_tokens}")
        print(f"  status:", end="")
        for run in sorted(by_run.keys()):
            if by_run[run]:
                print(f" {run_map.get(run, run)}={by_run[run]}", end="")
        print()
        failed = [d for d in docs if d.run == "4"]
        if failed:
            print(f"   FAIL: {len(failed)} docs")
            for d in failed[:5]:
                print(f"      - {d.name[:70]}")


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def import_source(source_key, tenant_id, args):
    """导入单个数据源"""
    # 数据源级别配置覆盖
    sources_cfg = getattr(args, '_sources_cfg', None) or {}
    src_cfg = sources_cfg.get(source_key, {})

    cfg = SOURCE_CONFIGS[source_key]
    chunk_tokens = (getattr(args, 'chunk_tokens', None) or
                    src_cfg.get('chunk_tokens') or
                    cfg["default_chunk_tokens"])
    enable_graphrag = getattr(args, 'enable_graphrag', False) or src_cfg.get('enable_graphrag', False)
    enable_raptor = getattr(args, 'enable_raptor', False) or src_cfg.get('enable_raptor', False)

    # GraphRAG / RAPTOR 详细配置 (从 YAML)
    graphrag_cfg = src_cfg.get('graphrag', None) if enable_graphrag else None
    raptor_cfg = src_cfg.get('raptor', None) if enable_raptor else None

    # MinIO config (支持命令行 / YAML 覆盖)
    minio_config = {
        "host": getattr(args, 'minio_host', None) or MINIO_DEFAULTS["host"],
        "user": getattr(args, 'minio_user', None) or MINIO_DEFAULTS["user"],
        "password": getattr(args, 'minio_password', None) or MINIO_DEFAULTS["password"],
        "bucket": getattr(args, 'minio_bucket', None) or MINIO_DEFAULTS["bucket"],
    }

    # Step 1: KB
    kb_name = cfg["kb_name"]
    if getattr(args, 'kb_name', None):
        kb_name = args.kb_name
    elif getattr(args, 'project', None):
        if args.project == "other":
            kb_name = "ekb_wiki" if source_key == "wiki" else "ekb_pan"
        else:
            ver = args.project.replace("SIP", "")
            suffix = {"wiki": "wiki", "wangpan": "pan", "html": "docs",
                      "git": "git", "local": "register_html"}.get(source_key, source_key)
            kb_name = f"ekb_{ver}_{suffix}"
    if getattr(args, 'dev', False):
        if not kb_name.endswith("_dev"):
            kb_name = kb_name + "_dev"
    print(f"\nKB '{kb_name}'")
    kb = ensure_kb(tenant_id, kb_name, args.embd_id, chunk_tokens,
                   graphrag_cfg, raptor_cfg)
    flags = " + ".join(filter(None, [
        "向量", "GraphRAG" if enable_graphrag else None,
        "RAPTOR" if enable_raptor else None
    ]))
    print(f"  KB: {kb.id} [{flags}]")

    # Step 2: 外部数据
    pg_config = {
        "host": args.pg_host or PG_DEFAULTS["host"],
        "port": args.pg_port or PG_DEFAULTS["port"],
        "user": args.pg_user or PG_DEFAULTS["user"],
        "password": args.pg_password or PG_DEFAULTS["password"],
        "dbname": args.pg_db or PG_DEFAULTS["dbname"],
    }
    rows = fetch_rows(pg_config, cfg["source_table"], args.limit, args.doc_id,
                      project_filter=getattr(args, 'project', None),
                      task_id=getattr(args, 'task_id', None))
    if not rows and source_key in ("wiki", "wangpan"):
        # Fallback: SIP4.0 → libra, SIP4.5 → libra_h, SIP5.0 → draco
        fb_filters = {"SIP4.0": "libra", "SIP4.5": "libra_h", "SIP5.0": "draco"}
        fb = fb_filters.get(getattr(args, 'project', ''))
        if fb:
            print(f"  project filter 无结果，fallback 到 {fb}")
            rows = fetch_rows(pg_config, cfg["source_table"], args.limit, args.doc_id,
                              project_filter=fb)
    print(f"\n{cfg['source_table']}: {len(rows)} 行")
    if not rows:
        print(f"  无数据，跳过\n")
        return kb.id, 0

    # wangpan / local 源：MinIO 实际路径是 PG 原始 minio_key
    # file_hash 用 local_md5 做变更检测
    if source_key in ("wangpan", "local"):
        for r in rows:
            if r.get("local_md5"):
                r["file_hash"] = r["local_md5"]

    # Step 3: 增量对比 & 导入
    print(f"\n增量分析...")
    from api.db.db_models import Document

    # 外部数据按 minio_key 建索引
    ext_map = {r['minio_key']: r for r in rows}

    # RAGFlow 已有文档按 location(=minio_key) 建索引
    existing = list(Document.select().where(Document.kb_id == kb.id))
    existing_map = {d.location: d for d in existing}

    new_rows = [(k, ext_map[k]) for k in ext_map if k not in existing_map]
    changed_rows = []
    unchanged = 0
    for k, ext_row in ext_map.items():
        if k in existing_map:
            old = existing_map[k]
            if old.content_hash != ext_row.get('file_hash', ''):
                changed_rows.append((k, ext_row, old))
            else:
                unchanged += 1

    deleted_docs = [existing_map[k] for k in existing_map if k not in ext_map]

    print(f"  外部: {len(rows)} | 已有: {len(existing)}")
    print(f"  新增: {len(new_rows)} | 变更: {len(changed_rows)} | 未变: {unchanged} | 删除: {len(deleted_docs)}")

    # MinIO client 复用（读首行解析 EKB 标记）
    minio_client = Minio(minio_config["host"], access_key=minio_config["user"],
                         secret_key=minio_config["password"],
                         secure=minio_config.get("secure", False))
    bucket = minio_config.get("bucket", MINIO_DEFAULTS["bucket"])
    # 主 project 以 PG project_name / --project 为基准；不取标记里的第一个
    pg_project = getattr(args, 'project', None) or "4.5"
    ekb_anomaly_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ekb_anomalies.log")

    # 删除已不存在于外部的文档 (仅在无 limit 全量同步且非 task_id 过滤时)
    if deleted_docs and args.limit == 0 and not getattr(args, 'task_id', None):
        print(f"\n  删除 {len(deleted_docs)} 篇...")
        for d in deleted_docs:
            try:
                Document.delete_by_id(d.id)
                print(f"    - {d.name[:50]}")
            except Exception as e:
                print(f"    - FAIL: {d.name[:50]}: {e}")

    # 更新已变更的文档
    if changed_rows:
        print(f"\n  更新 {len(changed_rows)} 篇...")
        from api.db.services.file2document_service import File2DocumentService
        from api.db.services.task_service import queue_tasks
        for _k, ext_row, old_doc in changed_rows:
            new_hash = ext_row.get('file_hash', '')
            Document.update(
                content_hash=new_hash,
                run="0",  # 重置为待处理
                chunk_num=0,
                token_num=0,
            ).where(Document.id == old_doc.id).execute()
            print(f"    ~ {old_doc.name[:50]}")
            # 重建文档级元数据（覆盖 project 关联，含新解析的 EKB 标记）
            try:
                _tags = extract_ekb_tags(minio_client, bucket, ext_row['minio_key'])
                if _tags and pg_project not in _tags:
                    log_ekb_anomaly(ext_row, pg_project, _tags, ekb_anomaly_log, source_key)
                    print(f"  [EKBTAG] 更新跳过元数据 {old_doc.name[:50]} "
                          f"(PG={pg_project}, 标记={_tags}) → 见 ekb_anomalies.log")
                    continue
                _meta = enrich_project_meta(build_full_meta(ext_row, source_key, pg_project),
                                            _tags, pg_project)
                from api.db.services.doc_metadata_service import DocMetadataService
                DocMetadataService.update_document_metadata(old_doc.id, _meta)
            except Exception as _e:
                print(f"  [META] 更新元数据失败 {old_doc.name[:50]}: {_e}")

    # 导入新文档
    print(f"\n导入 {len(new_rows)} 篇...")

    doc_ids = []
    for _k, row in new_rows:
        try:
            # 解析文档第一行 EKB 标记 (仅 md；xlsx 跳过走兜底)
            tags = extract_ekb_tags(minio_client, bucket, row['minio_key'])
            if tags and pg_project not in tags:
                # 有标记但不含主 project → 异常，记录并跳过（不静默入库）
                log_ekb_anomaly(row, pg_project, tags, ekb_anomaly_log, source_key)
                print(f"  [EKBTAG] 跳过 {row.get('title') or row.get('wangpan_file_name', '')[:50]} "
                      f"(PG={pg_project}, 标记={tags}) → 见 ekb_anomalies.log")
                continue

            # 组装完整元数据并注入 project 关联
            meta = build_full_meta(row, source_key, pg_project)
            meta = enrich_project_meta(meta, tags, pg_project)

            doc_id = insert_one_document(kb.id, tenant_id, row, chunk_tokens,
                                         graphrag_cfg, raptor_cfg, minio_config,
                                         meta=meta, write_meta=not getattr(args, 'no_meta', False))
            doc_ids.append(doc_id)
        except Exception as e:
            print(f"  FAIL: {str(e)[:60]}")

    # 合并新旧 doc_ids（新增 + 变更的都要重新解析）
    changed_ids = [old.id for _, _, old in changed_rows]
    doc_ids = changed_ids + doc_ids

    if not doc_ids:
        print(f"\nno changes，跳过解析")
        return kb.id, 0

    # Step 4: 解析
    if not args.no_parse and doc_ids:
        print(f"\nparsing {len(doc_ids)} docs...")
        queue_parse(tenant_id, kb.id, doc_ids)

        # 等待向量化完成
        if args.wait_for_parse:
            print(f"  waiting (timeout={args.wait_timeout}s)")
            done, fail, timed_out = wait_for_parse(
                kb.id, doc_ids,
                timeout=args.wait_timeout,
                poll_interval=getattr(args, 'wait_interval', 10)
            )
            if timed_out:
                print(f"  timeout: {done} done, {fail} fail, {processing} pending")
            else:
                print(f"  parsing done: {done} done, {fail} fail")
                # 自动重试失败文档
                if fail > 0 and getattr(args, 'auto_retry', False):
                    print(f"\n自动重试 {fail} 篇失败文档...")
                    retry_failed_docs(kb.id, tenant_id)

    # Step 5: 知识图谱
    if (enable_graphrag or enable_raptor) and doc_ids:
        print(f"\n触发知识图谱 ({len(doc_ids)} 篇)...")
        g_id, r_id = queue_graph_tasks(tenant_id, kb.id, doc_ids,
                                       enable_graphrag, enable_raptor)

        # 等待图谱生成完成
        if args.wait_for_parse and (g_id or r_id):
            wait_for_graph_tasks(kb.id, g_id, r_id,
                                 timeout=args.wait_timeout,
                                 poll_interval=getattr(args, 'wait_interval', 15))

    return kb.id, len(doc_ids)


def retry_failed_docs(kb_id, tenant_id):
    """重试 KB 下所有失败的文档"""
    from api.db.db_models import Document
    from api.db.services.file2document_service import File2DocumentService
    from api.db.services.task_service import queue_tasks

    failed = list(Document.select().where(
        Document.kb_id == kb_id,
        Document.run == "4"
    ))

    if not failed:
        print("  没有失败文档")
        return

    # 重置为 pending
    for d in failed:
        Document.update(run="0").where(Document.id == d.id).execute()

    # 重新入队
    for d in failed:
        bucket, name = File2DocumentService.get_storage_address(doc_id=d.id)
        task_doc = {
            "id": d.id, "kb_id": d.kb_id, "tenant_id": tenant_id,
            "parser_id": d.parser_id or "naive", "pipeline_id": d.pipeline_id or "",
            "parser_config": d.parser_config or {},
            "name": d.name, "location": d.location,
            "size": d.size, "type": d.type or "txt",
            "source_type": d.source_type or "local",
        }
        queue_tasks(task_doc, bucket, name, 0)

    print(f"  已重新入队 {len(failed)} 篇")

    # 再次等待
    doc_ids = [d.id for d in failed]
    done, fail, _ = wait_for_parse(kb_id, doc_ids, timeout=3600)
    print(f"\n[OK] 重试结果: {done} done, {fail} fail")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RAGFlow 批量导入工具 — 按数据源 (wiki/html/wangpan) 分别创建KB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用配置文件
  python3 batch_import.py --config batch_config.yaml
  python3 batch_import.py -c batch_config.yaml

  # 命令行
  python3 batch_import.py --source wiki --limit 10
  python3 batch_import.py --source html --enable-graphrag --enable-raptor --wait

  # 等待向量化完成后再退出
  python3 batch_import.py --source wiki --wait

  # 查看状态
  python3 batch_import.py --status
        """)

    # 配置文件
    parser.add_argument("--config", "-c", default=None,
                        help="YAML 配置文件路径")

    parser.add_argument("--source", default="wiki",
                        choices=["wiki", "html", "wangpan", "git", "local", "all"],
                        help="数据源 (默认: wiki)")
    parser.add_argument("--project", default=None,
                        choices=["4.0", "4.5", "5.0"],
                        help="按版本号过滤数据")
    parser.add_argument("--task-id", default=None,
                        help="按 task_id 过滤 (local_metadata 源)")
    parser.add_argument("--limit", type=int, default=0,
                        help="每个数据源的导入上限 (0=全部)")
    parser.add_argument("--doc-id", type=int, default=0,
                        help="只导入指定 PG ID 的文档（用于单篇测试）")
    parser.add_argument("--status", action="store_true",
                        help="查看所有KB状态")
    parser.add_argument("--kb-name", default=None,
                        help="覆盖自动生成的 KB 名")
    parser.add_argument("--dev", action="store_true",
                        help="导入到 _dev 后缀的 KB（如 ekb_4.5_wiki_dev）")

    # RAGFlow
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--embd-id", default=None)

    # 外部连接 (可选，默认使用内置配置)
    parser.add_argument("--pg-host")
    parser.add_argument("--pg-port", type=int)
    parser.add_argument("--pg-user")
    parser.add_argument("--pg-password")
    parser.add_argument("--pg-db")
    parser.add_argument("--minio-host")
    parser.add_argument("--minio-user")
    parser.add_argument("--minio-password")
    parser.add_argument("--minio-bucket")

    # 管道开关
    parser.add_argument("--enable-graphrag", action="store_true",
                        help="启用 GraphRAG 知识图谱")
    parser.add_argument("--enable-raptor", action="store_true",
                        help="启用 RAPTOR 层级摘要")
    parser.add_argument("--no-parse", action="store_true",
                        help="不触发分块+向量化")
    parser.add_argument("--no-meta", action="store_true",
                        help="跳过每篇文档级元数据(ES doc_meta)写入，批量导入提速用")
    parser.add_argument("--chunk-tokens", type=int,
                        help="覆盖默认 chunk token 数")

    # 等待 & 重试
    parser.add_argument("--wait", "--wait-for-parse", dest="wait_for_parse",
                        action="store_true",
                        help="等待向量化完成后再退出")
    parser.add_argument("--wait-timeout", type=int, default=7200,
                        help="等待timeout秒数 (默认 7200)")
    parser.add_argument("--wait-interval", type=int, default=10,
                        help="轮询间隔秒数 (默认 10)")
    parser.add_argument("--auto-retry", action="store_true",
                        help="向量化失败后自动重试一次")

    args = parser.parse_args()

    # ── 加载 YAML 配置文件 ─────────────────────────────────────
    if args.config:
        yaml_cfg = load_yaml_config(args.config)
        # YAML 值作为默认值，命令行参数覆盖 (argparse 已设置的非 None 值优先)
        for k, v in yaml_cfg.items():
            if k == "_sources_cfg":
                setattr(args, '_sources_cfg', v)
            elif getattr(args, k.replace('_', '-'), None) is None:
                setattr(args, k, v)

    # 确保 _sources_cfg 存在
    if not hasattr(args, '_sources_cfg'):
        setattr(args, '_sources_cfg', None)

    # ── 初始化 RAGFlow ───────────────────────────────────────────
    # 自动查找 RAGFlow 安装路径
    _ragflow_home = os.environ.get("RAGFLOW_HOME")
    if _ragflow_home:
        sys.path.insert(0, _ragflow_home)
    else:
        # 尝试常见路径
        for _candidate in [
            os.getcwd(),
            os.path.join(os.path.dirname(__file__), "..", ".."),
            "/ragflow",
            os.path.expanduser("~/dev/ragflow"),
        ]:
            if os.path.isdir(os.path.join(_candidate, "common")):
                sys.path.insert(0, _candidate)
                break
        else:
            print("FAIL: 找不到 RAGFlow 路径，请设置 RAGFLOW_HOME 环境变量")
            sys.exit(1)
    from common.config_utils import read_config
    from common.settings import init_settings
    from api.db.db_models import Tenant
    read_config(); init_settings()

    tenant_id = args.tenant_id or Tenant.select().first().id

    # ── 查看状态 ─────────────────────────────────────────────────
    if args.status:
        show_all_status(tenant_id)
        return

    # ── 确定数据源 ───────────────────────────────────────────────
    sources = ["wiki", "html", "wangpan", "git", "local"] if args.source == "all" else [args.source]

    # ── 逐个导入 ─────────────────────────────────────────────────
    results = {}
    for src in sources:
        print(f"\n=== {src.upper()}: {SOURCE_CONFIGS[src]['description']} ===")
        kb_id, count = import_source(src, tenant_id, args)
        results[src] = {"kb_id": kb_id, "count": count}

    # ── 汇总 ─────────────────────────────────────────────────────
    print(f"\n=== import done ===")
    total = 0
    for src, info in results.items():
        cfg = SOURCE_CONFIGS[src]
        print(f"  {cfg['kb_name']}  KB={info['kb_id']}  docs={info['count']}")
        total += info['count']
    print(f"  total: {total} docs")
    print(f"\n 监控: docker logs -f ragflow-test")


if __name__ == "__main__":
    main()
