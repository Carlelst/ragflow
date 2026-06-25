#!/usr/bin/env python3
"""
KB 运维 HTTP 服务 — 供 Dify Workflow 调用

端点:
  POST /import   {"source": "wiki", "doc_id": 0, "wait": true}
  POST /cleanup  {"kb": "enflame-wiki", "docs_only": false}
  GET  /status   {"kb": "enflame-wiki"}

启动:
  python3 kb_ops_server.py --port 8100
"""

import argparse, json, os, subprocess, threading
from flask import Flask, request, jsonify

app = Flask(__name__)

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
IMPORT_SCRIPT = os.path.join(TOOLS_DIR, "batch_import.py")
CONFIG = os.path.join(TOOLS_DIR, "batch_config.yaml")
CLEANUP_SCRIPT = os.path.join(TOOLS_DIR, "kb_cleanup.py")

_tasks: dict[str, dict] = {}
_lock = threading.Lock()


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
        cmd = ["python3", IMPORT_SCRIPT, "--config", CONFIG, "--source", source]
        if doc_id:
            cmd += ["--doc-id", str(doc_id)]
        if wait:
            cmd.append("--wait")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=TOOLS_DIR)
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


@app.route("/cleanup", methods=["POST"])
def api_cleanup():
    body = request.get_json(silent=True) or {}
    kb = body.get("kb", "enflame-wiki")
    docs_only = body.get("docs_only", False)

    cmd = ["python3", CLEANUP_SCRIPT, "--kb", kb]
    if docs_only:
        cmd.append("--docs-only")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=TOOLS_DIR)
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
        conn = pymysql.connect(host="127.0.0.1", port=3306, user="root",
                               password="infini_rag_flow", database="rag_flow")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                  SUM(CASE WHEN progress >= 1.0 THEN 1 ELSE 0 END) as done,
                  SUM(CASE WHEN progress < 0 THEN 1 ELSE 0 END) as failed,
                  SUM(CASE WHEN progress >= 0 AND progress < 1 THEN 1 ELSE 0 END) as pending,
                  COUNT(*) as total
                FROM document WHERE kb_id = (SELECT id FROM knowledgebase WHERE name=%s)
            """, (kb,))
            row = cur.fetchone()
        conn.close()
        return jsonify({"done": row[0], "failed": row[1], "pending": row[2], "total": row[3]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/tasks", methods=["GET"])
def api_tasks():
    with _lock:
        return jsonify(_tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    print(f"kb_ops_server starting on :{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
