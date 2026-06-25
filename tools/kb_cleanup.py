#!/usr/bin/env python3
"""
KB 清理工具 — 删除 MySQL 文档/任务、ES chunk、MinIO 临时文件

用法:
  python3 kb_cleanup.py --kb enflame-wiki              # 清理全部
  python3 kb_cleanup.py --kb enflame-wiki --docs-only   # 只清文档
  python3 kb_cleanup.py --kb enflame-wiki --es-only     # 只清 ES
"""

import argparse, os, sys

MYSQL_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.environ.get("MYSQL_PORT", "3306")
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASS = os.environ.get("MYSQL_PASSWORD", "infini_rag_flow")
MYSQL_DB = os.environ.get("MYSQL_DB", "rag_flow")

ES_HOST = os.environ.get("ES_HOST", "http://localhost:1200")
ES_USER = os.environ.get("ES_USER", "elastic")
ES_PASS = os.environ.get("ES_PASSWORD", "infini_rag_flow")

MINIO_HOST = os.environ.get("MINIO_HOST", "172.16.90.36:9000")
MINIO_ACCESS = os.environ.get("MINIO_ACCESS", "minioadmin")
MINIO_SECRET = os.environ.get("MINIO_SECRET", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "rag-data")


def _get_kb_id(kb_name):
    import pymysql
    conn = pymysql.connect(host=MYSQL_HOST, port=int(MYSQL_PORT), user=MYSQL_USER,
                           password=MYSQL_PASS, database=MYSQL_DB)
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM knowledgebase WHERE name=%s", (kb_name,))
        row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def _get_tenant_id(kb_id):
    import pymysql
    conn = pymysql.connect(host=MYSQL_HOST, port=int(MYSQL_PORT), user=MYSQL_USER,
                           password=MYSQL_PASS, database=MYSQL_DB)
    with conn.cursor() as cur:
        cur.execute("SELECT tenant_id FROM knowledgebase WHERE id=%s", (kb_id,))
        row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def clean_mysql(kb_id):
    import pymysql
    conn = pymysql.connect(host=MYSQL_HOST, port=int(MYSQL_PORT), user=MYSQL_USER,
                           password=MYSQL_PASS, database=MYSQL_DB)
    with conn.cursor() as cur:
        # 先删关联，再删文档（避免子查询失效）
        cur.execute("DELETE FROM task WHERE doc_id IN (SELECT id FROM document WHERE kb_id=%s)", (kb_id,))
        tasks = cur.rowcount
        cur.execute("DELETE FROM file2document WHERE document_id IN (SELECT id FROM document WHERE kb_id=%s)", (kb_id,))
        f2d = cur.rowcount
        cur.execute("DELETE FROM document WHERE kb_id=%s", (kb_id,))
        docs = cur.rowcount
        # 清理所有孤儿记录
        cur.execute("DELETE FROM file2document WHERE document_id NOT IN (SELECT id FROM document)")
        orphan_f2d = cur.rowcount
        cur.execute("DELETE FROM file WHERE id NOT IN (SELECT file_id FROM file2document)")
        orphan_files = cur.rowcount
        cur.execute("SELECT COUNT(*) FROM document WHERE kb_id=%s", (kb_id,))
        remaining = cur.fetchone()[0]
        # 修正 KB 缓存的 doc_num
        cur.execute("UPDATE knowledgebase SET doc_num=(SELECT COUNT(*) FROM document WHERE kb_id=%s) WHERE id=%s", (kb_id, kb_id))
    conn.commit()
    conn.close()
    print(f"  MySQL: {docs} docs, {f2d} links, {tasks} tasks, {orphan_f2d}+{orphan_files} orphans, {remaining} remaining")


def clean_es(kb_id):
    import requests
    tenant_id = _get_tenant_id(kb_id)
    if not tenant_id:
        print("  ES: tenant not found, skip")
        return
    idx = f"ragflow_{tenant_id}"
    url = f"{ES_HOST}/{idx}/_delete_by_query?refresh=true"
    resp = requests.post(url, json={"query": {"match_all": {}}},
                         auth=(ES_USER, ES_PASS), timeout=120)
    data = resp.json()
    deleted = data.get("deleted", 0)
    print(f"  ES ({idx}): {deleted} chunks deleted")


def clean_minio():
    from minio import Minio
    c = Minio(MINIO_HOST, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)
    count = 0
    for o in c.list_objects(MINIO_BUCKET, prefix="_vlm_enriched/", recursive=True):
        c.remove_object(MINIO_BUCKET, o.object_name)
        count += 1
    print(f"  MinIO: {count} enriched files removed")


def main():
    parser = argparse.ArgumentParser(description="RAGFlow KB Cleanup")
    parser.add_argument("--kb", required=True, help="KB name")
    parser.add_argument("--docs-only", action="store_true")
    parser.add_argument("--es-only", action="store_true")
    parser.add_argument("--minio-only", action="store_true")
    args = parser.parse_args()

    kb_id = _get_kb_id(args.kb)
    if not kb_id:
        print(f"KB '{args.kb}' not found")
        sys.exit(1)
    print(f"KB: {args.kb} (id={kb_id})")

    if args.docs_only:
        clean_mysql(kb_id)
    elif args.es_only:
        clean_es(kb_id)
    elif args.minio_only:
        clean_minio()
    else:
        clean_mysql(kb_id)
        clean_es(kb_id)
        clean_minio()
    print("Done.")


if __name__ == "__main__":
    main()
