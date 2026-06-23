#!/usr/bin/env python3
"""
RAGFlow 数据导入工具 — 从外部 PG + MinIO 批量导入文档

用法:
  # 基础导入 (仅向量索引)
  python3 import_docs.py \\
    --pg-host 10.9.200.14 --pg-password enflame \\
    --minio-host 172.16.90.36:9000 --minio-user minioadmin --minio-password minioadmin \\
    --minio-bucket rag-data \\
    --kb-name enflame-wiki --source-table wiki_metadata --limit 10

  # 开启 GraphRAG + RAPTOR
  python3 import_docs.py ... --enable-graphrag --enable-raptor

  # 只开启 RAPTOR (层级摘要)
  python3 import_docs.py ... --enable-raptor
"""

import argparse, json, random, string, sys, os
from datetime import datetime


def new_uuid():
    h = ''.join(random.choices(string.hexdigits, k=32))
    return ''.join([h[:8], h[8:12], h[12:16], h[16:20], h[20:]])


def fetch_external_rows(pg_config, source_table, limit=0):
    import psycopg2
    import psycopg2.extras
    conn = psycopg2.connect(**pg_config)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    query = f"SELECT * FROM {source_table} ORDER BY id"
    if limit > 0:
        query += f" LIMIT {limit}"
    cur.execute(query)
    rows = cur.fetchall()
    cur.close(); conn.close()
    return rows


def get_minio_size(minio_config, bucket, key):
    from minio import Minio
    client = Minio(minio_config["host"], access_key=minio_config["user"],
                   secret_key=minio_config["password"], secure=minio_config.get("secure", False))
    try:
        return client.stat_object(bucket, key).size
    except Exception:
        try:
            obj = client.get_object(bucket, key)
            data = obj.read(); obj.close(); obj.release_conn()
            return len(data)
        except Exception:
            return 0


def infer_suffix(minio_key):
    ext = minio_key.rsplit('.', 1)[-1].lower() if '.' in minio_key else ''
    return {'md': 'txt', 'markdown': 'txt', 'pdf': 'pdf', 'docx': 'docx',
            'xlsx': 'xlsx', 'txt': 'txt', 'csv': 'csv', 'html': 'html',
            'json': 'json'}.get(ext, 'txt')


def ensure_knowledge_base(tenant_id, kb_name, embd_id, chunk_token_num=256,
                          enable_graphrag=False, enable_raptor=False):
    from api.db.db_models import Tenant
    from api.db.services.knowledgebase_service import KnowledgebaseService
    from common.constants import ParserType

    kb = KnowledgebaseService.query(name=kb_name, tenant_id=tenant_id)
    if kb:
        print(f"  KB exists: {kb[0].id}")
        return kb[0].id

    kb_id = new_uuid()
    tenant = Tenant.get_by_id(tenant_id)
    KnowledgebaseService.save(**{
        "id": kb_id, "tenant_id": tenant_id, "name": kb_name,
        "created_by": tenant_id,
        "embd_id": embd_id or tenant.embd_id,
        "parser_id": ParserType.NAIVE.value,
        "parser_config": _build_parser_config(chunk_token_num, enable_graphrag, enable_raptor),
        "pipeline_id": "",
    })
    flags = []
    if enable_graphrag: flags.append("GraphRAG")
    if enable_raptor: flags.append("RAPTOR")
    flag_str = " + " + ", ".join(flags) if flags else ""
    print(f"  KB created: {kb_id}{flag_str}")
    return kb_id


def _build_parser_config(chunk_token_num, enable_graphrag=False, enable_raptor=False):
    cfg = {
        "chunk_token_num": chunk_token_num,
        "delimiter": "\n",
        "lapped_percent": 10,
    }
    if enable_graphrag:
        cfg["graphrag"] = True
    if enable_raptor:
        cfg["raptor"] = True
    return cfg


def insert_document(kb_id, tenant_id, row, minio_config, chunk_token_num=256,
                    enable_graphrag=False, enable_raptor=False):
    from api.db.db_models import Document
    from api.db.services.document_service import DocumentService

    minio_key = row['minio_key']
    title = row.get('title') or minio_key.rsplit('/', 1)[-1]
    file_hash = row.get('file_hash', '')
    suffix = infer_suffix(minio_key)
    bucket = minio_config.get("bucket", "rag-data")
    size = get_minio_size(minio_config, bucket, minio_key)

    name = title[:250]
    if suffix and not name.endswith('.' + suffix):
        name = name[:250 - len(suffix) - 1] + '.' + suffix

    doc = {
        "id": new_uuid(), "kb_id": kb_id,
        "parser_id": "naive", "pipeline_id": "",
        "parser_config": _build_parser_config(chunk_token_num, enable_graphrag, enable_raptor),
        "source_type": "local", "type": suffix,
        "created_by": tenant_id, "name": name,
        "location": minio_key, "size": size,
        "suffix": suffix, "content_hash": file_hash,
    }
    DocumentService.insert(doc)
    return doc["id"]


def queue_parsing(tenant_id, kb_id, doc_ids):
    from api.db.db_models import Document
    from api.db.services.file2document_service import File2DocumentService
    from api.db.services.task_service import queue_tasks

    for doc_id in doc_ids:
        doc = Document.get_by_id(doc_id)
        bucket, name = File2DocumentService.get_storage_address(doc_id=doc_id)
        task_doc = {
            "id": doc_id, "kb_id": kb_id, "tenant_id": tenant_id,
            "parser_id": doc.parser_id or "naive",
            "pipeline_id": doc.pipeline_id or "",
            "parser_config": doc.parser_config or {},
            "name": doc.name, "location": doc.location,
            "size": doc.size, "type": doc.type or "txt",
            "source_type": doc.source_type or "local",
        }
        queue_tasks(task_doc, bucket, name, 0)


def queue_graphrag(tenant_id, kb_id, doc_ids, with_graphrag=False, with_raptor=False):
    from api.db.db_models import Document
    from api.db.services.document_service import queue_raptor_o_graphrag_tasks

    if not doc_ids:
        return

    sample_doc = {"id": doc_ids[0]}

    if with_graphrag:
        try:
            queue_raptor_o_graphrag_tasks(sample_doc, "graphrag", 0, doc_ids=doc_ids)
            print(f"  GraphRAG queued ({len(doc_ids)} docs)")
        except Exception as e:
            print(f"  GraphRAG FAILED: {e}")

    if with_raptor:
        try:
            queue_raptor_o_graphrag_tasks(sample_doc, "raptor", 0, doc_ids=doc_ids)
            print(f"  RAPTOR  queued ({len(doc_ids)} docs)")
        except Exception as e:
            print(f"  RAPTOR  FAILED: {e}")


# ── 主入口 ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RAGFlow 数据导入工具 — 从外部 PG + MinIO 批量导入文档")
    # 外部 PG
    parser.add_argument("--pg-host", required=True)
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--pg-password", required=True)
    parser.add_argument("--pg-db", default="metadata")
    # 外部 MinIO
    parser.add_argument("--minio-host", required=True)
    parser.add_argument("--minio-user", required=True)
    parser.add_argument("--minio-password", required=True)
    parser.add_argument("--minio-bucket", default="rag-data")
    # RAGFlow
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--kb-name", required=True)
    parser.add_argument("--embd-id", default=None)
    # 数据源
    parser.add_argument("--source-table", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    # 管道开关 — 默认全开
    parser.add_argument("--no-parse", action="store_true",
                        help="不触发解析 (分块+向量化)")
    parser.add_argument("--no-graphrag", action="store_true",
                        help="不触发 GraphRAG")
    parser.add_argument("--no-raptor", action="store_true",
                        help="不触发 RAPTOR")
    parser.add_argument("--graphrag-only", action="store_true",
                        help="图谱部分仅触发 GraphRAG (不触发 RAPTOR)")
    parser.add_argument("--raptor-only", action="store_true",
                        help="图谱部分仅触发 RAPTOR (不触发 GraphRAG)")

    args = parser.parse_args()

    # 图谱开关逻辑
    enable_graphrag = not args.no_graphrag and not args.raptor_only
    enable_raptor = not args.no_raptor and not args.graphrag_only

    # ── 初始化 ───────────────────────────────────────────────────
    _ragflow_home = os.environ.get("RAGFLOW_HOME")
    if _ragflow_home:
        sys.path.insert(0, _ragflow_home)
    else:
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
            print("❌ 找不到 RAGFlow 路径，请设置 RAGFLOW_HOME 环境变量")
            sys.exit(1)
    from common.config_utils import read_config
    from common.settings import init_settings
    from api.db.db_models import Tenant
    read_config(); init_settings()

    if args.tenant_id:
        tenant_id = args.tenant_id
    else:
        tenant_id = Tenant.select().first().id
    print(f"📦 Tenant: {tenant_id}")

    # ── Step 1: KB ───────────────────────────────────────────────
    print(f"\n🔧 Step 1: 知识库 '{args.kb_name}'")
    kb_id = ensure_knowledge_base(tenant_id, args.kb_name, args.embd_id,
                                  args.chunk_tokens, enable_graphrag, enable_raptor)

    # ── Step 2: 外部数据 ─────────────────────────────────────────
    print(f"\n📥 Step 2: 从外部 PG 读取 {args.source_table}")
    pg_config = {"host": args.pg_host, "port": args.pg_port, "user": args.pg_user,
                 "password": args.pg_password, "dbname": args.pg_db}
    rows = fetch_external_rows(pg_config, args.source_table, args.limit)
    print(f"  Fetched {len(rows)} rows")
    if not rows:
        print("  No data, exiting.")
        return

    # ── Step 3: 导入文档 ─────────────────────────────────────────
    print(f"\n📝 Step 3: 导入文档")
    minio_config = {"host": args.minio_host, "user": args.minio_user,
                    "password": args.minio_password, "bucket": args.minio_bucket}
    doc_ids = []
    for i, row in enumerate(rows):
        try:
            doc_id = insert_document(kb_id, tenant_id, row, minio_config,
                                     args.chunk_tokens, enable_graphrag, enable_raptor)
            doc_ids.append(doc_id)
            print(f"  [{i+1}/{len(rows)}]  {(row.get('title') or row['minio_key'].rsplit('/',1)[-1])[:60]}")
        except Exception as e:
            print(f"  [{i+1}/{len(rows)}]  FAIL: {str(e)[:80]}")
    print(f"  Imported {len(doc_ids)}/{len(rows)} documents")

    # ── Step 4: 触发解析 ─────────────────────────────────────────
    if not args.no_parse and doc_ids:
        print(f"\n🔄 Step 4: 触发文档解析 (分块 + Embedding)")
        queue_parsing(tenant_id, kb_id, doc_ids)
        print(f"  Queued {len(doc_ids)} documents for parsing")

    # ── Step 5: 知识图谱 / RAPTOR ────────────────────────────────
    graph_enabled = enable_graphrag or enable_raptor
    if graph_enabled and doc_ids:
        labels = []
        if enable_graphrag: labels.append("GraphRAG")
        if enable_raptor: labels.append("RAPTOR")
        print(f"\n🧠 Step 5: 触发 {' + '.join(labels)}")
        queue_graphrag(tenant_id, kb_id, doc_ids, enable_graphrag, enable_raptor)

    # ── 汇总 ─────────────────────────────────────────────────────
    flags = ["向量索引"]
    if enable_graphrag: flags.append("GraphRAG 知识图谱")
    if enable_raptor: flags.append("RAPTOR 层级摘要")

    print(f"\n{'='*60}")
    print(f"✅ 导入完成!")
    print(f"  KB:       {kb_id} ({args.kb_name})")
    print(f"  文档数:   {len(doc_ids)}")
    print(f"  管道:     {' + '.join(flags)}")
    print(f"  监控:     docker logs -f ragflow-test")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
