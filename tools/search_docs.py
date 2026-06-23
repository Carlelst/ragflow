#!/usr/bin/env python3
"""
RAGFlow 检索与维护工具 — 查询、增量同步、图谱重建、KB 管理

用法:
  # 查看文档状态
  python3 search_docs.py --kb-name enflame-wiki --status

  # 增量同步 (对比外部 PG，只处理变化)
  python3 search_docs.py --kb-name enflame-wiki --sync \\
    --pg-host 10.9.200.14 --pg-password enflame

  # 重建知识图谱 / RAPTOR
  python3 search_docs.py --kb-name enflame-wiki --regenerate-graphrag
  python3 search_docs.py --kb-name enflame-wiki --regenerate-raptor
  python3 search_docs.py --kb-name enflame-wiki --regenerate-all

  # 删除知识库
  python3 search_docs.py --kb-name enflame-wiki --delete
"""

import argparse, json, sys, os
from datetime import datetime
from collections import defaultdict


def find_kb(kb_name, tenant_id=None):
    from api.db.services.knowledgebase_service import KnowledgebaseService
    if tenant_id:
        kb = KnowledgebaseService.query(name=kb_name, tenant_id=tenant_id)
    else:
        kb = KnowledgebaseService.query(name=kb_name)
    return kb[0] if kb else None


def get_kb_stats(kb_id):
    from api.db.db_models import Document
    docs = list(Document.select().where(Document.kb_id == kb_id))
    stats = {"total": len(docs), "by_status": defaultdict(int), "total_chunks": 0, "total_tokens": 0}
    for d in docs:
        run_map = {"0": "未处理", "1": "处理中", "2": "取消", "3": "已完成", "4": "失败"}
        stats["by_status"][run_map.get(d.run, d.run)] += 1
        stats["total_chunks"] += (d.chunk_num or 0)
        stats["total_tokens"] += (d.token_num or 0)
    return stats


def delete_kb(kb_id, kb_name):
    from api.db.db_models import Document, Knowledgebase
    print(f"⚠️  警告: 将删除知识库 '{kb_name}' 及其所有文档!")
    if input("确认? (yes/no): ").lower() != "yes":
        print("取消删除"); return
    Document.delete().where(Document.kb_id == kb_id).execute()
    Knowledgebase.delete_by_id(kb_id)
    print(f"✅ 已删除: {kb_name}")


def show_status(kb_id, kb_name):
    from api.db.db_models import Document
    docs = list(Document.select().where(Document.kb_id == kb_id))
    stats = get_kb_stats(kb_id)
    print(f"\n📊 知识库: {kb_name} ({kb_id})")
    print(f"  文档:  {stats['total']}  |  chunks: {stats['total_chunks']}  |  tokens: {stats['total_tokens']}")
    print(f"  状态:  ", end="")
    for s, c in sorted(stats["by_status"].items()):
        print(f"{s}={c}  ", end="")
    print()
    print(f"\n{'名称':<52s} {'chunks':>6s} {'tokens':>6s}  {'状态'}")
    print("-" * 90)
    for d in docs:
        rm = {"0": "⏳未处理", "1": "🔄处理中", "2": "⏸取消", "3": "✅已完成", "4": "❌失败"}
        print(f"{d.name[:50]:<52s} {d.chunk_num or 0:>6d} {d.token_num or 0:>6d}  {rm.get(d.run, d.run)}")


def sync_documents(kb_id, tenant_id, pg_config, source_table, chunk_tokens=256):
    import psycopg2, psycopg2.extras
    from api.db.db_models import Document, DB
    from api.db.services.document_service import DocumentService
    from api.db.services.file2document_service import File2DocumentService
    from api.db.services.task_service import queue_tasks

    # 1. 外部 PG
    conn = psycopg2.connect(**pg_config)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(f"SELECT * FROM {source_table} ORDER BY id")
    ext_rows, ext_map = cur.fetchall(), {r['minio_key']: r for r in cur.fetchall()}
    cur.close(); conn.close()

    # 2. RAGFlow
    docs = list(Document.select().where(Document.kb_id == kb_id))
    doc_map = {d.location: d for d in docs}

    new_docs = [e for k, e in ext_map.items() if k not in doc_map]
    updated_docs = [(doc_map[k], e) for k, e in ext_map.items()
                    if k in doc_map and doc_map[k].content_hash != e.get('file_hash', '')]
    deleted_docs = [d for k, d in doc_map.items() if k not in ext_map]

    print(f"\n📊 增量分析: 外部{len(ext_rows)} | RAGFlow{len(docs)} | +{len(new_docs)} ~{len(updated_docs)} -{len(deleted_docs)}")

    new_ids = []
    for ext in new_docs:
        import random, string
        doc_id = ''.join(random.choices(string.hexdigits, k=32))
        # 简单插入
        suffix = ext['minio_key'].rsplit('.', 1)[-1].lower() if '.' in ext['minio_key'] else 'txt'
        if suffix == 'md': suffix = 'txt'
        name = (ext.get('title') or ext['minio_key'].rsplit('/', 1)[-1])[:255]
        if not name.endswith('.' + suffix):
            name = name[:250 - len(suffix)] + '.' + suffix
        doc = {
            "id": doc_id, "kb_id": kb_id, "parser_id": "naive", "pipeline_id": "",
            "parser_config": {"chunk_token_num": chunk_tokens, "delimiter": "\n"},
            "source_type": "local", "type": suffix, "created_by": tenant_id,
            "name": name, "location": ext['minio_key'],
            "size": 0, "suffix": suffix,
            "content_hash": ext.get('file_hash', ''),
        }
        DocumentService.insert(doc)
        new_ids.append(doc_id)
        print(f"  + {name[:50]}")

    for doc, ext in updated_docs:
        DB.execute_sql("UPDATE document SET content_hash=%s, run='0' WHERE id=%s",
                       (ext.get('file_hash', ''), doc.id))
        new_ids.append(doc.id)
        print(f"  ~ {doc.name[:50]}")

    for doc in deleted_docs:
        DocumentService.remove_document(doc, tenant_id)
        print(f"  - {doc.name[:50]}")

    if new_ids:
        print(f"\n🔄 触发 {len(new_ids)} 个文档解析...")
        for doc_id in new_ids:
            doc = Document.get_by_id(doc_id)
            bucket, name = File2DocumentService.get_storage_address(doc_id=doc_id)
            queue_tasks({
                "id": doc_id, "kb_id": kb_id, "tenant_id": tenant_id,
                "parser_id": doc.parser_id or "naive", "pipeline_id": doc.pipeline_id or "",
                "parser_config": doc.parser_config or {},
                "name": doc.name, "location": doc.location,
                "size": doc.size, "type": doc.type or "txt",
                "source_type": doc.source_type or "local",
            }, bucket, name, 0)
        print(f"  Done!")

    return len(new_docs), len(updated_docs), len(deleted_docs)


def regenerate_graph_tasks(kb_id, task_type):
    from api.db.db_models import Document
    from api.db.services.document_service import queue_raptor_o_graphrag_tasks
    docs = list(Document.select().where(Document.kb_id == kb_id))
    if not docs:
        print("  无文档"); return
    doc_ids = [d.id for d in docs]
    sample_doc = {"id": doc_ids[0]}
    try:
        queue_raptor_o_graphrag_tasks(sample_doc, task_type, 0, doc_ids=doc_ids)
        print(f"  ✅ {task_type} 已排队 ({len(doc_ids)} 文档)")
    except Exception as e:
        print(f"  ❌ {task_type} 失败: {e}")


# ── 主入口 ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAGFlow 检索与维护工具")
    parser.add_argument("--kb-name", required=True)
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--status", action="store_true", help="查看文档状态")
    parser.add_argument("--sync", action="store_true", help="增量同步")
    parser.add_argument("--regenerate-graphrag", action="store_true", help="重建 GraphRAG")
    parser.add_argument("--regenerate-raptor", action="store_true", help="重建 RAPTOR")
    parser.add_argument("--regenerate-all", action="store_true", help="重建 GraphRAG + RAPTOR")
    parser.add_argument("--delete", action="store_true", help="删除知识库")
    # sync 参数
    parser.add_argument("--pg-host"); parser.add_argument("--pg-password")
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--pg-db", default="metadata")
    parser.add_argument("--source-table", default="wiki_metadata")
    parser.add_argument("--chunk-tokens", type=int, default=256)

    args = parser.parse_args()

    if not any([args.status, args.sync, args.regenerate_graphrag,
                args.regenerate_raptor, args.regenerate_all, args.delete]):
        parser.print_help(); sys.exit(1)

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

    tenant_id = args.tenant_id or Tenant.select().first().id
    kb = find_kb(args.kb_name, tenant_id)
    if not kb:
        print(f"❌ 知识库 '{args.kb_name}' 不存在"); sys.exit(1)

    if args.status:
        show_status(kb.id, args.kb_name)
    if args.sync:
        if not args.pg_host or not args.pg_password:
            print("❌--sync 需要 --pg-host 和 --pg-password"); sys.exit(1)
        sync_documents(kb.id, tenant_id,
                       {"host": args.pg_host, "port": args.pg_port,
                        "user": args.pg_user, "password": args.pg_password,
                        "dbname": args.pg_db},
                       args.source_table, args.chunk_tokens)
    if args.regenerate_graphrag or args.regenerate_all:
        print(f"\n🧠 重建 GraphRAG...")
        regenerate_graph_tasks(kb.id, "graphrag")
    if args.regenerate_raptor or args.regenerate_all:
        print(f"\n🦖 重建 RAPTOR...")
        regenerate_graph_tasks(kb.id, "raptor")
    if args.delete:
        delete_kb(kb.id, args.kb_name)


if __name__ == "__main__":
    main()
