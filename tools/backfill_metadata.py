#!/usr/bin/env python3
"""
EKB 元数据回填工具 — 对已向量化的文档批量写入 doc_meta（方案 A：doc 级注入）。

对 ekb_4.5_wiki_dev / ekb_4.5_pan_dev 两个 KB 的所有已存在文档：
  1. 按 minio_key 从 PG 查元数据行（wiki_metadata / wangpan_metadata）
  2. 读 MinIO 文件首行解析【EKB_X.Y】标记（仅 md；xlsx 走兜底）
  3. 校验：有标记但不含 PG project_name → 记 ekb_anomalies.log + 跳过
  4. 调 DocMetadataService.update_document_metadata 写入 ES doc_meta 索引

用法（容器内）:
  cd /ragflow && python3 tools/backfill_metadata.py
  python3 tools/backfill_metadata.py --kb ekb_4.5_wiki_dev --project 4.5 --source wiki
  python3 tools/backfill_metadata.py --kb ekb_4.5_pan_dev  --project 4.5 --source wangpan
"""
import argparse, os, sys, json, re, time

# 初始化 RAGFlow（复用 batch_import 的路径查找逻辑）
_ragflow_home = os.environ.get("RAGFLOW_HOME")
if _ragflow_home:
    sys.path.insert(0, _ragflow_home)
else:
    for _c in [os.getcwd(),
               os.path.join(os.path.dirname(__file__), "..", ".."),
               "/ragflow"]:
        if os.path.isdir(os.path.join(_c, "common")):
            sys.path.insert(0, _c); break

from common.config_utils import read_config
read_config()
from common import settings as S
S.init_settings()

from minio import Minio
from api.db.db_models import Document, Knowledgebase
from api.db.services.doc_metadata_service import DocMetadataService
from batch_import import (build_full_meta, enrich_project_meta,
                          extract_ekb_tags, log_ekb_anomaly,
                          PG_DEFAULTS, MINIO_DEFAULTS)

import psycopg2, psycopg2.extras


def backfill_kb(kb_name, pg_project, source_key, pg_config, minio_config, anomaly_log):
    kb = Knowledgebase.select().where(Knowledgebase.name == kb_name).first()
    if not kb:
        print(f"  KB {kb_name} 不存在，跳过"); return
    print(f"\n=== {kb_name} (id={kb.id}) ===")

    docs = list(Document.select().where(Document.kb_id == kb.id))
    print(f"  文档总数: {len(docs)}")

    mc = Minio(minio_config["host"], access_key=minio_config["user"],
               secret_key=minio_config["password"],
               secure=minio_config.get("secure", False))
    bucket = minio_config.get("bucket", MINIO_DEFAULTS["bucket"])

    conn = psycopg2.connect(**pg_config)
    table = "wiki_metadata" if source_key == "wiki" else "wangpan_metadata"

    ok = skip_nometa = skip_anomaly = fail = 0
    for i, doc in enumerate(docs):
        mk = doc.location
        # 1. 查 PG
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(f"SELECT * FROM {table} WHERE minio_key = %s", (mk,))
        row = cur.fetchone()
        cur.close()
        if not row:
            skip_nometa += 1
            continue

        # 2. 解析 EKB 标记（仅 md）
        tags = extract_ekb_tags(mc, bucket, mk)

        # 3. 异常校验
        if tags and pg_project not in tags:
            log_ekb_anomaly(row, pg_project, tags, anomaly_log, source_key)
            skip_anomaly += 1
            print(f"  [EKBTAG] 跳过 {doc.name[:50]} (PG={pg_project}, 标记={tags})")
            continue

        # 4. 写 doc_meta
        meta = build_full_meta(row, source_key, pg_project)
        meta = enrich_project_meta(meta, tags, pg_project)
        try:
            DocMetadataService.update_document_metadata(doc.id, meta)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  [FAIL] {doc.name[:50]}: {str(e)[:60]}")

        if (i + 1) % 100 == 0:
            print(f"  进度 {i+1}/{len(docs)} (ok={ok} skip={skip_nometa+skip_anomaly} fail={fail})")

    conn.close()
    print(f"  完成: ok={ok} 无PG元数据={skip_nometa} 异常跳过={skip_anomaly} 失败={fail}")
    return dict(ok=ok, skip_nometa=skip_nometa, skip_anomaly=skip_anomaly, fail=fail)


def main():
    ap = argparse.ArgumentParser(description="EKB 元数据回填")
    ap.add_argument("--kb", default=None, help="单个 KB 名（默认跑 wiki_dev + pan_dev）")
    ap.add_argument("--project", default="4.5")
    ap.add_argument("--source", default=None, choices=["wiki", "wangpan"],
                    help="数据源（与 --kb 配合）")
    args = ap.parse_args()

    pg_config = dict(PG_DEFAULTS)
    pg_config["host"] = "172.16.90.36"
    pg_config["password"] = "postgres"
    pg_config["dbname"] = "postgres"
    minio_config = dict(MINIO_DEFAULTS)
    minio_config["bucket"] = "ekb"
    anomaly_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ekb_anomalies.log")

    jobs = []
    if args.kb:
        if not args.source:
            print("--kb 需要 --source wiki|wangpan"); return
        jobs.append((args.kb, args.project, args.source))
    else:
        jobs.append(("ekb_4.5_wiki_dev", "4.5", "wiki"))
        jobs.append(("ekb_4.5_pan_dev",  "4.5", "wangpan"))

    print(f"docStoreConn: {type(S.docStoreConn).__name__}")
    print(f"PG: {pg_config['host']}/{pg_config['dbname']}  MinIO bucket: {minio_config['bucket']}")
    print(f"异常日志: {anomaly_log}")

    total = dict(ok=0, skip_nometa=0, skip_anomaly=0, fail=0)
    for kb_name, proj, src in jobs:
        r = backfill_kb(kb_name, proj, src, pg_config, minio_config, anomaly_log)
        if r:
            for k, v in r.items(): total[k] += v

    print(f"\n=== 总计: ok={total['ok']} 无PG={total['skip_nometa']} "
          f"异常={total['skip_anomaly']} 失败={total['fail']} ===")
    if total['skip_anomaly']:
        print(f"⚠ {total['skip_anomaly']} 篇文档 EKB 标记与 PG project 不一致，见 {anomaly_log}")


if __name__ == "__main__":
    main()
