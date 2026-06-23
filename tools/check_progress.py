#!/usr/bin/env python3
"""查看 GraphRAG / RAPTOR 生成进度"""
import sys, os

# 自动找 RAGFlow
for _c in [os.getcwd(), "/ragflow", os.path.expanduser("~/dev/ragflow")]:
    if os.path.isdir(os.path.join(_c, "common")):
        sys.path.insert(0, _c); break

from common.config_utils import read_config
from common.settings import init_settings
read_config(); init_settings()

from api.db.db_models import Knowledgebase, Task
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--kb-name", default="enflame-wiki")
args = parser.parse_args()

kb = Knowledgebase.select().where(Knowledgebase.name == args.kb_name).first()
if not kb:
    print(f"KB '{args.kb_name}' not found"); sys.exit(1)

gr = "✅" if kb.graphrag_task_finish_at else "🔄"
rp = "✅" if kb.raptor_task_finish_at else "🔄"
print(f"GraphRAG: {gr}  |  RAPTOR: {rp}  |  KB: {kb.id}")
print()

tasks = list(Task.select().where(
    Task.task_type.in_(['graphrag', 'raptor'])
).order_by(Task.id.desc()).limit(3))

for t in tasks:
    msgs = (t.progress_msg or '').strip().split('\n')
    relevant = [m for m in msgs if m.strip() and '**ERROR' not in m and 'Task has been' not in m]
    print(f"[{t.task_type}] {t.progress*100:.0f}%")
    for m in relevant[-5:]:
        print(f"  {m.strip()[:140]}")
    print()
