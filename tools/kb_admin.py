#!/usr/bin/env python3
"""
KB 管理工具 — 通过 REST API 创建/删除/查看知识库

用法:
  python3 kb_admin.py create --name enflame-wiki [--embd qwen3-vl-embedding-8b@VLLM]
  python3 kb_admin.py delete --name enflame-wiki
  python3 kb_admin.py show [--name enflame-wiki]
"""

import argparse, json, os, sys
import requests

RAGFLOW_URL = os.environ.get("RAGFLOW_URL", "http://127.0.0.1:8088")
RAGFLOW_API_KEY = os.environ.get("RAGFLOW_API_KEY", "ragflow-307044760fae4f548209426ba6191d9e")
HEADERS = {"Authorization": f"Bearer {RAGFLOW_API_KEY}", "Content-Type": "application/json"}
API = f"{RAGFLOW_URL}/api/v1/datasets"


def create(name, embd="qwen3-vl-embedding-8b@VLLM", chunk_tokens=600, delimiter="\n"):
    resp = requests.post(API, json={
        "name": name,
        "embedding_model": embd,
        "chunk_method": "naive",
        "parser_config": {"chunk_token_num": chunk_tokens, "delimiter": delimiter},
    }, headers=HEADERS, timeout=10)
    data = resp.json()
    if data.get("code") == 0:
        kb = data["data"]
        print(f"KB created: id={kb['id']} name={kb['name']}")
        return kb
    raise Exception(f"Create failed: {data.get('message', resp.text)}")


def delete(name=None, kb_id=None):
    if kb_id:
        ids = [kb_id]
    elif name:
        kb = _find(name)
        if not kb:
            print(f"KB '{name}' not found")
            return False
        ids = [kb["id"]]
    else:
        print("Specify --name or --id")
        return False
    resp = requests.delete(API, json={"ids": ids}, headers=HEADERS, timeout=10)
    data = resp.json()
    if data.get("code") == 0:
        print(f"KB deleted: {ids[0]}")
        return True
    print(f"Delete failed: {data.get('message')}")
    return False


def _find(name):
    resp = requests.get(f"{API}?name={name}", headers=HEADERS, timeout=10)
    data = resp.json()
    if data.get("code") == 0 and data.get("data"):
        return data["data"][0]
    return None


def show(name=None):
    if name:
        kb = _find(name)
        if not kb:
            print(f"KB '{name}' not found")
            return
        print_kb(kb)
    else:
        resp = requests.get(API, headers=HEADERS, timeout=10)
        data = resp.json()
        if data.get("code") == 0:
            for kb in data.get("data", []):
                print_kb(kb)
            print(f"\nTotal: {len(data.get('data', []))}")


def print_kb(kb):
    print(f"\n  Name: {kb['name']}")
    print(f"  ID: {kb['id']}")
    print(f"  Embedding: {kb.get('embedding_model', '')}")
    print(f"  Chunks: {kb.get('chunk_count', 0)}")
    print(f"  Docs: {kb.get('document_count', 0)}")
    cfg = kb.get("parser_config", {})
    print(f"  Chunk tokens: {cfg.get('chunk_token_num', '-')}")
    print(f"  Delimiter: {cfg.get('delimiter', '-')}")


def main():
    parser = argparse.ArgumentParser(description="RAGFlow KB Admin")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("create")
    p.add_argument("--name", required=True)
    p.add_argument("--embd", default="qwen3-vl-embedding-8b@VLLM")
    p.add_argument("--chunk-tokens", type=int, default=600)
    p.add_argument("--delimiter", default="\n")

    p = sub.add_parser("delete")
    p.add_argument("--name")
    p.add_argument("--id")

    p = sub.add_parser("show")
    p.add_argument("--name")

    args = parser.parse_args()
    if args.cmd == "create":
        create(args.name, args.embd, args.chunk_tokens, args.delimiter)
    elif args.cmd == "delete":
        delete(args.name, args.id)
    elif args.cmd == "show":
        show(args.name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
