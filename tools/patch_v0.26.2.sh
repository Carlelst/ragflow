#!/bin/bash
# RAGFlow v0.26.2 补丁：降低 embedding truncation 到 8000
# 解决 tiktoken 和 qwen3 tokenizer 计数不一致导致 8193 token 报错
#
# Worker 数量：docker-compose.remote.yml 中 command: --workers=8
# docker run 时加: docker run ... finiflow/ragflow:v0.26.2 --workers=8
sed -i 's/truncate_to=8191/truncate_to=4000/g' /ragflow/rag/llm/embedding_model.py
echo "patched truncate_to=4000"
