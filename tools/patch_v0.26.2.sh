#!/bin/bash
# RAGFlow v0.26.2 补丁：降低 embedding truncation 到 8100
# 解决 tiktoken 和 qwen3 tokenizer 计数不一致导致 8193 token 报错
sed -i 's/truncate_to=8191/truncate_to=8100/g' /ragflow/rag/llm/embedding_model.py
echo "patched truncate_to=8100"
