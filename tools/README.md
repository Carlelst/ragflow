# RAGFlow 企业知识库工具集

将外部 PostgreSQL + MinIO 中的文档批量导入 RAGFlow，支持向量索引、GraphRAG 知识图谱、RAPTOR 层级摘要。按数据源独立管理，支持增量同步。

## 架构

```
外部 PostgreSQL                    RAGFlow
┌─────────────────┐              ┌─────────────────────────┐
│ metadata 库      │              │ MySQL (元数据管理)        │
│                 │   导入脚本    │                         │
│ wiki_metadata   │──────────────│ → tenant               │
│ html_metadata   │              │ → knowledgebase ×3     │
│ wangpan_metadata│              │ → document             │
└────────┬────────┘              │ → file / file2document │
         │                       └───────────┬─────────────┘
         │ minio_key                          │
         ▼                                   ▼
┌─────────────────┐              ┌─────────────────────────┐
│ 外部 MinIO       │    Worker 读取 │ Elasticsearch           │
│ rag-data bucket  │◄─────────────│                         │
│ *.md / *.pdf ... │              │ 向量索引 (chunk)         │
└─────────────────┘              │ 知识图谱 (entity/relation)│
                                 │ 层级摘要 (RAPTOR)        │
                                 └─────────────────────────┘

检索链路: 用户查询 → 多路融合 → Rerank → 带引用的问答
```

## 目录

- [快速开始](#快速开始)
- [工具说明](#工具说明)
- [数据源配置](#数据源配置)
- [管道开关](#管道开关)
- [增量同步](#增量同步)
- [命令参考](#命令参考)

---

## 快速开始

### 环境要求

- RAGFlow 已启动（API Server + Task Executor + Elasticsearch + Redis）
- 外部 PostgreSQL 可访问（`10.9.200.14:5432`）
- 外部 MinIO 可访问（`172.16.90.36:9000`）
- Embedding 模型已配置（`scs_gateway_embedding`，最大 512 tokens）

### 一键运行

```bash
cd /home/shentao.lu/dev/ragflow

# 1. 分别测试 (每源 10 条)
python3 tools/batch_import.py --source wiki --limit 10
python3 tools/batch_import.py --source html --limit 10
python3 tools/batch_import.py --source wangpan --limit 10

# 2. 查看导入状态
python3 tools/batch_import.py --status

# 3. 全量导入
python3 tools/batch_import.py --source all

# 4. 监控处理进度
docker logs -f ragflow-test
```

---

## 工具说明

| 工具 | 用途 | 何时使用 |
|------|------|---------|
| `batch_import.py` | 按数据源批量导入文档 | 首次导入、新增数据源 |
| `search_docs.py` | 增量同步 / 知识库维护 | 定期更新、重建图谱、查看状态 |
| `import_docs.py` | 单数据源精细导入 | 需要自定义参数的场景（一般用 batch_import.py） |

### batch_import.py

```bash
# 基础导入
python3 tools/batch_import.py --source <wiki|html|wangpan|all>

# 带知识图谱
python3 tools/batch_import.py --source wiki --enable-graphrag --enable-raptor

# 覆盖外部服务地址
python3 tools/batch_import.py --source html \
  --pg-host 10.9.200.14 --pg-password enflame \
  --chunk-tokens 400

# 查看所有知识库
python3 tools/batch_import.py --status
```

### search_docs.py

```bash
# 查看文档状态
python3 tools/search_docs.py --kb-name enflame-wiki --status

# 增量同步 (file_hash 对比)
python3 tools/search_docs.py --kb-name enflame-wiki --sync \
  --pg-host 10.9.200.14 --pg-password enflame

# 重建知识图谱
python3 tools/search_docs.py --kb-name enflame-wiki --regenerate-all
python3 tools/search_docs.py --kb-name enflame-wiki --regenerate-graphrag
python3 tools/search_docs.py --kb-name enflame-wiki --regenerate-raptor

# 删除知识库
python3 tools/search_docs.py --kb-name enflame-wiki --delete
```

---

## 数据源配置

每个数据源创建独立的知识库，互不干扰：

| 数据源 | 外部表 | KB 名称 | 文档类型 | 默认 chunk tokens |
|--------|--------|---------|---------|------------------|
| `wiki` | `wiki_metadata` | `enflame-wiki` | Confluence Markdown (.md→.txt) | 256 |
| `html` | `html_metadata` | `enflame-docs` | 文档站 HTML (.md→.txt) | 512 |
| `wangpan` | `wangpan_metadata` | `enflame-wangpan` | 企业网盘文件 | 512 |

外部连接默认值（可通过命令行覆盖）：

```python
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
```

### 元数据字段映射

| PG 字段 | RAGFlow 字段 | 用途 |
|---------|-------------|------|
| `minio_key` | `document.location` | MinIO 对象路径 |
| `title` | `document.name` | 文档文件名（追加 `.txt` 后缀） |
| `file_hash` | `document.content_hash` | 增量更新时的变更检测 |

---

## 管道开关

文档导入后可配置三种处理管道：

| 管道 | 说明 | 依赖 | 典型耗时 (每篇) |
|------|------|------|----------------|
| 向量索引 | 分块 → Embedding → ES | 无 | ~5s |
| GraphRAG | 实体提取 → 社区检测 → 摘要 | **必须先跑向量索引** | ~30-60s (LLM) |
| RAPTOR | 聚类 → 层级摘要 | **必须先跑向量索引** | ~30-60s (LLM) |

**重要**: GraphRAG 和 RAPTOR 从 ES 中读取已有的 chunk 数据，不直接读原始文件。必须先完成向量索引步骤。

### 开关控制

```bash
# 默认: 仅向量索引
python3 tools/batch_import.py --source wiki

# 向量 + 知识图谱
python3 tools/batch_import.py --source wiki --enable-graphrag --enable-raptor

# 仅知识图谱 (不生成向量)
python3 tools/batch_import.py --source wiki --no-parse --enable-graphrag

# 在已导入的知识库上补充图谱
python3 tools/search_docs.py --kb-name enflame-wiki --regenerate-all
```

### 所需模型

| 管道 | 模型 | 用途 |
|------|------|------|
| 向量索引 | `scs_gateway_embedding` | 文本 → 向量 |
| GraphRAG | `scs_qwen3.5-397b` (Chat) | 实体/关系提取 + 社区摘要 |
| RAPTOR | `scs_qwen3.5-397b` (Chat) | 层级摘要 |
| 重排序 | `scs_bge-reranker` | 检索结果精排 |

---

## 增量同步

`search_docs.py --sync` 支持增量更新，适合定时任务：

```
外部 PG (wiki_metadata)
        │
        ├── minio_key 新增 → 导入新文档 + 重新解析
        ├── file_hash 变更 → 标记更新 + 重新解析
        ├── file_hash 不变 → 跳过
        └── minio_key 删除 → 从 RAGFlow 移除
```

```bash
# 手动触发
python3 tools/search_docs.py --kb-name enflame-wiki --sync \
  --pg-host 10.9.200.14 --pg-password enflame

# 定时任务 (crontab)
0 2 * * * cd /path/to/ragflow && python3 tools/search_docs.py \
  --kb-name enflame-wiki --sync \
  --pg-host 10.9.200.14 --pg-password enflame >> /var/log/ragflow-sync.log 2>&1
```

---

## 命令参考

### batch_import.py

```
usage: batch_import.py [-h] [--source {wiki,html,wangpan,all}]
                       [--limit LIMIT] [--status]
                       [--tenant-id TENANT_ID] [--embd-id EMBEDDING_ID]
                       [--pg-host PG_HOST] [--pg-password PG_PASSWORD]
                       [--enable-graphrag] [--enable-raptor]
                       [--no-parse] [--chunk-tokens CHUNK_TOKENS]

参数:
  --source          数据源 (wiki|html|wangpan|all)，默认 wiki
  --limit           每个数据源导入上限，0=全部，默认 0
  --status          查看所有 KB 状态
  --enable-graphrag 启用 GraphRAG 知识图谱
  --enable-raptor   启用 RAPTOR 层级摘要
  --no-parse        只写入文档记录，不触发分块+向量化
  --chunk-tokens    覆盖默认 chunk token 数
  --pg-host/--pg-password  覆盖 PostgreSQL 连接信息
```

### search_docs.py

```
usage: search_docs.py [-h] --kb-name KB_NAME
                      [--status] [--sync] [--delete]
                      [--regenerate-graphrag] [--regenerate-raptor]
                      [--regenerate-all]
                      [--pg-host PG_HOST] [--pg-password PG_PASSWORD]
                      [--chunk-tokens CHUNK_TOKENS]

参数:
  --kb-name             知识库名称 (必填)
  --status              查看文档处理状态
  --sync                增量同步 (对比外部 PG)
  --regenerate-graphrag 重新生成 GraphRAG
  --regenerate-raptor   重新生成 RAPTOR
  --regenerate-all      重新生成 GraphRAG + RAPTOR
  --delete              删除知识库 (需确认)
```

### import_docs.py (单源精细控制)

```
usage: import_docs.py [-h] --pg-host PG_HOST --pg-password PG_PASSWORD
                      --minio-host MINIO_HOST --minio-user MINIO_USER
                      --minio-password MINIO_PASSWORD
                      --kb-name KB_NAME --source-table TABLE_NAME
                      [--limit LIMIT] [--enable-graphrag] [--enable-raptor]
                      [--no-parse]

参数:
  比 batch_import.py 多了 --minio-host/minio-user/minio-password 和 --source-table
  适用于自定义表名或非标准 MinIO 连接
```

---

## 故障排查

### 文档解析失败: "file type not supported yet"

文档名必须以 `.txt` / `.pdf` / `.docx` 结尾。Markdown 文件需重命名为 `.txt` 后缀以使用 TxtParser。

### Embedding 失败: "maximum context length is 512 tokens"

`scs_gateway_embedding` 最大 512 token。将 `--chunk-tokens` 设为 200-256。

### GraphRAG 无结果

确认已先完成向量索引步骤（不能 `--no-parse` + `--enable-graphrag` 一起用，除非之前已经解析过）。

### 任务队列堆积

```bash
# 检查 Worker 状态
docker logs ragflow-test | grep heartbeat
# done=10 表示已完成 10 个任务
# failed=0 表示无失败
# pending>0 表示队列中有待处理任务
```
