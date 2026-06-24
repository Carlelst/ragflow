# RAGFlow Project Instructions for GitHub Copilot

This file provides context, build instructions, and coding standards for the RAGFlow project.
It is structured to follow GitHub Copilot's [customization guidelines](https://docs.github.com/en/copilot/concepts/prompting/response-customization).

## 1. Project Overview
RAGFlow is an open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding. It is a full-stack application with a Python backend and a React/TypeScript frontend.

- **Backend**: Python 3.10+ (Flask/Quart)
- **Frontend**: TypeScript, React, UmiJS
- **Architecture**: Microservices based on Docker.
  - `api/`: Backend API server.
  - `rag/`: Core RAG logic (indexing, retrieval).
  - `deepdoc/`: Document parsing and OCR.
  - `web/`: Frontend application.

## 2. Directory Structure
- `api/`: Backend API server (Flask/Quart).
  - `apps/`: API Blueprints (Knowledge Base, Chat, etc.).
  - `db/`: Database models and services.
- `rag/`: Core RAG logic.
  - `llm/`: LLM, Embedding, and Rerank model abstractions.
- `deepdoc/`: Document parsing and OCR modules.
- `agent/`: Agentic reasoning components.
- `web/`: Frontend application (React + UmiJS).
- `docker/`: Docker deployment configurations.
- `sdk/`: Python SDK.
- `test/`: Backend tests.

## 3. Build Instructions

### Backend (Python)
The project uses **uv** for dependency management.

1. **Setup Environment**:
   ```bash
   uv sync --python 3.13 --all-extras
   uv run python3 download_deps.py
   ```

2. **Run Server**:
   - **Pre-requisite**: Start dependent services (MySQL, ES/Infinity, Redis, MinIO).
     ```bash
     docker compose -f docker/docker-compose-base.yml up -d
     ```
   - **Launch**:
     ```bash
     source .venv/bin/activate
     export PYTHONPATH=$(pwd)
     bash docker/launch_backend_service.sh
     ```

### Frontend (TypeScript/React)
Located in `web/`.

1. **Install Dependencies**:
   ```bash
   cd web
   npm install
   ```

2. **Run Dev Server**:
   ```bash
   npm run dev
   ```
   Runs on port 8000 by default.

### Docker Deployment
To run the full stack using Docker:
```bash
cd docker
docker compose -f docker-compose.yml up -d
```

## 4. Testing Instructions

### Backend Tests
- **Run All Tests**:
  ```bash
  uv run pytest
  ```
- **Run Specific Test**:
  ```bash
  uv run pytest test/test_api.py
  ```

### Frontend Tests
- **Run Tests**:
  ```bash
  cd web
  npm run test
  ```

## 5. Coding Standards & Guidelines
- **Python Formatting**: Use `ruff` for linting and formatting.
  ```bash
  ruff check
  ruff format
  ```
- **Frontend Linting**:
  ```bash
  cd web
  npm run lint
  ```
- **Pre-commit**: Ensure pre-commit hooks are installed.
  ```bash
  pre-commit install
  pre-commit run --all-files
  ```

## 6. 企业知识库运维

### Wiki 文档
- 文档页面: http://wiki.enflame.cn/pages/viewpage.action?pageId=370532312
- 目录 (TOC): `<ac:structured-macro ac:name="toc" ac:schema-version="1"/>` 放在内容最前面
- 更新命令: `bash ~/.claude/skills/wiki/wiki_skill.sh update --page-id 370532312 --body "..."`
- Wiki skill 路径: `~/.claude/skills/wiki/wiki_skill.sh`

### 环境信息
| 机器 | 服务 | 端口 |
|------|------|------|
| 192.168.20.21 | RAGFlow 测试 (ragflow-test) | 8088 |
| 10.9.200.13 | RAGFlow (ragflow-server) + MySQL + ES + Proxy + Dify | 8088/3307/1200/8090/8086 |
| 172.16.90.36 | MinIO | 9000 |
| 10.9.200.14 | PostgreSQL (wiki_metadata/html_metadata/wangpan_metadata) | 5432 |

### 关键凭据
- RAGFlow API key: `ragflow-307044760fae4f548209426ba6191d9e`
- KB ID (enflame-wiki): `abfeeC35Ff4AfcDfF5Fa88b8D38Fb4Ce`
- MySQL: root / infini_rag_flow (本机3306, 200.13:3307)
- MinIO: minioadmin / minioadmin @ 172.16.90.36:9000, bucket: rag-data
- PG: postgres / enflame @ 10.9.200.14:5432, db: metadata
- VLLM API: http://172.16.90.45:8082/v1 (scs_qwen3.5-397b)
- Embedding: qwen3-vl-embedding-8b (代理172.16.90.45:8082, 源头10.12.116.244:8006)
- Dify: admin@enflame.cn / EnflameAdmin123!

### 导入工具
- 脚本: `tools/batch_import.py`, 配置: `tools/batch_config.yaml`
- 图片预处理: `tools/preprocess_images.py`
- 运行环境: RAGFlow Docker 容器内（venv 缺依赖）
- 命令: `PYTHONUNBUFFERED=1 docker exec -e VLM_ENABLED=1 ragflow-test python3 /ragflow/tools/batch_import.py --config /ragflow/tools/batch_config.yaml --source wiki --wait`

### 已知 Bug 和修复
- Redis host 需含端口: `dify-redis-1:6379`
- batch_import --limit 时禁止删除 (limit > 0 跳过删除逻辑)
- MinIO bucket 字段导致路径重复拼接，去掉 service_conf 中 minio.bucket

