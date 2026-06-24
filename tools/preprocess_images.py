#!/usr/bin/env python3
"""
预处理 Wiki Markdown 文档中的图片 — 用 VLM 生成描述，嵌入正文。

对每个 wiki 文档：
1. 从 MinIO 读取 markdown
2. 提取图片相对路径引用 `![...](...)`
3. 从 MinIO 找到对应图片文件
4. 发送给 VLM 生成描述
5. 将描述 + 原始 wiki 链接附加到 markdown 正文末尾
6. 写回 MinIO（覆盖原文件）

用法:
  python3 preprocess_images.py --limit 5           # 测试5篇
  python3 preprocess_images.py                     # 全部233篇
"""

import argparse
import os
import re
import sys
import base64
import requests
from urllib.parse import urlparse, unquote
from minio import Minio
import psycopg2

# ══════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════

MINIO_HOST = os.environ.get("MINIO_HOST", "172.16.90.36:9000")
MINIO_ACCESS = os.environ.get("MINIO_ACCESS", "minioadmin")
MINIO_SECRET = os.environ.get("MINIO_SECRET", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "rag-data")

PG_HOST = os.environ.get("PG_HOST", "10.9.200.14")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "enflame")
PG_DB = os.environ.get("PG_DB", "metadata")

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://172.16.90.45:8082/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "scs_qwen3.5-397b")
VLLM_MAX_TOKENS = int(os.environ.get("VLLM_MAX_TOKENS", "512"))
VLLM_TIMEOUT = int(os.environ.get("VLLM_TIMEOUT", "60"))


def init_minio():
    return Minio(MINIO_HOST, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)


def init_pg():
    return psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB)


def extract_image_refs(md_content: str) -> list[tuple[str, str]]:
    """从 markdown 中提取图片引用，返回 [(相对路径, 原始引用文本), ...]"""
    refs = []
    for m in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', md_content):
        alt = m.group(1) or ""
        path = m.group(2)
        refs.append((path, m.group(0)))
    return refs


def resolve_image_minio(minio_client: Minio, doc_minio_key: str, relative_path: str) -> str | None:
    """
    将 markdown 中的相对图片路径解析为 MinIO 中的实际路径。
    策略：
    1. 先按相对路径解析
    2. 如果不存在，尝试在上层 images/ 目录查找同名文件
    """
    doc_dir = os.path.dirname(doc_minio_key)
    resolved = os.path.normpath(os.path.join(doc_dir, relative_path)).replace("\\", "/")

    try:
        minio_client.stat_object(MINIO_BUCKET, resolved)
        return resolved
    except:
        pass

    # 策略2: 在上层 images/ 目录查找同名文件
    filename = os.path.basename(relative_path)
    # 向上查找 images/ 目录
    parts = doc_dir.split("/")
    for i in range(len(parts), 0, -1):
        candidate = "/".join(parts[:i]) + "/images/" + filename
        try:
            minio_client.stat_object(MINIO_BUCKET, candidate)
            return candidate
        except:
            continue

    return None


def describe_image_vllm(image_bytes: bytes, alt_text: str, wiki_url: str) -> str:
    """用 VLLM 模型描述图片内容"""
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "请用中文描述这张图片的内容。如果是截图，描述其中显示的信息、数据和关键内容。"
        "如果是架构图或流程图，描述其结构和关系。限制在150字以内。"
    )
    if alt_text:
        prompt = f"图片alt文本: {alt_text}\n{prompt}"

    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ],
        "max_tokens": VLLM_MAX_TOKENS,
        "temperature": 0.1,
    }

    resp = requests.post(
        f"{VLLM_BASE_URL}/chat/completions",
        json=payload,
        timeout=VLLM_TIMEOUT,
    )
    resp.raise_for_status()
    result = resp.json()
    return result["choices"][0]["message"]["content"].strip()


def process_one_doc(minio_client: Minio, pg_cur, minio_key: str, source_url: str, limit: int = None):
    """处理单篇文档的图片"""
    try:
        data = minio_client.get_object(MINIO_BUCKET, minio_key)
        content = data.read().decode("utf-8")
        data.close()
        data.release_conn()
    except Exception as e:
        print(f"  ❌ 读取MinIO失败: {e}")
        return 0

    refs = extract_image_refs(content)
    if not refs:
        return 0

    modified = False
    processed = 0
    for rel_path, orig_ref in refs[:limit] if limit else refs:
        # 跳过外部URL和非图片文件
        if rel_path.startswith(("http://", "https://", "data:")):
            continue
        if not rel_path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")):
            continue

        minio_img_path = resolve_image_minio(minio_client, minio_key, rel_path)
        if not minio_img_path:
            # 也检查是否文件名存在于 MinIO 根级 images 目录
            continue

        try:
            img_data = minio_client.get_object(MINIO_BUCKET, minio_img_path)
            img_bytes = img_data.read()
            img_data.close()
            img_data.release_conn()
        except Exception as e:
            print(f"    ⚠ 下载图片失败 {minio_img_path}: {e}")
            continue

        # 提取 alt 文本
        alt_match = re.search(r'!\[([^\]]*)\]', orig_ref)
        alt_text = alt_match.group(1) if alt_match else ""

        # VLM 描述
        try:
            description = describe_image_vllm(img_bytes, alt_text, source_url)
        except Exception as e:
            print(f"    ⚠ VLM调用失败: {e}")
            continue

        # 构建附加内容
        append_text = f"\n\n> 📷 **图片描述**: {description}\n> 🔗 原文链接: {source_url}"

        # 在原始引用后插入描述
        content = content.replace(orig_ref, orig_ref + append_text)
        modified = True
        processed += 1
        print(f"    ✅ {os.path.basename(rel_path)}: {description[:60]}...")

    if modified:
        # 写回 MinIO
        try:
            import io
            data_bytes = content.encode("utf-8")
            minio_client.put_object(
                MINIO_BUCKET, minio_key,
                io.BytesIO(data_bytes), len(data_bytes),
                content_type="text/markdown",
            )
        except Exception as e:
            print(f"  ❌ 写回MinIO失败: {e}")
            return 0

    return processed


def main():
    parser = argparse.ArgumentParser(description="预处理 Wiki 文档中的图片")
    parser.add_argument("--limit", type=int, default=0, help="限制处理篇数(0=全部)")
    parser.add_argument("--doc-id", type=int, default=0, help="只处理指定PG ID的文档")
    args = parser.parse_args()

    minio_client = init_minio()
    pg_conn = init_pg()
    pg_cur = pg_conn.cursor()

    if args.doc_id:
        pg_cur.execute(
            "SELECT minio_key, source_url, title FROM wiki_metadata WHERE id = %s",
            (args.doc_id,),
        )
    else:
        pg_cur.execute(
            "SELECT minio_key, source_url, title FROM wiki_metadata WHERE minio_key IS NOT NULL ORDER BY id"
        )

    total = 0
    total_imgs = 0

    for row in pg_cur.fetchall()[:args.limit] if args.limit else pg_cur.fetchall():
        minio_key, source_url, title = row
        if not minio_key:
            continue

        print(f"\n📄 [{total+1}] {title[:60]}")
        n = process_one_doc(minio_client, pg_cur, minio_key, source_url)
        total += 1
        total_imgs += n

    pg_cur.close()
    pg_conn.close()
    print(f"\n{'='*60}")
    print(f"✅ 完成: {total} 篇文档, {total_imgs} 张图片")


if __name__ == "__main__":
    main()
