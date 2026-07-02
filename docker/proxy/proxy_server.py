#!/usr/bin/env python3
"""
RAGFlow Dify 外部知识库 API 代理 — 生产级 v2.1

Dify External Knowledge API 规范:
  请求: POST /retrieval
        Headers: Authorization: Bearer <api_key>
        Body: {"knowledge_id": "xxx", "query": "xxx", "top_k": 10}
  响应: {"records": [{"content": "...", "score": 0.9,
                       "title": "...", "source_url": "...", "metadata": {...}}]}

endpoints:
  GET  /health      — 健康检查（含 cache 指标）
  POST /retrieval   — 知识检索
  POST /feedback    — 来源质量评价收集
  POST /shutdown    — 优雅关闭
"""

import json, logging, os, re, signal, sys, time
from datetime import datetime

import httpx
from flask import Flask, request, jsonify, g

# ===========================================================================
# Logging（最先初始化，后续模块可用）
# ===========================================================================

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry, ensure_ascii=False)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("ragflow-dify-proxy")

# ===========================================================================
# 配置 — proxy_config.json 优先，环境变量可覆盖
# ===========================================================================

def _load_config():
    """加载配置：proxy_config.json → 环境变量覆盖 → 硬编码兜底"""
    CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_FILE = os.path.join(CONFIG_DIR, "proxy_config.json")

    # 默认值（兜底）
    cfg = {
        "ragflow": {"base_url": "http://127.0.0.1:8088", "api_key": ""},
        "dify": {"console_base": "http://127.0.0.1:8086"},
        "retrieval": {"default_top_k": 10, "request_timeout": 90, "connect_timeout": 5},
        "pool": {"max_connections": 200, "max_keepalive": 50, "retry_max": 2},
        "cache": {"ttl_seconds": 60, "size": 2000, "redis_url": ""},
        "circuit_breaker": {"fail_threshold": 5, "cooldown_seconds": 30, "half_open_max": 1},
        "feedback": {"root_dir": "/home/ai.hse", "file": "rag-feedback.jsonl"},
        "monitoring": {"slow_threshold_seconds": 5.0},
    }

    # 1. 读取配置文件
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                file_cfg = json.load(f)
            for section in cfg:
                if section in file_cfg:
                    cfg[section].update({k: v for k, v in file_cfg[section].items() if k in cfg[section]})
            logger.info("config loaded from %s", CONFIG_FILE)
        except Exception as e:
            logger.warning("config file error, using defaults: %s", str(e)[:100])
    else:
        logger.warning("config file not found: %s, using defaults", CONFIG_FILE)

    # 2. 环境变量覆盖（向后兼容）
    env_map = {
        ("ragflow", "base_url"): "RAGFLOW_BASE_URL",
        ("ragflow", "api_key"): "RAGFLOW_API_KEY",
        ("dify", "console_base"): "DIFY_CONSOLE_BASE",
        ("retrieval", "default_top_k"): "DEFAULT_TOP_K",
        ("retrieval", "request_timeout"): "REQUEST_TIMEOUT",
        ("retrieval", "connect_timeout"): "CONNECT_TIMEOUT",
        ("pool", "max_connections"): "POOL_MAX_SIZE",
        ("pool", "max_keepalive"): "POOL_MAX_KEEPALIVE",
        ("pool", "retry_max"): "RETRY_MAX",
        ("cache", "ttl_seconds"): "CACHE_TTL",
        ("cache", "size"): "CACHE_SIZE",
        ("cache", "redis_url"): "REDIS_URL",
        ("circuit_breaker", "fail_threshold"): "CB_FAIL_THRESHOLD",
        ("circuit_breaker", "cooldown_seconds"): "CB_COOLDOWN_SEC",
        ("circuit_breaker", "half_open_max"): "CB_HALF_OPEN_MAX",
        ("feedback", "root_dir"): "FEEDBACK_ROOT",
        ("monitoring", "slow_threshold_seconds"): "SLOW_THRESHOLD",
    }
    for (section, key), env_var in env_map.items():
        val = os.getenv(env_var)
        if val is not None:
            orig_type = type(cfg[section][key])
            try:
                cfg[section][key] = orig_type(val)
            except (ValueError, TypeError):
                cfg[section][key] = val

    return cfg

_config = _load_config()

# 展开为模块级变量（保持原有代码兼容）
RAGFLOW_BASE_URL  = _config["ragflow"]["base_url"]
RAGFLOW_API_KEY   = _config["ragflow"]["api_key"]
DIFY_CONSOLE_BASE = _config["dify"]["console_base"]
DEFAULT_TOP_K     = _config["retrieval"]["default_top_k"]
REQUEST_TIMEOUT   = _config["retrieval"]["request_timeout"]
CONNECT_TIMEOUT   = _config["retrieval"]["connect_timeout"]
POOL_MAX_SIZE     = _config["pool"]["max_connections"]
POOL_MAX_KEEPALIVE = _config["pool"]["max_keepalive"]
RETRY_MAX         = _config["pool"]["retry_max"]
CACHE_TTL         = _config["cache"]["ttl_seconds"]
CACHE_SIZE        = _config["cache"]["size"]
SLOW_THRESHOLD    = _config["monitoring"]["slow_threshold_seconds"]
FEEDBACK_ROOT     = _config["feedback"]["root_dir"]
FEEDBACK_FILE     = os.path.join(FEEDBACK_ROOT, _config["feedback"]["file"])
CB_FAIL_THRESHOLD = _config["circuit_breaker"]["fail_threshold"]
CB_COOLDOWN_SEC   = _config["circuit_breaker"]["cooldown_seconds"]
CB_HALF_OPEN_MAX  = _config["circuit_breaker"]["half_open_max"]
REDIS_URL         = _config["cache"]["redis_url"]

app = Flask(__name__)

# ===========================================================================
# Circuit Breaker
# ===========================================================================
class CircuitBreaker:
    """简单熔断器：N 次连续失败 → 打开 30s → half-open 试探 1 次"""

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, fail_threshold=CB_FAIL_THRESHOLD, cooldown=CB_COOLDOWN_SEC, half_open_max=CB_HALF_OPEN_MAX):
        self.fail_threshold = fail_threshold
        self.cooldown = cooldown
        self.half_open_max = half_open_max
        self.state = self.CLOSED
        self.fail_count = 0
        self.last_fail_time = 0
        self.half_open_count = 0

    def call(self, func, *args, **kwargs):
        if self.state == self.OPEN:
            if time.time() - self.last_fail_time > self.cooldown:
                self.state = self.HALF_OPEN
                self.half_open_count = 0
                logger.warning("circuit half-open, probing upstream")
            else:
                raise CircuitOpenError(f"circuit open, retry in {int(self.cooldown - (time.time() - self.last_fail_time))}s")

        if self.state == self.HALF_OPEN and self.half_open_count >= self.half_open_max:
            raise CircuitOpenError("circuit half-open limit reached")

        if self.state == self.HALF_OPEN:
            self.half_open_count += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self):
        if self.state == self.HALF_OPEN:
            logger.info("circuit closed (recovery confirmed)")
        self.state = self.CLOSED
        self.fail_count = 0

    def _on_failure(self):
        self.fail_count += 1
        self.last_fail_time = time.time()
        if self.fail_count >= self.fail_threshold:
            self.state = self.OPEN
            logger.error("circuit open after %d consecutive failures", self.fail_count)

class CircuitOpenError(Exception):
    pass

cb = CircuitBreaker()

# ===========================================================================
# HTTP Client
# ===========================================================================
class ClientPool:
    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT),
                limits=httpx.Limits(
                    max_connections=POOL_MAX_SIZE,
                    max_keepalive_connections=POOL_MAX_KEEPALIVE,
                ),
                headers={
                    "Authorization": f"Bearer {RAGFLOW_API_KEY}",
                    "Content-Type": "application/json",
                },
                http2=False,
                transport=httpx.HTTPTransport(retries=RETRY_MAX),
            )
        return self._client

    def close(self):
        if self._client:
            self._client.close()
            self._client = None

client_pool = ClientPool()

# ===========================================================================
# Cache — Redis 优先，回退内存
# ===========================================================================
class DummyCache:
    def get(self, key): return None
    def set(self, key, value, ttl=None): pass
    def clear(self): pass
    @property
    def size(self): return 0

_cache = None

def _init_cache():
    global _cache
    if CACHE_TTL <= 0:
        _cache = DummyCache()
        return
    try:
        import redis
        r = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2, decode_responses=False)
        r.ping()
        _cache = RedisCache(r)
        logger.info("cache redis connected")
    except Exception as e:
        from cachetools import TTLCache
        _cache = MemoryCache(TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL))
        logger.warning("cache fallback to memory (%s)", str(e)[:80])

class RedisCache:
    def __init__(self, client):
        self._r = client
    def _key(self, k):
        return f"ragflow-proxy:{k}"
    def get(self, key):
        raw = self._r.get(self._key(key))
        if raw:
            return json.loads(raw)
        return None
    def set(self, key, value, ttl=None):
        self._r.setex(self._key(key), ttl or CACHE_TTL, json.dumps(value, ensure_ascii=False))
    def clear(self):
        for k in self._r.scan_iter("ragflow-proxy:*"):
            self._r.delete(k)
    @property
    def size(self):
        return sum(1 for _ in self._r.scan_iter("ragflow-proxy:*"))

class MemoryCache:
    def __init__(self, ttl_cache):
        self._c = ttl_cache
    def get(self, key):
        return self._c.get(key)
    def set(self, key, value, ttl=None):
        self._c[key] = value
    def clear(self):
        self._c.clear()
    @property
    def size(self):
        return len(self._c)

_init_cache()

# ===========================================================================
# 请求追踪
# ===========================================================================
@app.before_request
def before_request():
    g.start_time = datetime.utcnow()
    g.request_id = request.headers.get("X-Request-Id", "")

@app.after_request
def after_request(response):
    elapsed = (datetime.utcnow() - g.start_time).total_seconds()
    if g.request_id:
        response.headers["X-Request-Id"] = g.request_id
    if elapsed > SLOW_THRESHOLD:
        logger.warning("slow_request path=%s duration=%.2fs rid=%s", request.path, elapsed, g.request_id)
    response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
    response.headers["X-Proxy"] = "ragflow-dify-proxy/2.1"
    return response

# ===========================================================================
# 健康检查
# ===========================================================================
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    try:
        resp = client_pool.client.get(
            f"{RAGFLOW_BASE_URL}/",
            timeout=httpx.Timeout(5.0, connect=2.0),
        )
        resp.close()
        upstream = "ok"
    except Exception:
        upstream = "unavailable"

    return jsonify({
        "status": "healthy" if upstream == "ok" else "degraded",
        "ragflow": upstream,
        "circuit": cb.state,
        "cache": {
            "enabled": CACHE_TTL > 0,
            "backend": type(_cache).__name__,
            "size": _cache.size,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }), 200 if upstream == "ok" else 503

# ===========================================================================
# 配置查看
# ===========================================================================
@app.route("/config", methods=["GET"])
def show_config():
    """返回当前运行时的配置（敏感信息脱敏）"""
    safe = json.loads(json.dumps(_config))
    # 脱敏 API key
    if safe.get("ragflow", {}).get("api_key"):
        k = safe["ragflow"]["api_key"]
        safe["ragflow"]["api_key"] = k[:8] + "****" + k[-4:] if len(k) > 12 else "****"
    safe["config_file"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxy_config.json")
    return jsonify(safe)

# ===========================================================================
# 核心：/retrieval
# ===========================================================================
SOURCE_URL_RE = re.compile(r'source_url:\s*"([^"]+)"')

def _build_source_url(chunk: dict, content: str) -> str:
    """为 chunk 构造人类可点击的源 URL。"""
    # 1. YAML frontmatter 中的 source_url（docs 常见）
    match = SOURCE_URL_RE.search(content)
    if match:
        return match.group(1)
    # 2. Dify 控制台链接
    dataset_id = chunk.get("dataset_id", "")
    document_id = chunk.get("document_id", "")
    if dataset_id and document_id:
        return f"{DIFY_CONSOLE_BASE}/console/datasets/{dataset_id}/documents/{document_id}"
    # 3. 无可用 URL
    return ""

def _to_dify_records(ragf_data: dict) -> list[dict]:
    records = []
    for chunk in ragf_data.get("data", {}).get("chunks", []):
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        records.append({
            "content": content,
            "score": round(float(
                chunk.get("similarity",
                          chunk.get("vector_similarity", 0))
            ), 4),
            "title": chunk.get("document_name",
                      chunk.get("document_keyword",
                        chunk.get("document_id", ""))),
            "source_url": _build_source_url(chunk, content),
            "metadata": {
                "document_id": chunk.get("document_id", ""),
                "dataset_id": chunk.get("dataset_id", ""),
                "chunk_id": chunk.get("id", ""),
                "positions": chunk.get("positions", []),
            },
        })
    # 按 score 降序，再截断
    records.sort(key=lambda r: r["score"], reverse=True)
    return records


@app.route("/retrieval", methods=["POST"])
def retrieval():
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"records": [], "error": "invalid JSON"}), 400

    if not body:
        return jsonify({"records": []})

    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"records": []})

    knowledge_id = body.get("knowledge_id", "")
    dataset_ids = [k.strip() for k in knowledge_id.split(",") if k.strip()] if knowledge_id else []
    top_k = int(body.get("top_k", DEFAULT_TOP_K))

    # 缓存
    cache_key = f"{query}|{knowledge_id}|{top_k}"
    cached = _cache.get(cache_key)
    if cached is not None:
        logger.info("cache_hit q=%s rid=%s", query[:80], g.request_id)
        return jsonify({"records": cached})

    # 调用 RAGFlow（带熔断）
    logger.info("retrieve kb=%s top=%d q=%s rid=%s", knowledge_id or "all", top_k, query[:100], g.request_id)

    def _call_ragflow():
        resp = client_pool.client.post(
            f"{RAGFLOW_BASE_URL}/api/v1/retrieval",
            json={"question": query, "dataset_ids": dataset_ids, "document_ids": [], "top_k": top_k},
        )
        resp.raise_for_status()
        return resp

    try:
        resp = cb.call(_call_ragflow)
        ragf_data = resp.json()
    except CircuitOpenError:
        logger.warning("circuit_open q=%s rid=%s", query[:80], g.request_id)
        return jsonify({"records": [], "error": "upstream overloaded, retry later"}), 503
    except httpx.TimeoutException:
        logger.error("upstream_timeout q=%s rid=%s", query[:80], g.request_id)
        return jsonify({"records": [], "error": "upstream timeout"}), 504
    except httpx.HTTPStatusError as e:
        logger.error("upstream_http_error status=%d q=%s rid=%s", e.response.status_code, query[:80], g.request_id)
        return jsonify({"records": [], "error": f"upstream {e.response.status_code}"}), 502
    except httpx.ConnectError:
        logger.error("upstream_connect_failed rid=%s", g.request_id)
        return jsonify({"records": [], "error": "upstream unavailable"}), 503
    except Exception as e:
        logger.error("upstream_error %s rid=%s", str(e)[:200], g.request_id)
        return jsonify({"records": [], "error": "upstream error"}), 502

    if ragf_data.get("code") != 0:
        msg = ragf_data.get("message", "unknown")
        logger.warning("ragflow_error code=%s msg=%s rid=%s", ragf_data.get("code"), msg, g.request_id)
        return jsonify({"records": [], "error": f"RAGFlow: {msg}"})

    records = _to_dify_records(ragf_data)
    if len(records) > top_k:
        records = records[:top_k]
    logger.info("retrieve_done records=%d q=%s rid=%s", len(records), query[:80], g.request_id)

    _cache.set(cache_key, records)
    return jsonify({"records": records})


# ===========================================================================
# 反馈收集
# ===========================================================================
@app.route("/feedback", methods=["POST"])
def submit_feedback():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status": "error", "message": "invalid JSON"}), 400

    if not data or "sources" not in data:
        return jsonify({"status": "error", "message": "missing sources"}), 400

    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "query": data.get("query", ""),
        "sources": data["sources"],
        "client": request.remote_addr,
    }

    try:
        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
        with open(FEEDBACK_FILE, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("feedback_recorded sources=%d client=%s rid=%s",
                    len(record["sources"]), record["client"], g.request_id)
        return jsonify({"status": "ok", "written": 1})
    except Exception as e:
        logger.error("feedback_write_error %s", str(e)[:200])
        return jsonify({"status": "error", "message": str(e)}), 500


# ===========================================================================
# 优雅关闭
# ===========================================================================
def _graceful_exit(signum, frame):
    logger.info("received signal %d, shutting down", signum)
    client_pool.close()
    sys.exit(0)

signal.signal(signal.SIGTERM, _graceful_exit)
signal.signal(signal.SIGINT, _graceful_exit)

@app.route("/shutdown", methods=["POST"])
def shutdown():
    client_pool.close()
    os.kill(os.getpid(), signal.SIGTERM)
    return jsonify({"status": "shutting_down"})


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8090
    logger.info("proxy starting on %s:%d (ragflow=%s, cache=%s, circuit=%s)",
                host, port, RAGFLOW_BASE_URL, type(_cache).__name__, cb.state)
    app.run(host=host, port=port)
