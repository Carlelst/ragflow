"""LLM 调用熔断器 (circuit breaker)。

问题背景:
  当算力(vLLM)变慢或不可用时, RAGFlow 的 LLM 请求会挂住 10-20 分钟才超时,
  任务失败后重新入队, executor 又拉起新请求, 形成"持续打爆算力"的恶性循环。
  6 个 executor 进程 × MAX_CONCURRENT_CHATS=80 的并发叠加, 进一步放大压力。

本模块在 LLM 调用统一入口(Base.async_chat)前检查熔断状态:
  closed(关闭) -> 连续失败 threshold 次 -> open(打开)
  open(打开)   -> 冷却 cooldown 秒后 -> half_open(半开)
  half_open     -> 放行 half_open 个试探请求, 全成功 -> closed; 任一失败 -> open

熔断只对"算力侧问题"类错误(超时/连接/服务端 5xx)触发;
  对 401/400/内容过滤等非算力错误不熔断。

注意: 进程内熔断器, 每个 executor 进程各自维护状态。
  跨进程共享需配合 Redis(见 rag/llm/circuit_breaker_redis.py 设计)。
"""
import logging
import os
import time
import threading

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """状态机: closed -> open -> half_open -> (closed | open)"""

    def __init__(
        self,
        name: str = "llm",
        failure_threshold: int | None = None,
        cooldown_seconds: int | None = None,
        half_open_max: int | None = None,
    ):
        self.name = name
        # 连续触发熔断的失败次数(可配)
        self.failure_threshold = failure_threshold or int(
            os.environ.get("LLM_CB_FAILURE_THRESHOLD", 3)
        )
        # 熔断打开后的冷却时长(秒)
        self.cooldown_seconds = cooldown_seconds or int(
            os.environ.get("LLM_CB_COOLDOWN_SECONDS", 60)
        )
        # 半开状态下放行的试探请求数
        self.half_open_max = half_open_max or int(
            os.environ.get("LLM_CB_HALF_OPEN_MAX", 3)
        )

        self.state = "closed"  # closed | open | half_open
        self.failures = 0
        self.opened_at = 0.0
        self.half_open_remaining = 0
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        """返回 True=放行, False=熔断拒绝(调用方应快速失败)。"""
        with self._lock:
            now = time.time()
            if self.state == "open":
                if now - self.opened_at >= self.cooldown_seconds:
                    # 冷却结束, 进入半开, 放行一个试探请求
                    self.state = "half_open"
                    self.half_open_remaining = self.half_open_max
                    logger.warning(
                        f"[circuit-breaker:{self.name}] cooldown over, half-open (allow {self.half_open_max} probes)"
                    )
                    return True
                logger.warning(
                    f"[circuit-breaker:{self.name}] OPEN since {(now - self.opened_at):.0f}s, "
                    f"rejecting LLM request (cooldown {self.cooldown_seconds}s)"
                )
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            if self.state == "half_open":
                self.half_open_remaining -= 1
                if self.half_open_remaining <= 0:
                    self.state = "closed"
                    self.failures = 0
                    logger.info(f"[circuit-breaker:{self.name}] recovered, closed")
            elif self.state == "closed" and self.failures > 0:
                # 关闭态下成功即清零失败计数, 避免历史失败累积误触发熔断
                self.failures = 0

    def record_failure(self, error_code: str | None = None) -> None:
        """记录一次失败。仅当错误码属于算力侧问题(超时/连接/服务端)时累加。

        error_code: LLMErrorCode 字符串 (如 "TIMEOUT", "CONNECTION_ERROR", "SERVER_ERROR")。
                    非算力错误(401/400/内容过滤等)不计入熔断。
        """
        # 只对算力侧错误熔断
        fatal_codes = {"TIMEOUT", "CONNECTION_ERROR", "SERVER_ERROR"}
        err = str(error_code) if error_code is not None else None
        if err and err not in fatal_codes:
            return
        with self._lock:
            if self.state == "half_open":
                # 半开试探失败 -> 立刻回到 open, 重新计时
                self.state = "open"
                self.opened_at = time.time()
                self.half_open_remaining = 0
                logger.warning(
                    f"[circuit-breaker:{self.name}] half-open probe failed ({error_code}), re-open"
                )
                return
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.state = "open"
                self.opened_at = time.time()
                self.half_open_remaining = 0
                logger.warning(
                    f"[circuit-breaker:{self.name}] {self.failures} consecutive failures "
                    f"({error_code}), tripping OPEN for {self.cooldown_seconds}s"
                )

    def reset(self) -> None:
        with self._lock:
            self.state = "closed"
            self.failures = 0
            self.half_open_remaining = 0
            logger.info(f"[circuit-breaker:{self.name}] manually reset to closed")


# 模块级单例: chat/embedding/rerank 共用
llm_circuit_breaker = CircuitBreaker()
