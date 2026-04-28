import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def rss_mb() -> float:
    """Return current resident memory in MB."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def log_memory(event: str) -> None:
    memory = rss_mb()
    if memory >= 0:
        logger.info("[Perf] %s | rss_mb=%.2f", event, memory)
    else:
        logger.info("[Perf] %s | rss_mb=unavailable", event)


@contextmanager
def timed_block(event: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        memory = rss_mb()
        if memory >= 0:
            logger.info(
                "[Perf] %s | elapsed_ms=%.2f | rss_mb=%.2f",
                event,
                elapsed_ms,
                memory,
            )
        else:
            logger.info("[Perf] %s | elapsed_ms=%.2f", event, elapsed_ms)
