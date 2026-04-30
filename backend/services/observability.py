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


@contextmanager
def memory_delta_block(event: str):
    start_memory = rss_mb()
    start = time.perf_counter()
    try:
        yield
    finally:
        end_memory = rss_mb()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if start_memory >= 0 and end_memory >= 0:
            logger.info(
                "[Perf] %s | elapsed_ms=%.2f | rss_start_mb=%.2f | rss_end_mb=%.2f | rss_delta_mb=%.2f",
                event,
                elapsed_ms,
                start_memory,
                end_memory,
                end_memory - start_memory,
            )
        else:
            logger.info("[Perf] %s | elapsed_ms=%.2f | rss_delta_mb=unavailable", event, elapsed_ms)
