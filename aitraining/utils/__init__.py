"""Debugging and logging utilities for Splix training."""

import logging
import sys
import time
from functools import wraps
from typing import Any, Callable, TypeVar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("splix")

# Color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"


def debug_print(msg: str, color: str = Colors.CYAN) -> None:
    """Print debug message with optional color."""
    print(f"{color}[DEBUG]{Colors.RESET} {msg}", file=sys.stderr)


def info_print(msg: str, color: str = Colors.GREEN) -> None:
    """Print info message with optional color."""
    print(f"{color}[INFO]{Colors.RESET} {msg}")


def warn_print(msg: str, color: str = Colors.YELLOW) -> None:
    """Print warning message with optional color."""
    print(f"{color}[WARN]{Colors.RESET} {msg}", file=sys.stderr)


def error_print(msg: str, color: str = Colors.RED) -> None:
    """Print error message with optional color."""
    print(f"{color}[ERROR]{Colors.RESET} {msg}", file=sys.stderr)


F = TypeVar("F", bound=Callable[..., Any])


def timeit(func: F) -> F:
    """Decorator to measure function execution time.
    
    Usage:
        @timeit
        def my_function():
            pass
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        debug_print(f"{func.__name__}() took {elapsed:.3f}s")
        return result
    return wrapper


class Timer:
    """Context manager for timing code blocks.
    
    Usage:
        with Timer("Loading model"):
            model = load_model()
    """
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start: float | None = None
        self.elapsed: float = 0.0
    
    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self
    
    def __exit__(self, *args: Any) -> None:
        if self.start is not None:
            self.elapsed = time.time() - self.start
            debug_print(f"{self.name}: {self.elapsed:.3f}s")


def log_config(config: dict[str, Any], label: str = "Configuration") -> None:
    """Pretty-print configuration dictionary."""
    info_print(f"{label}:")
    for key, value in config.items():
        print(f"  {Colors.DIM}{key}{Colors.RESET}: {value}")
