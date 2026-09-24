import threading

_guard = threading.Lock()
_symbol_locks: dict[str, threading.RLock] = {}


def symbol_lock(symbol: str) -> threading.RLock:
    """Serialize first-fetch work for a ticker (Yahoo + SQLite writes)."""
    key = symbol.strip().upper()
    with _guard:
        lock = _symbol_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _symbol_locks[key] = lock
        return lock
