import threading
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.db.models import Base
from app.schemas.settings import settings

# Serialize SQLite writes (API + background scheduler share one file).
sqlite_write_lock = threading.Lock()

connect_args = {"check_same_thread": False, "timeout": 60} if settings.is_sqlite else {}

engine_kwargs: dict = {
    "connect_args": connect_args,
    "pool_pre_ping": not settings.is_sqlite,
}
if settings.is_sqlite:
    # Avoid pooled connections holding SQLite write locks.
    engine_kwargs["poolclass"] = NullPool

engine = create_engine(settings.database_url, **engine_kwargs)


@event.listens_for(engine, "connect")
def _configure_sqlite_connection(dbapi_connection, _connection_record) -> None:
    if not settings.is_sqlite:
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=60000")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def sqlite_write():
    """Hold the process-wide SQLite write lock for a batched transaction."""
    if not settings.is_sqlite:
        yield
        return
    with sqlite_write_lock:
        yield


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
