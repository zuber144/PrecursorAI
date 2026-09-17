"""
core/database.py — Async SQLAlchemy engine, session, and startup bootstrap.

On startup (via init_db):
  1. Connect to the default 'postgres' database
  2. CREATE DATABASE IF NOT EXISTS precursorai
  3. Connect to precursorai and CREATE EXTENSION IF NOT EXISTS vector
  4. Create all ORM tables via Base.metadata.create_all()
"""
import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


# ── SQLAlchemy engine ─────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.APP_ENV == "development",
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ── Bootstrap helpers ─────────────────────────────────────────────────────────

def _build_postgres_url(db_url: str) -> tuple[str, str]:
    """
    Parse the DATABASE_URL and return:
      - A URL pointing to the default 'postgres' admin database
      - The target database name
    e.g. "postgresql+asyncpg://user:pass@host:5432/precursorai"
         → ("postgresql+asyncpg://user:pass@host:5432/postgres", "precursorai")
    """
    # Strip the SQLAlchemy dialect prefix for asyncpg direct usage
    raw = db_url.replace("postgresql+asyncpg://", "postgresql://")
    # Split off the database name
    base, _, db_name = raw.rpartition("/")
    # asyncpg DSN format
    admin_dsn = base.replace("postgresql://", "") + "/postgres"
    target_dsn = base.replace("postgresql://", "") + f"/{db_name}"
    return admin_dsn, db_name, target_dsn


async def ensure_database_exists() -> None:
    """Create the target database if it doesn't already exist."""
    admin_dsn, db_name, _ = _build_postgres_url(settings.DATABASE_URL)

    # asyncpg requires autocommit for CREATE DATABASE — use a raw connection
    conn = await asyncpg.connect(dsn=f"postgresql://{admin_dsn}")
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name
        )
        if not exists:
            # CREATE DATABASE cannot run inside a transaction block
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            print(f"[bootstrap] Database '{db_name}' created.")
        else:
            print(f"[bootstrap] Database '{db_name}' already exists.")
    finally:
        await conn.close()


async def enable_pgvector() -> None:
    """
    Enable the pgvector extension in the target database.
    Prints a warning (does NOT crash) if the extension is not installed on this
    PostgreSQL instance — this happens when connecting to a local Postgres instead
    of the Docker container (pgvector/pgvector:pg16) which has it pre-installed.
    """
    _, _, target_dsn = _build_postgres_url(settings.DATABASE_URL)
    conn = await asyncpg.connect(dsn=f"postgresql://{target_dsn}")
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("[bootstrap] pgvector extension enabled.")
    except asyncpg.exceptions.FeatureNotSupportedError:
        print(
            "[bootstrap] WARNING: pgvector extension is NOT available on this PostgreSQL instance.\n"
            "           Embedding/similarity features will fail at runtime.\n"
            "           Use the Docker container (pgvector/pgvector:pg16) for full functionality:\n"
            "             docker-compose up -d"
        )
    finally:
        await conn.close()


async def _pgvector_available() -> bool:
    """Check if the vector extension is installed and active in the target database."""
    _, _, target_dsn = _build_postgres_url(settings.DATABASE_URL)
    conn = await asyncpg.connect(dsn=f"postgresql://{target_dsn}")
    try:
        result = await conn.fetchval(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )
        return result == 1
    finally:
        await conn.close()


# Tables that require the pgvector extension (contain VECTOR columns)
VECTOR_TABLES = {"report_embeddings", "knowledge_embeddings"}


async def create_tables() -> None:
    """
    Create ORM-registered tables.
    If pgvector is not available, skips the embedding tables and prints a notice.
    All other tables are always created.
    """
    import app.models  # noqa: F401 — triggers __init__.py imports

    has_vector = await _pgvector_available()

    if has_vector:
        # Create everything
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("[bootstrap] All tables created (or already exist).")
    else:
        # Create only the non-vector tables by excluding embedding tables
        tables_to_create = [
            t for name, t in Base.metadata.tables.items()
            if name not in VECTOR_TABLES
        ]
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: Base.metadata.create_all(
                    sync_conn, tables=tables_to_create
                )
            )
        print(
            "[bootstrap] Core tables created (or already exist).\n"
            f"[bootstrap] SKIPPED vector tables: {VECTOR_TABLES}\n"
            "[bootstrap]   -> Run 'docker-compose up -d' then restart to enable embedding tables."
        )


async def init_db() -> None:
    """
    Full database bootstrap sequence — called once on application startup.
    Safe to call repeatedly (all operations are idempotent).

    Steps:
      1. Create the target database if missing
      2. Enable pgvector extension (warns but does NOT crash if unavailable)
      3. Create all tables (skips vector tables if pgvector is unavailable)
    """
    await ensure_database_exists()
    await enable_pgvector()
    await create_tables()

