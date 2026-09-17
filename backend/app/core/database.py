"""
core/database.py — Async SQLAlchemy engine, session, startup bootstrap.

Startup sequence:

    1. Connect to the default 'postgres' database
    2. CREATE DATABASE IF NOT EXISTS precursorai
    3. Enable pgvector extension
    4. Import all ORM models
    5. Create missing tables
    6. Synchronize existing table columns with ORM models
       - ADD missing columns
       - DROP extra columns only when AUTO_DROP_COLUMNS=True

IMPORTANT:
    SQLAlchemy's Base.metadata.create_all() does NOT modify existing tables.
    The schema synchronization below handles that for development/prototyping.
"""

import re
import logging
import asyncpg

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from app.core.config import settings


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Automatically add columns that exist in SQLAlchemy models but are missing
# from PostgreSQL.
AUTO_ADD_COLUMNS = True

# WARNING:
# Setting this to True will DROP database columns that are not present in
# SQLAlchemy models.
#
# Keep this FALSE unless you are intentionally removing columns.
AUTO_DROP_COLUMNS = False


# ─────────────────────────────────────────────────────────────────────────────
# SQLAlchemy engine
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_postgres_url(db_url: str) -> tuple[str, str, str]:
    """
    Parse DATABASE_URL and return:

        admin DSN
        database name
        target DSN

    Example:

        postgresql+asyncpg://user:pass@localhost:5432/precursorai

    becomes:

        admin DSN:
        user:pass@localhost:5432/postgres

        database:
        precursorai

        target DSN:
        user:pass@localhost:5432/precursorai
    """

    raw = db_url.replace(
        "postgresql+asyncpg://",
        "postgresql://",
    )

    base, _, db_name = raw.rpartition("/")

    admin_dsn = (
        base.replace("postgresql://", "")
        + "/postgres"
    )

    target_dsn = (
        base.replace("postgresql://", "")
        + f"/{db_name}"
    )

    return admin_dsn, db_name, target_dsn


async def ensure_database_exists() -> None:
    """Create the target database if it does not already exist."""

    admin_dsn, db_name, _ = _build_postgres_url(
        settings.DATABASE_URL
    )

    conn = await asyncpg.connect(
        dsn=f"postgresql://{admin_dsn}"
    )

    try:
        exists = await conn.fetchval(
            """
            SELECT 1
            FROM pg_database
            WHERE datname = $1
            """,
            db_name,
        )

        if not exists:
            # CREATE DATABASE cannot run inside a transaction.
            await conn.execute(
                f'CREATE DATABASE "{db_name}"'
            )

            print(
                f"[bootstrap] Database '{db_name}' created."
            )
        else:
            print(
                f"[bootstrap] Database '{db_name}' already exists."
            )

    finally:
        await conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# pgvector
# ─────────────────────────────────────────────────────────────────────────────

async def enable_pgvector() -> None:
    """
    Enable the pgvector extension in the target database.

    If pgvector is unavailable, the application does not crash here.
    """

    _, _, target_dsn = _build_postgres_url(
        settings.DATABASE_URL
    )

    conn = await asyncpg.connect(
        dsn=f"postgresql://{target_dsn}"
    )

    try:
        await conn.execute(
            "CREATE EXTENSION IF NOT EXISTS vector"
        )

        print(
            "[bootstrap] pgvector extension enabled."
        )

    except asyncpg.exceptions.FeatureNotSupportedError:

        print(
            "[bootstrap] WARNING: pgvector extension is "
            "NOT available on this PostgreSQL instance.\n"
            "[bootstrap] Embedding/similarity features "
            "will fail at runtime.\n"
            "[bootstrap] Use the Docker container "
            "(pgvector/pgvector:pg16) for full functionality."
        )

    finally:
        await conn.close()


async def _pgvector_available() -> bool:
    """Check whether the vector extension is installed."""

    _, _, target_dsn = _build_postgres_url(
        settings.DATABASE_URL
    )

    conn = await asyncpg.connect(
        dsn=f"postgresql://{target_dsn}"
    )

    try:
        result = await conn.fetchval(
            """
            SELECT 1
            FROM pg_extension
            WHERE extname = 'vector'
            """
        )

        return result == 1

    finally:
        await conn.close()


# Tables containing VECTOR columns.
VECTOR_TABLES = {
    "report_embeddings",
    "knowledge_embeddings",
}


# ─────────────────────────────────────────────────────────────────────────────
# SQL identifier safety
# ─────────────────────────────────────────────────────────────────────────────

_IDENTIFIER_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*$"
)


def _safe_identifier(value: str) -> str:
    """
    Validate a PostgreSQL identifier before putting it into SQL.

    Table/column names cannot safely be passed as normal SQL parameters,
    therefore we validate them and quote them ourselves.
    """

    if not _IDENTIFIER_RE.match(value):
        raise ValueError(
            f"Unsafe SQL identifier: {value}"
        )

    return f'"{value}"'


# ─────────────────────────────────────────────────────────────────────────────
# Type conversion
# ─────────────────────────────────────────────────────────────────────────────

def _get_postgres_column_type(column) -> str:
    """
    Convert SQLAlchemy column type into a PostgreSQL type string.

    SQLAlchemy already knows the dialect-specific type. We compile it
    against PostgreSQL so types such as UUID, JSONB, TIMESTAMP, etc.
    are represented correctly.
    """

    from sqlalchemy.dialects import postgresql

    try:
        return column.type.compile(
            dialect=postgresql.dialect()
        )
    except Exception:
        return str(column.type)


def _get_column_definition(column) -> str:
    """
    Build the PostgreSQL definition for a SQLAlchemy column.

    Example:

        BOOLEAN

    or:

        VARCHAR(255)

    or:

        UUID
    """

    column_type = _get_postgres_column_type(column)

    definition = column_type

    # Add NOT NULL only when the ORM column requires it.
    #
    # We intentionally do not add server defaults here because changing
    # existing production data safely requires more careful migrations.
    if not column.nullable:
        definition += " NOT NULL"

    return definition


# ─────────────────────────────────────────────────────────────────────────────
# Existing database inspection
# ─────────────────────────────────────────────────────────────────────────────

async def _get_existing_tables(conn) -> set[str]:
    """Return all user tables in the public schema."""

    rows = await conn.fetch(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        """
    )

    return {
        row["table_name"]
        for row in rows
    }


async def _get_existing_columns(
    conn,
    table_name: str,
) -> dict[str, dict]:
    """
    Return existing columns for a table.

    Result:

        {
            "id": {
                "data_type": "uuid",
                "is_nullable": "NO"
            },
            ...
        }
    """

    rows = await conn.fetch(
        """
        SELECT
            column_name,
            data_type,
            is_nullable,
            udt_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = $1
        """,
        table_name,
    )

    return {
        row["column_name"]: dict(row)
        for row in rows
    }


# ─────────────────────────────────────────────────────────────────────────────
# Schema synchronization
# ─────────────────────────────────────────────────────────────────────────────

async def synchronize_columns() -> None:
    """
    Synchronize PostgreSQL columns with SQLAlchemy ORM models.

    Actions:

        ORM column missing from DB
            -> ADD COLUMN

        DB column missing from ORM
            -> optionally DROP COLUMN

    Existing columns are NOT automatically altered because changing types,
    constraints, or defaults can destroy or invalidate existing data.
    """

    import app.models  # noqa: F401
    # Importing app.models ensures all model modules are registered
    # with Base.metadata.

    _, _, target_dsn = _build_postgres_url(
        settings.DATABASE_URL
    )

    conn = await asyncpg.connect(
        dsn=f"postgresql://{target_dsn}"
    )

    try:
        existing_tables = await _get_existing_tables(conn)

        print(
            "[schema] Checking ORM/database schema..."
        )

        for table_name, table in Base.metadata.tables.items():

            # create_all() should already have created missing tables.
            if table_name not in existing_tables:
                print(
                    f"[schema] Table '{table_name}' does not exist. "
                    "It will be created by create_all()."
                )
                continue

            existing_columns = await _get_existing_columns(
                conn,
                table_name,
            )

            orm_columns = {
                column.name: column
                for column in table.columns
            }

            # ─────────────────────────────────────────
            # ADD MISSING COLUMNS
            # ─────────────────────────────────────────

            if AUTO_ADD_COLUMNS:

                for column_name, column in orm_columns.items():

                    if column_name in existing_columns:
                        continue

                    definition = _get_column_definition(
                        column
                    )

                    safe_table = _safe_identifier(
                        table_name
                    )

                    safe_column = _safe_identifier(
                        column_name
                    )

                    sql = (
                        f"ALTER TABLE {safe_table} "
                        f"ADD COLUMN {safe_column} "
                        f"{definition}"
                    )

                    # A NOT NULL column cannot be added to a table
                    # containing existing rows unless it has a default.
                    #
                    # For prototype safety, add it as nullable first
                    # when necessary.
                    if (
                        not column.nullable
                        and column.default is None
                        and column.server_default is None
                    ):
                        sql = (
                            f"ALTER TABLE {safe_table} "
                            f"ADD COLUMN {safe_column} "
                            f"{_get_postgres_column_type(column)}"
                        )

                        print(
                            f"[schema] Adding missing nullable column "
                            f"'{table_name}.{column_name}' "
                            f"(ORM says NOT NULL; existing rows "
                            f"prevent direct NOT NULL addition)."
                        )

                    else:
                        print(
                            f"[schema] Adding missing column "
                            f"'{table_name}.{column_name}' "
                            f"as {definition}."
                        )

                    await conn.execute(sql)

            # ─────────────────────────────────────────
            # DROP EXTRA COLUMNS
            # ─────────────────────────────────────────

            if AUTO_DROP_COLUMNS:

                for column_name in existing_columns:

                    if column_name in orm_columns:
                        continue

                    # Safety: never automatically remove these.
                    if column_name in {
                        "id",
                        "created_at",
                        "updated_at",
                    }:
                        continue

                    safe_table = _safe_identifier(
                        table_name
                    )

                    safe_column = _safe_identifier(
                        column_name
                    )

                    print(
                        f"[schema] DROPPING extra column "
                        f"'{table_name}.{column_name}'."
                    )

                    await conn.execute(
                        f"ALTER TABLE {safe_table} "
                        f"DROP COLUMN {safe_column}"
                    )

        print(
            "[schema] Schema synchronization complete."
        )

    finally:
        await conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Create tables
# ─────────────────────────────────────────────────────────────────────────────

async def create_tables() -> None:
    """
    Create ORM-registered tables.

    If pgvector is available:
        create all tables.

    If pgvector is unavailable:
        skip embedding tables.
    """

    import app.models  # noqa: F401

    has_vector = await _pgvector_available()

    if has_vector:

        async with engine.begin() as conn:

            await conn.run_sync(
                Base.metadata.create_all
            )

        print(
            "[bootstrap] All tables created "
            "(or already exist)."
        )

    else:

        tables_to_create = [
            table
            for name, table in Base.metadata.tables.items()
            if name not in VECTOR_TABLES
        ]

        async with engine.begin() as conn:

            await conn.run_sync(
                lambda sync_conn:
                Base.metadata.create_all(
                    sync_conn,
                    tables=tables_to_create,
                )
            )

        print(
            "[bootstrap] Core tables created "
            "(or already exist)."
        )

        print(
            f"[bootstrap] SKIPPED vector tables: "
            f"{VECTOR_TABLES}"
        )

        print(
            "[bootstrap] -> Start pgvector/Docker "
            "and restart to enable embedding tables."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Full database initialization
# ─────────────────────────────────────────────────────────────────────────────

async def init_db() -> None:
    """
    Full database bootstrap sequence.

    Runs every time FastAPI starts.

        1. Create database if missing
        2. Enable pgvector
        3. Create missing tables
        4. Synchronize columns
    """

    print(
        "[bootstrap] Starting database initialization..."
    )

    # 1. Database
    await ensure_database_exists()

    # 2. pgvector
    await enable_pgvector()

    # 3. Tables
    await create_tables()

    # 4. Columns
    await synchronize_columns()

    print(
        "[bootstrap] Database initialization complete."
    )