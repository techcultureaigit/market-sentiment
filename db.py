"""Shared database helpers: connection string from .env and versioned migrations."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
MIGRATIONS_DIR = ROOT / "migrations"
load_dotenv(ROOT / ".env")


def get_dsn() -> str:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set (expected in .env or the environment)")
    return dsn


def apply_schema(conn) -> list[str]:
    """Apply all pending SQL migration files in migrations/ in sorted order."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    applied = {
        row[0]
        for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }
    newly_applied = []
    if MIGRATIONS_DIR.exists():
        for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
            version = sql_file.name
            if version not in applied:
                conn.execute(sql_file.read_text(encoding="utf-8"))
                conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (version,),
                )
                newly_applied.append(version)
    return newly_applied


if __name__ == "__main__":
    import psycopg

    with psycopg.connect(get_dsn()) as conn:
        applied = apply_schema(conn)
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        ).fetchall()
    print("Newly applied migrations:", applied or "none (up to date)")
    print("Tables:", ", ".join(t[0] for t in tables))
