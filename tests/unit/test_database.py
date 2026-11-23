"""SQLite setup, migrations, and connection settings."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from vocalis.core.exceptions import PersistenceError
from vocalis.persistence.database import Database, package_migrations_dir, sql_statements


def test_migrate_creates_the_parent_directory_and_enables_wal(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "history.db"
    database = Database(path)

    applied = database.migrate()

    assert applied == (1,)
    assert path.is_file()
    assert path.parent.is_dir()
    with database.connect() as connection:
        journal = connection.execute("PRAGMA journal_mode").fetchone()
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()
        assert journal is not None
        assert str(journal[0]).lower() == "wal"
        assert foreign_keys is not None
        assert int(foreign_keys[0]) == 1


def test_foreign_key_violations_are_rejected(tmp_path: Path) -> None:
    database = Database(tmp_path / "history.db")
    database.migrate()

    with database.connect() as connection:
        connection.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        connection.execute(
            "CREATE TABLE child (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parent(id))"
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO child (parent_id) VALUES (99)")


def test_migrations_run_once(tmp_path: Path) -> None:
    database = Database(tmp_path / "history.db")

    assert database.migrate() == (1,)
    assert database.migrate() == ()
    assert database.applied_versions() == (1,)

    with database.connect() as connection:
        count = connection.execute("SELECT COUNT(*) AS total FROM schema_migrations").fetchone()
    assert count is not None
    assert int(count["total"]) == 1


def test_failed_migration_is_not_recorded_and_can_be_retried(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    original = package_migrations_dir() / "0001_create_generations.sql"
    (migrations / original.name).write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    (migrations / "0002_bad.sql").write_text("NOPE;\n", encoding="utf-8")
    database = Database(tmp_path / "history.db", migrations_dir=migrations)

    with pytest.raises(PersistenceError, match="0002_bad.sql failed"):
        database.migrate()
    assert database.applied_versions() == (1,)

    with pytest.raises(PersistenceError, match="0002_bad.sql failed"):
        database.migrate()
    assert database.applied_versions() == (1,)

    (migrations / "0002_bad.sql").write_text(
        "ALTER TABLE generations ADD COLUMN review_note TEXT;\n",
        encoding="utf-8",
    )
    assert database.migrate() == (2,)
    assert database.applied_versions() == (1, 2)


def test_duplicate_or_unnumbered_migrations_are_rejected(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_a.sql").write_text("CREATE TABLE a (id INTEGER);\n", encoding="utf-8")
    (migrations / "0001_b.sql").write_text("CREATE TABLE b (id INTEGER);\n", encoding="utf-8")
    database = Database(tmp_path / "history.db", migrations_dir=migrations)
    with pytest.raises(PersistenceError, match="defined more than once"):
        database.migrate()

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.sql").write_text("CREATE TABLE notes (id INTEGER);\n", encoding="utf-8")
    with pytest.raises(PersistenceError, match="not numbered"):
        Database(tmp_path / "other.db", migrations_dir=empty).migrate()


def test_a_locked_database_times_out(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    ready = Database(path)
    ready.migrate()
    holder = sqlite3.connect(path, isolation_level=None)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        slow = Database(path, timeout=0.2)
        with pytest.raises(PersistenceError, match="unavailable"):
            slow.migrate()
    finally:
        holder.close()


def test_sql_statements_skip_comments() -> None:
    script = """
    -- comment
    CREATE TABLE example (id INTEGER);

    CREATE INDEX idx_example ON example (id);
    """
    assert sql_statements(script) == [
        "CREATE TABLE example (id INTEGER)",
        "CREATE INDEX idx_example ON example (id)",
    ]
