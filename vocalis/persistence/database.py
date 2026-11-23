"""Short-lived SQLite connections, WAL mode, and numbered SQL migrations."""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from vocalis.core.exceptions import PersistenceError

logger = logging.getLogger("vocalis.persistence")

DEFAULT_TIMEOUT_SECONDS = 5.0
_MIGRATION_NAME = re.compile(r"^(\d{4})_.+\.sql$")
_MIGRATION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class _Migration:
    version: int
    filename: str
    sql: str


def package_migrations_dir() -> Path:
    return Path(__file__).resolve().parent / "migrations"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Database:
    """Open a connection per operation and apply pending SQL migrations."""

    def __init__(
        self,
        path: Path,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        migrations_dir: Path | None = None,
    ) -> None:
        if timeout <= 0:
            raise PersistenceError("The database timeout must be greater than 0.")
        self.path = Path(path)
        self._timeout = timeout
        self._migrations_dir = migrations_dir or package_migrations_dir()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield one configured connection and always close it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            connection = sqlite3.connect(
                self.path,
                timeout=self._timeout,
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            logger.exception("Could not open the generation history database")
            raise PersistenceError("The generation history database is unavailable.") from exc
        try:
            try:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                journal = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            except sqlite3.Error as exc:
                logger.exception("Could not configure the generation history database")
                raise PersistenceError("The generation history database is unavailable.") from exc
            mode = "" if journal is None else str(journal[0]).lower()
            if mode != "wal":
                raise PersistenceError("The database could not enable WAL mode.")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run one write transaction and roll it back when it fails."""
        with self.connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as exc:
                logger.exception("Could not start a database transaction")
                raise PersistenceError("The generation history database is unavailable.") from exc
            try:
                yield connection
                connection.execute("COMMIT")
            except Exception:
                self._rollback(connection)
                raise

    def migrate(self) -> tuple[int, ...]:
        """Apply each pending migration once. Return the versions applied now."""
        with self.transaction() as connection:
            connection.execute(_MIGRATION_TABLE)
        applied = self.applied_versions()
        started: list[int] = []
        for migration in self._load_migrations():
            if migration.version in applied:
                continue
            self._apply(migration)
            started.append(migration.version)
            logger.info("Applied database migration version=%s name=%s", migration.version, migration.filename)
        return tuple(started)

    def applied_versions(self) -> tuple[int, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        return tuple(int(row["version"]) for row in rows)

    def _apply(self, migration: _Migration) -> None:
        statements = sql_statements(migration.sql)
        if not statements:
            raise PersistenceError(f"Migration {migration.filename} is empty.")
        with self.transaction() as connection:
            try:
                for statement in statements:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (migration.version, migration.filename, _utcnow().isoformat()),
                )
            except sqlite3.Error as exc:
                logger.exception(
                    "Database migration failed version=%s name=%s",
                    migration.version,
                    migration.filename,
                )
                raise PersistenceError(f"Migration {migration.filename} failed: {exc}") from exc

    def _load_migrations(self) -> tuple[_Migration, ...]:
        directory = self._migrations_dir
        if not directory.is_dir():
            raise PersistenceError("The database migration directory is missing.")
        found: dict[int, Path] = {}
        for path in sorted(directory.glob("*.sql")):
            match = _MIGRATION_NAME.match(path.name)
            if match is None:
                raise PersistenceError(f"Migration file name is not numbered: {path.name}.")
            version = int(match.group(1))
            if version in found:
                raise PersistenceError(f"Migration version {version} is defined more than once.")
            found[version] = path
        if not found:
            raise PersistenceError("No database migrations are available.")
        loaded: list[_Migration] = []
        for version in sorted(found):
            path = found[version]
            loaded.append(
                _Migration(
                    version=version,
                    filename=path.name,
                    sql=path.read_text(encoding="utf-8"),
                )
            )
        return tuple(loaded)

    @staticmethod
    def _rollback(connection: sqlite3.Connection) -> None:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            logger.exception("Could not roll back a database transaction")


def sql_statements(script: str) -> list[str]:
    """Split a migration into statements. Comments occupy their own lines."""
    statements: list[str] = []
    current: list[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        if not stripped:
            continue
        current.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(current).strip()
            if statement.endswith(";"):
                statement = statement[:-1].strip()
            if statement:
                statements.append(statement)
            current = []
    if current:
        statement = "\n".join(current).strip().rstrip(";").strip()
        if statement:
            statements.append(statement)
    return statements
