from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Callable


DEFAULT_PASSWORD_ITERATIONS = 600_000
DEFAULT_SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_COOKIE_NAME = "medguide_session"
_COOKIE_NAME_PATTERN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_DUMMY_SALT = bytes.fromhex("787e0cf39ac12e1e47c79964d10a499a")
_DUMMY_HASH = bytes.fromhex(
    "da2f3c18f2b2e9d771aa5ebdace55584"
    "82fcf316f57c5198aeb471b42f5ed960"
)


class AuthError(RuntimeError):
    """Base class for expected authentication failures."""


class UsernameUnavailableError(AuthError):
    """Raised when a normalized username is already registered."""


class InvalidCredentialsError(AuthError):
    """Raised for an unknown username or an incorrect password."""


class InvalidUsernameError(AuthError):
    """Raised when a username cannot be represented safely."""


class InvalidPasswordError(AuthError):
    """Raised when a password is outside the supported bounds."""


def _environment_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _environment_ttl() -> int:
    raw = (os.getenv("MEDGUIDE_AUTH_SESSION_TTL_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_SESSION_TTL_SECONDS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("MEDGUIDE_AUTH_SESSION_TTL_SECONDS must be an integer") from exc
    if not 300 <= value <= 90 * 24 * 60 * 60:
        raise ValueError("MEDGUIDE_AUTH_SESSION_TTL_SECONDS must be between 300 and 7776000")
    return value


def normalize_username(value: str) -> str:
    username = unicodedata.normalize("NFKC", str(value).strip())
    if not 3 <= len(username) <= 64:
        raise InvalidUsernameError("username must contain between 3 and 64 characters")
    if not username[0].isalnum() or not all(character.isalnum() or character in "._-" for character in username):
        raise InvalidUsernameError("username may contain letters, numbers, dots, underscores, and hyphens")
    return username


def _password_bytes(value: str) -> bytes:
    if not 8 <= len(value) <= 128:
        raise InvalidPasswordError("password must contain between 8 and 128 characters")
    encoded = value.encode("utf-8")
    if len(encoded) > 512:
        raise InvalidPasswordError("password is too long after UTF-8 encoding")
    return encoded


def _is_unique_constraint_error(error: BaseException) -> bool:
    original = getattr(error, "orig", error)
    sqlite_code = getattr(original, "sqlite_errorcode", None)
    if sqlite_code is not None:
        return sqlite_code in {sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY, sqlite3.SQLITE_CONSTRAINT_UNIQUE}
    return bool(original.args) and original.args[0] == 1062


class AuthStore:
    """SQLite-backed user and opaque browser-session repository."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
        cookie_name: str = DEFAULT_COOKIE_NAME,
        cookie_secure: bool = True,
        password_iterations: int = DEFAULT_PASSWORD_ITERATIONS,
        clock: Callable[[], float] = time.time,
        requires_shared_backend: bool = False,
    ) -> None:
        if not _COOKIE_NAME_PATTERN.fullmatch(cookie_name):
            raise ValueError("MEDGUIDE_AUTH_COOKIE_NAME is not a valid cookie name")
        if not 300 <= int(session_ttl_seconds) <= 90 * 24 * 60 * 60:
            raise ValueError("authentication session TTL is outside the supported range")
        if not 100_000 <= int(password_iterations) <= 2_000_000:
            raise ValueError("PBKDF2 iteration count is outside the supported range")

        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_ttl_seconds = int(session_ttl_seconds)
        self.cookie_name = cookie_name
        self.cookie_secure = bool(cookie_secure)
        self.password_iterations = int(password_iterations)
        self._clock = clock
        self._requires_shared_backend = bool(requires_shared_backend)
        self._initialize()

    @property
    def backend_name(self) -> str:
        return "sqlite"

    @property
    def shared(self) -> bool:
        return False

    @property
    def requires_shared_backend(self) -> bool:
        return self._requires_shared_backend

    @classmethod
    def from_environment(cls, *, production: bool) -> AuthStore | MySQLAuthStore:
        project_root = Path(__file__).resolve().parents[2]
        configured_path = (os.getenv("MEDGUIDE_AUTH_DB_PATH") or "").strip()
        database_path = Path(configured_path) if configured_path else project_root / "data" / "runtime" / "auth.sqlite3"
        cookie_name = (os.getenv("MEDGUIDE_AUTH_COOKIE_NAME") or DEFAULT_COOKIE_NAME).strip()
        backend = (os.getenv("MEDGUIDE_AUTH_BACKEND") or ("mysql" if production else "sqlite")).strip().lower()
        if backend not in {"sqlite", "mysql"}:
            raise ValueError("MEDGUIDE_AUTH_BACKEND must be mysql or sqlite")
        session_ttl = _environment_ttl()
        cookie_secure = _environment_bool("MEDGUIDE_COOKIE_SECURE", default=production)
        if production and backend == "mysql":
            auth_dsn = (os.getenv("MEDGUIDE_AUTH_DSN") or "").strip()
            if auth_dsn:
                return MySQLAuthStore(
                    auth_dsn,
                    session_ttl_seconds=session_ttl,
                    cookie_name=cookie_name,
                    cookie_secure=cookie_secure,
                )
            # Keep startup observable so health can return a controlled
            # degraded response, but mark the SQLite fallback as unsuitable
            # for a multi-worker production deployment.
            requires_shared = True
        else:
            requires_shared = False
            if production and not _environment_bool("MEDGUIDE_ALLOW_SQLITE_AUTH", default=False):
                requires_shared = True
        return cls(
            database_path,
            session_ttl_seconds=session_ttl,
            cookie_name=cookie_name,
            cookie_secure=cookie_secure,
            requires_shared_backend=requires_shared,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY COLLATE BINARY,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    password_iterations INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                ) WITHOUT ROWID
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_hash BLOB PRIMARY KEY,
                    username TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                ) WITHOUT ROWID
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS auth_sessions_expiry_idx ON auth_sessions (expires_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS auth_sessions_username_idx ON auth_sessions (username)"
            )

    @staticmethod
    def _derive_password(password: bytes, salt: bytes, iterations: int) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password, salt, iterations, dklen=32)

    @staticmethod
    def _token_hash(token: str) -> bytes:
        return hashlib.sha256(token.encode("utf-8")).digest()

    def register(self, username: str, password: str) -> str:
        normalized = normalize_username(username)
        password_bytes = _password_bytes(password)
        salt = secrets.token_bytes(16)
        password_hash = self._derive_password(password_bytes, salt, self.password_iterations)
        now = int(self._clock())
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO users (
                        username, password_salt, password_hash, password_iterations, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (normalized, salt, password_hash, self.password_iterations, now),
                )
        except sqlite3.IntegrityError as exc:
            raise UsernameUnavailableError("username is already registered") from exc
        return normalized

    def register_and_create_session(
        self, username: str, password: str, *, previous_token: str | None = None
    ) -> tuple[str, str, int]:
        normalized = normalize_username(username)
        password_bytes = _password_bytes(password)
        salt = secrets.token_bytes(16)
        password_hash = self._derive_password(password_bytes, salt, self.password_iterations)
        now = int(self._clock())
        expires_at = now + self.session_ttl_seconds
        last_error: sqlite3.IntegrityError | None = None

        for _ in range(3):
            token = secrets.token_urlsafe(32)
            operation = "user"
            try:
                with self._connect() as connection:
                    connection.execute(
                        """
                        INSERT INTO users (
                            username, password_salt, password_hash, password_iterations, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (normalized, salt, password_hash, self.password_iterations, now),
                    )
                    operation = "cleanup"
                    connection.execute("DELETE FROM auth_sessions WHERE expires_at <= ?", (now,))
                    operation = "session"
                    connection.execute(
                        """
                        INSERT INTO auth_sessions (token_hash, username, created_at, expires_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (self._token_hash(token), normalized, now, expires_at),
                    )
                    operation = "cleanup"
                    if previous_token and len(previous_token) <= 512:
                        connection.execute(
                            "DELETE FROM auth_sessions WHERE token_hash = ?",
                            (self._token_hash(previous_token),),
                        )
                    operation = "commit"
                return normalized, token, expires_at
            except sqlite3.IntegrityError as exc:
                if not _is_unique_constraint_error(exc):
                    raise
                if operation == "user":
                    raise UsernameUnavailableError("username is already registered") from exc
                if operation != "session":
                    raise
                last_error = exc

        raise RuntimeError("could not allocate a unique authentication token") from last_error

    def verify_credentials(self, username: str, password: str) -> str:
        try:
            normalized = normalize_username(username)
            password_bytes = _password_bytes(password)
        except (InvalidUsernameError, InvalidPasswordError) as exc:
            raise InvalidCredentialsError("invalid username or password") from exc

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT password_salt, password_hash, password_iterations
                FROM users
                WHERE username = ?
                """,
                (normalized,),
            ).fetchone()

        salt = bytes(row["password_salt"]) if row is not None else _DUMMY_SALT
        expected = bytes(row["password_hash"]) if row is not None else _DUMMY_HASH
        iterations = int(row["password_iterations"]) if row is not None else self.password_iterations
        if not 100_000 <= iterations <= 2_000_000:
            raise InvalidCredentialsError("invalid username or password")
        actual = self._derive_password(password_bytes, salt, iterations)
        if row is None or not hmac.compare_digest(actual, expected):
            raise InvalidCredentialsError("invalid username or password")
        return normalized

    def create_session(self, username: str) -> tuple[str, int]:
        normalized = normalize_username(username)
        now = int(self._clock())
        expires_at = now + self.session_ttl_seconds
        for _ in range(3):
            token = secrets.token_urlsafe(32)
            try:
                with self._connect() as connection:
                    connection.execute("DELETE FROM auth_sessions WHERE expires_at <= ?", (now,))
                    connection.execute(
                        """
                        INSERT INTO auth_sessions (token_hash, username, created_at, expires_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (self._token_hash(token), normalized, now, expires_at),
                    )
                return token, expires_at
            except sqlite3.IntegrityError:
                continue
        raise RuntimeError("could not allocate a unique authentication token")

    def authenticate_session(self, token: str | None) -> str | None:
        if not token or len(token) > 512:
            return None
        token_hash = self._token_hash(token)
        now = int(self._clock())
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT username, expires_at
                FROM auth_sessions
                WHERE token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            if int(row["expires_at"]) <= now:
                connection.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,))
                return None
            return str(row["username"])

    def revoke_session(self, token: str | None) -> None:
        if not token or len(token) > 512:
            return
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?",
                (self._token_hash(token),),
            )


class MySQLAuthStore:
    """Shared MySQL-backed account/session repository for multi-worker deployments."""

    def __init__(
        self,
        dsn: str,
        *,
        session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
        cookie_name: str = DEFAULT_COOKIE_NAME,
        cookie_secure: bool = True,
        password_iterations: int = DEFAULT_PASSWORD_ITERATIONS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not dsn.strip():
            raise ValueError("authentication DSN is required")
        if not _COOKIE_NAME_PATTERN.fullmatch(cookie_name):
            raise ValueError("MEDGUIDE_AUTH_COOKIE_NAME is not a valid cookie name")
        if not 300 <= int(session_ttl_seconds) <= 90 * 24 * 60 * 60:
            raise ValueError("authentication session TTL is outside the supported range")
        if not 100_000 <= int(password_iterations) <= 2_000_000:
            raise ValueError("PBKDF2 iteration count is outside the supported range")
        from sqlalchemy import create_engine

        self.dsn = dsn.strip()
        self.engine = create_engine(
            self.dsn,
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_size=5,
            max_overflow=5,
        )
        self.session_ttl_seconds = int(session_ttl_seconds)
        self.cookie_name = cookie_name
        self.cookie_secure = bool(cookie_secure)
        self.password_iterations = int(password_iterations)
        self._clock = clock
        self._closed = False
        self._initialize()

    @property
    def backend_name(self) -> str:
        return "mysql"

    @property
    def shared(self) -> bool:
        return True

    @property
    def requires_shared_backend(self) -> bool:
        return True

    def _initialize(self) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            connection.execute(text(
                """
                CREATE TABLE IF NOT EXISTS medguide_users (
                    username VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin PRIMARY KEY,
                    password_salt VARBINARY(64) NOT NULL,
                    password_hash VARBINARY(64) NOT NULL,
                    password_iterations INT NOT NULL,
                    created_at BIGINT NOT NULL
                ) ENGINE=InnoDB
                """
            ))
            connection.execute(text(
                """
                CREATE TABLE IF NOT EXISTS medguide_auth_sessions (
                    token_hash VARBINARY(32) PRIMARY KEY,
                    username VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
                    created_at BIGINT NOT NULL,
                    expires_at BIGINT NOT NULL,
                    CONSTRAINT medguide_auth_user_fk
                      FOREIGN KEY (username) REFERENCES medguide_users(username)
                      ON DELETE CASCADE,
                    INDEX medguide_auth_expiry_idx (expires_at),
                    INDEX medguide_auth_username_idx (username)
                ) ENGINE=InnoDB
                """
            ))

    @staticmethod
    def _derive_password(password: bytes, salt: bytes, iterations: int) -> bytes:
        return AuthStore._derive_password(password, salt, iterations)

    @staticmethod
    def _token_hash(token: str) -> bytes:
        return AuthStore._token_hash(token)

    def register(self, username: str, password: str) -> str:
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        normalized = normalize_username(username)
        password_bytes = _password_bytes(password)
        salt = secrets.token_bytes(16)
        password_hash = self._derive_password(password_bytes, salt, self.password_iterations)
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO medguide_users "
                        "(username, password_salt, password_hash, password_iterations, created_at) "
                        "VALUES (:username, :salt, :hash, :iterations, :created_at)"
                    ),
                    {
                        "username": normalized,
                        "salt": salt,
                        "hash": password_hash,
                        "iterations": self.password_iterations,
                        "created_at": int(self._clock()),
                    },
                )
        except IntegrityError as exc:
            raise UsernameUnavailableError("username is already registered") from exc
        return normalized

    def register_and_create_session(
        self, username: str, password: str, *, previous_token: str | None = None
    ) -> tuple[str, str, int]:
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        normalized = normalize_username(username)
        password_bytes = _password_bytes(password)
        salt = secrets.token_bytes(16)
        password_hash = self._derive_password(password_bytes, salt, self.password_iterations)
        now = int(self._clock())
        expires_at = now + self.session_ttl_seconds
        last_error: IntegrityError | None = None

        for _ in range(3):
            token = secrets.token_urlsafe(32)
            operation = "user"
            try:
                with self.engine.begin() as connection:
                    connection.execute(
                        text(
                            "INSERT INTO medguide_users "
                            "(username, password_salt, password_hash, password_iterations, created_at) "
                            "VALUES (:username, :salt, :hash, :iterations, :created_at)"
                        ),
                        {
                            "username": normalized,
                            "salt": salt,
                            "hash": password_hash,
                            "iterations": self.password_iterations,
                            "created_at": now,
                        },
                    )
                    operation = "cleanup"
                    connection.execute(
                        text("DELETE FROM medguide_auth_sessions WHERE expires_at <= :now"),
                        {"now": now},
                    )
                    operation = "session"
                    connection.execute(
                        text(
                            "INSERT INTO medguide_auth_sessions "
                            "(token_hash, username, created_at, expires_at) "
                            "VALUES (:token_hash, :username, :created_at, :expires_at)"
                        ),
                        {
                            "token_hash": self._token_hash(token),
                            "username": normalized,
                            "created_at": now,
                            "expires_at": expires_at,
                        },
                    )
                    operation = "cleanup"
                    if previous_token and len(previous_token) <= 512:
                        connection.execute(
                            text("DELETE FROM medguide_auth_sessions WHERE token_hash = :token_hash"),
                            {"token_hash": self._token_hash(previous_token)},
                        )
                    operation = "commit"
                return normalized, token, expires_at
            except IntegrityError as exc:
                if not _is_unique_constraint_error(exc):
                    raise
                if operation == "user":
                    raise UsernameUnavailableError("username is already registered") from exc
                if operation != "session":
                    raise
                last_error = exc

        raise RuntimeError("could not allocate a unique authentication token") from last_error

    def verify_credentials(self, username: str, password: str) -> str:
        from sqlalchemy import text

        try:
            normalized = normalize_username(username)
            password_bytes = _password_bytes(password)
        except (InvalidUsernameError, InvalidPasswordError) as exc:
            raise InvalidCredentialsError("invalid username or password") from exc
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT password_salt, password_hash, password_iterations "
                    "FROM medguide_users WHERE username = :username"
                ),
                {"username": normalized},
            ).mappings().first()
        salt = bytes(row["password_salt"]) if row is not None else _DUMMY_SALT
        expected = bytes(row["password_hash"]) if row is not None else _DUMMY_HASH
        iterations = int(row["password_iterations"]) if row is not None else self.password_iterations
        if not 100_000 <= iterations <= 2_000_000:
            raise InvalidCredentialsError("invalid username or password")
        actual = self._derive_password(password_bytes, salt, iterations)
        if row is None or not hmac.compare_digest(actual, expected):
            raise InvalidCredentialsError("invalid username or password")
        return normalized

    def create_session(self, username: str) -> tuple[str, int]:
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        normalized = normalize_username(username)
        now = int(self._clock())
        expires_at = now + self.session_ttl_seconds
        for _ in range(3):
            token = secrets.token_urlsafe(32)
            try:
                with self.engine.begin() as connection:
                    connection.execute(
                        text("DELETE FROM medguide_auth_sessions WHERE expires_at <= :now"),
                        {"now": now},
                    )
                    connection.execute(
                        text(
                            "INSERT INTO medguide_auth_sessions "
                            "(token_hash, username, created_at, expires_at) "
                            "VALUES (:token_hash, :username, :created_at, :expires_at)"
                        ),
                        {
                            "token_hash": self._token_hash(token),
                            "username": normalized,
                            "created_at": now,
                            "expires_at": expires_at,
                        },
                    )
                return token, expires_at
            except IntegrityError:
                continue
        raise RuntimeError("could not allocate a unique authentication token")

    def authenticate_session(self, token: str | None) -> str | None:
        from sqlalchemy import text

        if not token or len(token) > 512:
            return None
        now = int(self._clock())
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    "SELECT username, expires_at FROM medguide_auth_sessions "
                    "WHERE token_hash = :token_hash"
                ),
                {"token_hash": self._token_hash(token)},
            ).mappings().first()
            if row is None:
                return None
            if int(row["expires_at"]) <= now:
                connection.execute(
                    text("DELETE FROM medguide_auth_sessions WHERE token_hash = :token_hash"),
                    {"token_hash": self._token_hash(token)},
                )
                return None
            return str(row["username"])

    def revoke_session(self, token: str | None) -> None:
        from sqlalchemy import text

        if not token or len(token) > 512:
            return
        with self.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM medguide_auth_sessions WHERE token_hash = :token_hash"),
                {"token_hash": self._token_hash(token)},
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.engine.dispose()

    def shutdown(self) -> None:
        self.close()
