from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import SQLQueryResult


class SQLSafetyError(ValueError):
    pass


@dataclass(frozen=True)
class _SQLToken:
    kind: str
    value: str
    raw: str
    quoted: bool = False


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?")
_LIMIT_LITERAL_RE = re.compile(r"[0-9]+\Z")


class ReadOnlySQLGuard:
    TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
        "doctor_schedules": ("doctor_name", "department", "date", "shift", "available_slots"),
        "drug_inventory": ("name", "stock", "unit", "updated_at"),
        "exam_prices": ("test_name", "price", "department", "notes"),
    }
    FORBIDDEN = frozenset(
        {
            "insert",
            "update",
            "delete",
            "drop",
            "alter",
            "truncate",
            "attach",
            "pragma",
            "union",
            "intersect",
            "except",
            "with",
            "recursive",
            "returning",
            "create",
            "replace",
            "vacuum",
            "reindex",
            "explain",
        }
    )
    _CLAUSE_WORDS = frozenset({"where", "order", "limit", "offset", "group", "having"})
    _RESERVED_WORDS = frozenset(
        {
            "and",
            "as",
            "asc",
            "by",
            "desc",
            "from",
            "full",
            "inner",
            "join",
            "left",
            "like",
            "not",
            "null",
            "on",
            "or",
            "right",
            "select",
            "is",
            "distinct",
            "using",
            "cross",
            *FORBIDDEN,
            *_CLAUSE_WORDS,
        }
    )
    _PREDICATE_OPERATORS = frozenset({"=", "!=", "<>", "<", "<=", ">", ">=", "like"})

    def __init__(self, db_path: Path | None = None, backend: Any | None = None) -> None:
        self.db_path = db_path or Path(__file__).resolve().parents[2] / "data" / "runtime" / "medguide.sqlite3"
        self.backend = backend
        if self.backend is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_schema()

    @property
    def backend_name(self) -> str:
        return "mysql" if self.backend is not None else "sqlite-test"

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS doctor_schedules (
                    doctor_name TEXT, department TEXT, date TEXT, shift TEXT, available_slots INTEGER
                );
                CREATE TABLE IF NOT EXISTS drug_inventory (
                    name TEXT, stock INTEGER, unit TEXT, updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS exam_prices (
                    test_name TEXT, price REAL, department TEXT, notes TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_doctor_schedules_department
                    ON doctor_schedules (department);
                CREATE INDEX IF NOT EXISTS idx_drug_inventory_name
                    ON drug_inventory (name);
                CREATE INDEX IF NOT EXISTS idx_exam_prices_test_name
                    ON exam_prices (test_name);
                """
            )
            if conn.execute("SELECT COUNT(*) FROM doctor_schedules").fetchone()[0] == 0:
                conn.executemany("INSERT INTO doctor_schedules VALUES (?, ?, ?, ?, ?)", [
                    ("李医生", "呼吸内科", "2026-06-18", "上午", 12),
                    ("周医生", "消化内科", "2026-06-18", "下午", 8),
                    ("陈医生", "全科医学科", "2026-06-19", "上午", 5),
                ])
            if conn.execute("SELECT COUNT(*) FROM drug_inventory").fetchone()[0] == 0:
                conn.executemany("INSERT INTO drug_inventory VALUES (?, ?, ?, ?)", [
                    ("布洛芬缓释胶囊", 128, "盒", "2026-06-17"),
                    ("氯雷他定片", 64, "盒", "2026-06-17"),
                    ("口服补液盐", 32, "盒", "2026-06-17"),
                ])
            if conn.execute("SELECT COUNT(*) FROM exam_prices").fetchone()[0] == 0:
                conn.executemany("INSERT INTO exam_prices VALUES (?, ?, ?, ?)", [
                    ("血常规", 35.0, "检验科", "以院内实时价格为准"),
                    ("胸部正位片", 80.0, "医学影像科", "孕期请先告知医生"),
                    ("腹部超声", 120.0, "超声医学科", "是否空腹遵医嘱"),
                ])

    @staticmethod
    def _is_word(token: _SQLToken, word: str) -> bool:
        return token.kind == "word" and token.value == word

    @classmethod
    def _identifier(cls, token: _SQLToken) -> str | None:
        if token.kind == "identifier":
            return token.value
        if token.kind == "word" and token.value not in cls._RESERVED_WORDS:
            return token.value
        return None

    @classmethod
    def _tokenize(cls, sql: str) -> list[_SQLToken]:
        """Tokenize enough SQL to validate structure without evaluating expressions."""
        tokens: list[_SQLToken] = []
        index = 0
        while index < len(sql):
            char = sql[index]
            if char.isspace():
                index += 1
                continue
            if char == "\x00":
                raise SQLSafetyError("查询包含非法字符")
            if sql.startswith("--", index) or sql.startswith("/*", index):
                raise SQLSafetyError("查询包含被禁止的 SQL 片段")
            if char == ";":
                raise SQLSafetyError("查询包含被禁止的 SQL 片段")
            if char == "'":
                start = index
                index += 1
                while index < len(sql):
                    if sql[index] != "'":
                        index += 1
                        continue
                    if index + 1 < len(sql) and sql[index + 1] == "'":
                        index += 2
                        continue
                    index += 1
                    break
                else:
                    raise SQLSafetyError("字符串字面量未闭合")
                raw = sql[start:index]
                if "\x00" in raw:
                    raise SQLSafetyError("查询包含非法字符")
                tokens.append(_SQLToken("string", raw, raw))
                continue
            if char in ('"', "`", "["):
                start = index
                closing = "]" if char == "[" else char
                index += 1
                value_start = index
                value: list[str] = []
                while index < len(sql):
                    if sql[index] == closing:
                        if closing != "]" and index + 1 < len(sql) and sql[index + 1] == closing:
                            value.append(sql[value_start:index])
                            value.append(closing)
                            index += 2
                            value_start = index
                            continue
                        value.append(sql[value_start:index])
                        index += 1
                        break
                    index += 1
                else:
                    raise SQLSafetyError("引用标识符未闭合")
                identifier = "".join(value).lower()
                if not identifier or "\x00" in identifier or any(ord(item) > 127 for item in identifier):
                    raise SQLSafetyError("标识符不在白名单内")
                tokens.append(_SQLToken("identifier", identifier, sql[start:index], quoted=True))
                continue
            if _IDENTIFIER_RE.match(sql, index):
                match = _IDENTIFIER_RE.match(sql, index)
                assert match is not None
                raw = match.group(0)
                tokens.append(_SQLToken("word", raw.lower(), raw))
                index = match.end()
                continue
            if "0" <= char <= "9":
                match = _NUMBER_RE.match(sql, index)
                assert match is not None
                raw = match.group(0)
                tokens.append(_SQLToken("number", raw, raw))
                index = match.end()
                continue
            if char == "?":
                tokens.append(_SQLToken("parameter", char, char))
                index += 1
                continue
            if char in ":@$":
                raise SQLSafetyError("仅允许 ? 参数占位符")
            if sql.startswith(("<=", ">=", "<>", "!=", "||"), index):
                raw = sql[index : index + 2]
                tokens.append(_SQLToken("operator", raw, raw))
                index += 2
                continue
            if char in "=<>+-*/%":
                tokens.append(_SQLToken("operator", char, char))
                index += 1
                continue
            if char in ",().*":
                tokens.append(_SQLToken("punctuation", char, char))
                index += 1
                continue
            raise SQLSafetyError("查询包含无法识别的 SQL 片段")
        return tokens

    @classmethod
    def _column_ref(
        cls,
        tokens: list[_SQLToken],
        index: int,
        aliases: dict[str, str],
        *,
        allow_star: bool = False,
    ) -> tuple[str, int]:
        if index >= len(tokens):
            raise SQLSafetyError("查询字段不在白名单内")
        first = cls._identifier(tokens[index])
        if first is None:
            raise SQLSafetyError("查询字段不在白名单内")
        index += 1
        qualifier: str | None = None
        if index < len(tokens) and tokens[index].raw == ".":
            index += 1
            if index >= len(tokens):
                raise SQLSafetyError("查询字段不在白名单内")
            qualifier = first
            if allow_star and tokens[index].raw == "*":
                if qualifier not in aliases:
                    raise SQLSafetyError("查询字段不在白名单内")
                return "*", index + 1
            first = cls._identifier(tokens[index])
            if first is None:
                raise SQLSafetyError("查询字段不在白名单内")
            index += 1
        if qualifier is not None:
            table = aliases.get(qualifier)
            if table is None or first not in cls.TABLE_COLUMNS[table]:
                raise SQLSafetyError("查询字段不在白名单内")
            return first, index
        matches = [table for table in aliases.values() if first in cls.TABLE_COLUMNS[table]]
        if len(matches) != 1:
            raise SQLSafetyError("查询字段不在白名单内")
        return first, index

    @classmethod
    def _validate_projection(cls, tokens: list[_SQLToken], aliases: dict[str, str]) -> None:
        if not tokens:
            raise SQLSafetyError("查询字段不在白名单内")
        index = 0
        if cls._is_word(tokens[index], "distinct"):
            index += 1
        if index >= len(tokens):
            raise SQLSafetyError("查询字段不在白名单内")
        while index < len(tokens):
            if tokens[index].raw == "*":
                raise SQLSafetyError("查询字段必须显式列入白名单")
            else:
                _, index = cls._column_ref(tokens, index, aliases)
                if index < len(tokens) and cls._is_word(tokens[index], "as"):
                    index += 1
                    if index >= len(tokens) or cls._identifier(tokens[index]) is None:
                        raise SQLSafetyError("查询字段不在白名单内")
                    index += 1
            if index == len(tokens):
                return
            if tokens[index].raw != ",":
                raise SQLSafetyError("查询字段不在白名单内")
            index += 1
            if index == len(tokens):
                raise SQLSafetyError("查询字段不在白名单内")

    @classmethod
    def _validate_where(cls, tokens: list[_SQLToken], aliases: dict[str, str]) -> None:
        if not tokens:
            raise SQLSafetyError("WHERE 条件不能为空")
        index = 0
        expect_predicate = True
        while index < len(tokens):
            if not expect_predicate:
                if not (cls._is_word(tokens[index], "and") or cls._is_word(tokens[index], "or")):
                    raise SQLSafetyError("WHERE 条件包含不支持的表达式")
                index += 1
                expect_predicate = True
                if index == len(tokens):
                    raise SQLSafetyError("WHERE 条件不能为空")
            _, index = cls._column_ref(tokens, index, aliases)
            if index >= len(tokens):
                raise SQLSafetyError("WHERE 条件缺少比较运算符")
            if cls._is_word(tokens[index], "is"):
                index += 1
                if index < len(tokens) and cls._is_word(tokens[index], "not"):
                    index += 1
                if index >= len(tokens) or not cls._is_word(tokens[index], "null"):
                    raise SQLSafetyError("WHERE 条件包含不支持的表达式")
                index += 1
            else:
                if cls._is_word(tokens[index], "not"):
                    index += 1
                    if index >= len(tokens) or not cls._is_word(tokens[index], "like"):
                        raise SQLSafetyError("WHERE 条件包含不支持的表达式")
                elif not (
                    tokens[index].kind == "operator"
                    and tokens[index].value in cls._PREDICATE_OPERATORS
                ) and not cls._is_word(tokens[index], "like"):
                    raise SQLSafetyError("WHERE 条件包含不支持的表达式")
                index += 1
                if index >= len(tokens):
                    raise SQLSafetyError("WHERE 条件缺少比较值")
                if (
                    tokens[index].kind in {"string", "number", "parameter"}
                    or tokens[index].quoted
                    or cls._is_word(tokens[index], "null")
                ):
                    index += 1
                else:
                    _, index = cls._column_ref(tokens, index, aliases)
            expect_predicate = False
        if expect_predicate:
            raise SQLSafetyError("WHERE 条件不能为空")

    @classmethod
    def _validate_order(cls, tokens: list[_SQLToken], aliases: dict[str, str]) -> None:
        if len(tokens) < 2 or not cls._is_word(tokens[0], "by"):
            raise SQLSafetyError("ORDER BY 子句格式非法")
        index = 1
        while index < len(tokens):
            _, index = cls._column_ref(tokens, index, aliases)
            if index < len(tokens) and (cls._is_word(tokens[index], "asc") or cls._is_word(tokens[index], "desc")):
                index += 1
            if index == len(tokens):
                return
            if tokens[index].raw != ",":
                raise SQLSafetyError("ORDER BY 子句格式非法")
            index += 1
            if index == len(tokens):
                raise SQLSafetyError("ORDER BY 子句格式非法")

    @staticmethod
    def _decimal_within(raw: str, maximum: int) -> bool:
        canonical = raw.lstrip("0") or "0"
        limit = str(maximum)
        return len(canonical) < len(limit) or (len(canonical) == len(limit) and canonical <= limit)

    @classmethod
    def _parse_table(
        cls,
        tokens: list[_SQLToken],
        index: int,
        aliases: dict[str, str],
    ) -> tuple[int, str]:
        if index >= len(tokens):
            raise SQLSafetyError("表不在只读白名单内")
        table = cls._identifier(tokens[index])
        if table is None or table not in cls.TABLE_COLUMNS:
            raise SQLSafetyError("表不在只读白名单内")
        index += 1
        alias = table
        if index < len(tokens) and cls._is_word(tokens[index], "as"):
            index += 1
            if index >= len(tokens):
                raise SQLSafetyError("表别名格式非法")
            alias = cls._identifier(tokens[index]) or ""
            index += 1
        elif index < len(tokens) and cls._identifier(tokens[index]) is not None:
            alias = cls._identifier(tokens[index]) or ""
            index += 1
        if not alias or alias in aliases:
            raise SQLSafetyError("表别名格式非法")
        aliases[alias] = table
        return index, table

    @classmethod
    def _validate(
        cls,
        sql: str,
    ) -> tuple[str, tuple[str, ...]]:
        if not isinstance(sql, str):
            raise SQLSafetyError("SQL 查询必须是文本")
        tokens = cls._tokenize(sql.strip())
        if not tokens or not cls._is_word(tokens[0], "select"):
            raise SQLSafetyError("仅允许 SELECT 查询")
        for position, token in enumerate(tokens):
            if token.kind == "word" and token.value in cls.FORBIDDEN:
                raise SQLSafetyError("查询包含被禁止的 SQL 片段")
            if token.raw in {"(", ")"}:
                if any(cls._is_word(previous, "limit") for previous in tokens[:position]):
                    raise SQLSafetyError("LIMIT 必须是单个非负整数")
                raise SQLSafetyError("查询包含不支持的子查询或表达式")
        if any(cls._is_word(token, "select") for token in tokens[1:]):
            raise SQLSafetyError("查询包含不支持的子查询或表达式")

        from_positions = [index for index, token in enumerate(tokens) if cls._is_word(token, "from")]
        if len(from_positions) != 1:
            raise SQLSafetyError("查询必须包含唯一 FROM 子句")
        from_index = from_positions[0]

        aliases: dict[str, str] = {}
        index, first_table = cls._parse_table(tokens, from_index + 1, aliases)
        table_names = {first_table}
        while index < len(tokens):
            if cls._is_word(tokens[index], "inner"):
                index += 1
                if index >= len(tokens) or not cls._is_word(tokens[index], "join"):
                    raise SQLSafetyError("JOIN 子句格式非法")
            if index < len(tokens) and cls._is_word(tokens[index], "join"):
                index += 1
                index, joined_table = cls._parse_table(tokens, index, aliases)
                table_names.add(joined_table)
                if index >= len(tokens) or not cls._is_word(tokens[index], "on"):
                    raise SQLSafetyError("JOIN 必须包含安全的 ON 条件")
                index += 1
                _, index = cls._column_ref(tokens, index, aliases)
                if index >= len(tokens) or tokens[index].raw != "=":
                    raise SQLSafetyError("JOIN 必须包含安全的 ON 条件")
                index += 1
                _, index = cls._column_ref(tokens, index, aliases)
                continue
            break

        # Projection is checked again after FROM so aliases and qualified fields are resolvable.
        cls._validate_projection(tokens[1:from_index], aliases)
        clause_order = {"where": 1, "order": 2, "limit": 3, "offset": 4}
        seen: set[str] = set()
        has_limit = False
        while index < len(tokens):
            word = tokens[index].value if tokens[index].kind == "word" else ""
            if word not in clause_order:
                raise SQLSafetyError("查询包含不支持的 SQL 子句")
            if word in seen or any(clause_order[item] > clause_order[word] for item in seen):
                raise SQLSafetyError("SQL 子句顺序或重复项非法")
            seen.add(word)
            index += 1
            start = index
            while index < len(tokens):
                candidate = tokens[index]
                if candidate.kind == "word" and candidate.value in clause_order:
                    break
                index += 1
            part = tokens[start:index]
            if word == "where":
                cls._validate_where(part, aliases)
            elif word == "order":
                cls._validate_order(part, aliases)
            elif word == "limit":
                has_limit = True
                if len(part) != 1 or part[0].kind != "number" or not _LIMIT_LITERAL_RE.fullmatch(part[0].raw):
                    raise SQLSafetyError("LIMIT 必须是单个非负整数")
                if not cls._decimal_within(part[0].raw, 50):
                    raise SQLSafetyError("LIMIT 最大为 50")
            else:
                if not has_limit:
                    raise SQLSafetyError("OFFSET 必须跟随 LIMIT")
                if len(part) != 1 or part[0].kind != "number" or not _LIMIT_LITERAL_RE.fullmatch(part[0].raw):
                    raise SQLSafetyError("OFFSET 必须是单个非负整数")
                if not cls._decimal_within(part[0].raw, 100000):
                    raise SQLSafetyError("OFFSET 最大为 100000")
        validated = sql.strip()
        if not has_limit:
            validated += " LIMIT 50"
        return validated, tuple(sorted(table_names))

    def execute(self, intent: str, sql: str, params: tuple[Any, ...] = ()) -> SQLQueryResult:
        try:
            validated, _ = self._validate(sql)
            placeholder_count = sum(token.kind == "parameter" for token in self._tokenize(validated))
            if placeholder_count != len(params):
                raise SQLSafetyError("SQL 参数数量与占位符不一致")
        except SQLSafetyError as exc:
            return SQLQueryResult(intent, sql, params, (), (), blocked=True, reason=str(exc))
        try:
            if self.backend is not None:
                ensure_ready = getattr(self.backend, "ensure_ready", None)
                backend_ready = bool(ensure_ready()) if callable(ensure_ready) else bool(getattr(self.backend, "ready", False))
                if not backend_ready:
                    raise RuntimeError("MySQL adapter is not ready")
                columns, rows = self.backend.execute(validated, params)
            else:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("PRAGMA case_sensitive_like = ON")
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(validated, params)
                    rows = tuple(dict(row) for row in cursor.fetchall())
                    columns = tuple(column[0] for column in cursor.description or ())
            return SQLQueryResult(intent, validated, params, columns, rows)
        except Exception as exc:
            # Adapter/driver exceptions must not escape the API boundary.  Do
            # not return raw driver text because it can contain DSNs, table
            # names, or other deployment details.
            reason = (
                "只读查询未执行：共享数据源暂时不可用"
                if isinstance(exc, RuntimeError)
                else "只读查询未执行，请稍后重试"
            )
            return SQLQueryResult(intent, validated, params, (), (), blocked=True, reason=reason)

    def from_natural_language(self, text: str) -> SQLQueryResult | None:
        normalized = text.strip()
        if not any(word in normalized for word in ("库存", "有货", "排班", "号源", "价格", "多少钱", "费用")):
            return None
        if any(word in normalized for word in ("库存", "有货")):
            drug = self._extract_term(normalized, ("库存", "有货", "药品"))
            if not drug:
                return SQLQueryResult(
                    "drug_inventory",
                    "",
                    (),
                    (),
                    (),
                    blocked=True,
                    reason="请提供需要查询的具体药品名称",
                )
            sql = "SELECT name, stock, unit, updated_at FROM drug_inventory WHERE name LIKE ? ORDER BY stock DESC LIMIT 50"
            return self.execute("drug_inventory", sql, (f"{drug}%",))
        if any(word in normalized for word in ("排班", "号源", "医生")):
            department = self._extract_department(normalized)
            if not department:
                return SQLQueryResult(
                    "doctor_schedules",
                    "",
                    (),
                    (),
                    (),
                    blocked=True,
                    reason="请提供需要查询的具体科室名称",
                )
            sql = "SELECT doctor_name, department, date, shift, available_slots FROM doctor_schedules WHERE department LIKE ? ORDER BY date, shift LIMIT 50"
            return self.execute("doctor_schedules", sql, (f"{department}%",))
        exam = self._extract_term(normalized, ("价格", "多少钱", "费用", "检查", "检验"))
        if not exam:
            return SQLQueryResult(
                "exam_prices",
                "",
                (),
                (),
                (),
                blocked=True,
                reason="请提供需要查询的具体检查或检验名称",
            )
        sql = "SELECT test_name, price, department, notes FROM exam_prices WHERE test_name LIKE ? ORDER BY price LIMIT 50"
        return self.execute("exam_prices", sql, (f"{exam}%",))

    @staticmethod
    def _extract_term(text: str, markers: tuple[str, ...]) -> str:
        cleaned = text
        for marker in markers:
            cleaned = cleaned.replace(marker, " ")
        cleaned = re.sub(r"[？?，。,、的有多少费用是多少钱查询一下帮我看查]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Natural-language terms are embedded into a bounded LIKE predicate.
        # Reject wildcard and escape metacharacters instead of accidentally
        # turning a request such as ``查库存 %`` into a full-table query.
        if not cleaned or any(character in cleaned for character in ("%", "_", "\\")):
            return ""
        return cleaned

    @staticmethod
    def _extract_department(text: str) -> str:
        for department in ("呼吸内科", "消化内科", "全科医学科", "心内科", "皮肤科"):
            if department in text:
                return department
        return ""
