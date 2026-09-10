import sqlite3

import pytest

from app.sql_guard import ReadOnlySQLGuard


def test_natural_language_query_is_read_only_and_limited() -> None:
    result = ReadOnlySQLGuard().from_natural_language("帮我查一下布洛芬库存")
    assert result is not None
    assert result.blocked is False
    assert "LIMIT 50" in result.sql
    assert result.rows[0]["name"] == "布洛芬缓释胶囊"


@pytest.mark.parametrize(
    ("text", "table", "index_name", "parameter"),
    [
        ("帮我查一下布洛芬库存", "drug_inventory", "idx_drug_inventory_name", "布洛芬%"),
        ("呼吸内科医生排班", "doctor_schedules", "idx_doctor_schedules_department", "呼吸内科%"),
        ("血常规检查价格", "exam_prices", "idx_exam_prices_test_name", "血常规%"),
    ],
)
def test_nl2sql_templates_use_indexable_prefix_search(
    text: str,
    table: str,
    index_name: str,
    parameter: str,
) -> None:
    guard = ReadOnlySQLGuard()
    result = guard.from_natural_language(text)

    assert result is not None and result.blocked is False
    assert result.params == (parameter,)
    assert not str(result.params[0]).startswith("%")

    with sqlite3.connect(guard.db_path) as connection:
        connection.execute("PRAGMA case_sensitive_like = ON")
        plan = connection.execute(f"EXPLAIN QUERY PLAN {result.sql}", result.params).fetchall()

    assert any(index_name in str(row[3]) and f"SEARCH {table}" in str(row[3]) for row in plan)


def test_dangerous_sql_is_blocked() -> None:
    result = ReadOnlySQLGuard().execute("x", "SELECT name FROM drug_inventory; DROP TABLE drug_inventory")
    assert result.blocked is True
    assert "禁止" in (result.reason or "")


def test_unknown_table_is_blocked() -> None:
    result = ReadOnlySQLGuard().execute("x", "SELECT name FROM patient_records")
    assert result.blocked is True
    assert "白名单" in (result.reason or "")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name FROM drug_inventory LIMIT -1",
        "SELECT name FROM drug_inventory LIMIT +100",
        "SELECT name FROM drug_inventory LIMIT 0,100",
        "SELECT name FROM drug_inventory LIMIT 51",
        "SELECT name FROM drug_inventory LIMIT ?",
        "SELECT name FROM drug_inventory LIMIT 1 + 1",
        "SELECT name FROM drug_inventory LIMIT 0x20",
        "SELECT name FROM drug_inventory LIMIT 1.0",
        "SELECT name FROM drug_inventory LIMIT (SELECT 1)",
    ],
)
def test_limit_must_be_a_single_literal_at_most_50(sql: str) -> None:
    result = ReadOnlySQLGuard().execute("x", sql)
    assert result.blocked is True


def test_huge_limit_is_blocked_without_integer_overflow() -> None:
    result = ReadOnlySQLGuard().execute("x", "SELECT name FROM drug_inventory LIMIT " + "9" * 5000)
    assert result.blocked is True
    assert "LIMIT" in (result.reason or "")


def test_limit_keyword_inside_literal_does_not_disable_default_limit() -> None:
    result = ReadOnlySQLGuard().execute("x", "SELECT name FROM drug_inventory WHERE name = 'limit 100'")
    assert result.blocked is False
    assert result.sql.endswith("LIMIT 50")

    quoted = ReadOnlySQLGuard().execute("x", 'SELECT name FROM drug_inventory WHERE name = "limit 100"')
    assert quoted.blocked is False
    assert quoted.sql.endswith("LIMIT 50")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name FROM (SELECT name FROM drug_inventory) LIMIT 1",
        "WITH rows AS (SELECT name FROM drug_inventory) SELECT name FROM rows LIMIT 1",
        "SELECT name FROM drug_inventory UNION SELECT test_name FROM exam_prices LIMIT 1",
        "SELECT name, (SELECT stock FROM drug_inventory) FROM drug_inventory LIMIT 1",
        "SELECT sqlite_version() FROM drug_inventory LIMIT 1",
        "SELECT name || stock FROM drug_inventory LIMIT 1",
        "SELECT name FROM drug_inventory /* LIMIT 1 */",
        "SELECT name FROM drug_inventory -- LIMIT 1",
        "SELECT name FROM drug_inventory; SELECT name FROM drug_inventory",
    ],
)
def test_query_shape_is_strictly_read_only(sql: str) -> None:
    result = ReadOnlySQLGuard().execute("x", sql)
    assert result.blocked is True


def test_parameterized_predicate_remains_supported() -> None:
    result = ReadOnlySQLGuard().execute(
        "x",
        "SELECT name, stock FROM drug_inventory WHERE name LIKE ? ORDER BY stock DESC LIMIT 10",
        ("%布洛芬%",),
    )
    assert result.blocked is False
    assert result.rows[0]["name"] == "布洛芬缓释胶囊"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM drug_inventory LIMIT 1",
        "SELECT d.* FROM drug_inventory AS d LIMIT 1",
    ],
)
def test_wildcards_cannot_bypass_the_field_allowlist(sql: str) -> None:
    result = ReadOnlySQLGuard().execute("x", sql)
    assert result.blocked is True
    assert "字段" in (result.reason or "")


@pytest.mark.parametrize("placeholder", (":name", "@name", "$name"))
def test_only_portable_qmark_parameters_are_allowed(placeholder: str) -> None:
    result = ReadOnlySQLGuard().execute(
        "x",
        f"SELECT name FROM drug_inventory WHERE name = {placeholder} LIMIT 1",
        ("布洛芬缓释胶囊",),
    )
    assert result.blocked is True
    assert "占位符" in (result.reason or "")


def test_parameter_count_is_validated_before_the_backend_call() -> None:
    result = ReadOnlySQLGuard().execute(
        "x",
        "SELECT name FROM drug_inventory WHERE name = ? LIMIT 1",
    )
    assert result.blocked is True
    assert "参数数量" in (result.reason or "")


@pytest.mark.parametrize("text", ("查库存", "医生排班", "检查价格"))
def test_nl2sql_requires_a_specific_lookup_term(text: str) -> None:
    result = ReadOnlySQLGuard().from_natural_language(text)
    assert result is not None
    assert result.blocked is True
    assert "具体" in (result.reason or "")


@pytest.mark.parametrize("text", ("查库存 %", "查库存 _", r"查库存 \\"))
def test_nl2sql_rejects_wildcard_lookup_terms(text: str) -> None:
    result = ReadOnlySQLGuard().from_natural_language(text)
    assert result is not None
    assert result.blocked is True
    assert "具体" in (result.reason or "") or "通配" in (result.reason or "")
