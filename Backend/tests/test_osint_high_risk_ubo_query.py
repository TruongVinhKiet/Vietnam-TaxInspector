import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.routers import osint


class _Result:
    def __init__(self, scalar_value=None, rows=None):
        self._scalar_value = scalar_value
        self._rows = rows or []

    def scalar(self):
        return self._scalar_value

    def fetchall(self):
        return self._rows


class _FakeDB:
    def __init__(self):
        self.sql_log = []

    def execute(self, statement, params=None):
        sql = str(statement)
        self.sql_log.append(sql)

        if "information_schema.tables" in sql:
            return _Result(scalar_value=True)

        if "SELECT COUNT(*)" in sql and "FROM offshore_entities oe" in sql:
            return _Result(scalar_value=1)

        if "FROM offshore_entities oe" in sql and "ARRAY_REMOVE(ARRAY_AGG(DISTINCT CASE" in sql:
            # Regression lock: this alias usage caused AmbiguousColumn in PostgreSQL.
            assert "GROUP BY oe.entity_code, oe.proxy_tax_code, display_name, country, risk_score" not in sql
            assert "ORDER BY risk_score DESC" not in sql

            return _Result(
                rows=[
                    (
                        "BVI000001",
                        "9900000001",
                        "Offshore Shell A",
                        "British Virgin Islands (BVI)",
                        88.2,
                        2,
                        ["shareholder", None],
                        ["0101000001", "0202000002", "BVI000001", None],
                    )
                ]
            )

        raise AssertionError(f"Unexpected SQL in test double: {sql}")


def test_list_high_risk_ubo_uses_unambiguous_group_and_order_sql():
    db = _FakeDB()

    response = osint.list_high_risk_ubo(page=1, page_size=20, min_risk=60.0, country=None, db=db)

    assert response.total == 1
    assert response.page == 1
    assert response.page_size == 20
    assert len(response.items) == 1

    first = response.items[0]
    assert first.offshore_id == "BVI000001"
    assert first.risk_score == 88.2
    assert first.connected_domestic_count == 2
    assert first.relation_types == ["shareholder"]
    assert first.top_domestic_tax_codes == ["0101000001", "0202000002"]
