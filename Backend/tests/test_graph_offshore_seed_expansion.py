import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.routers import graph


class _Result:
    def __init__(self, scalar_value=None, rows=None, keys=None):
        self._scalar_value = scalar_value
        self._rows = rows or []
        self._keys = keys or []

    def scalar(self):
        return self._scalar_value

    def fetchall(self):
        return self._rows

    def keys(self):
        return self._keys


class _FakeDB:
    def __init__(self):
        self.sql_log = []
        self.rollback_called = False

    def execute(self, statement, params=None):
        sql = str(statement)
        self.sql_log.append((sql, params or {}))

        if "information_schema.tables" in sql:
            return _Result(scalar_value=True)

        if "FROM offshore_entities" in sql:
            return _Result(rows=[("BVI-0711", "9900000711")])

        if "FROM ownership_links" in sql and "child_tax_code" in sql and "parent_tax_code = :identifier" in sql:
            identifier = (params or {}).get("identifier")
            if identifier == "9900000711":
                return _Result(rows=[("0101000001",), ("0202000002",)])
            if identifier == "BVI-0711":
                return _Result(rows=[("0202000002",), ("0303000003",)])
            return _Result(rows=[])

        if "FROM ownership_links" in sql and "parent_tax_code" in sql and "child_tax_code = :identifier" in sql:
            return _Result(rows=[])

        raise AssertionError(f"Unexpected SQL in fake DB: {sql}")

    def rollback(self):
        self.rollback_called = True


class _SubgraphDB:
    def execute(self, statement, params=None):
        sql = str(statement)

        if "SELECT geom FROM companies LIMIT 1" in sql:
            return _Result(rows=[])

        if "SELECT DISTINCT seller_tax_code, buyer_tax_code" in sql and "FROM invoices" in sql:
            return _Result(rows=[("0101000001", "0202000002")])

        if "SELECT tax_code, name, industry, registration_date, risk_score, is_active" in sql and "FROM companies" in sql:
            return _Result(
                rows=[
                    ("9900000711", "[OFFSHORE] Holding 0711", "Offshore Entity", None, 97.0, True, 0.0, 0.0),
                    ("0101000001", "Cong ty A", "Thuong mai", None, 40.0, True, 0.0, 0.0),
                    ("0202000002", "Cong ty B", "San xuat", None, 35.0, True, 0.0, 0.0),
                ],
                keys=["tax_code", "name", "industry", "registration_date", "risk_score", "is_active", "lat", "lng"],
            )

        if "FROM invoices" in sql and "invoice_number" in sql:
            return _Result(
                rows=[("0101000001", "0202000002", 1500000000.0, 10.0, "2024-10-01", "INV-001")],
                keys=["seller_tax_code", "buyer_tax_code", "amount", "vat_rate", "date", "invoice_number"],
            )

        raise AssertionError(f"Unexpected SQL in fake subgraph DB: {sql}")

    def rollback(self):
        return None


def test_collect_seed_tax_codes_from_ownership_merges_proxy_and_entity_links_without_duplicates():
    db = _FakeDB()

    result = graph._collect_seed_tax_codes_from_ownership(db, "9900000711", limit=10)

    assert result == ["0101000001", "0202000002", "0303000003"]


def test_collect_seed_tax_codes_from_ownership_respects_limit():
    db = _FakeDB()

    result = graph._collect_seed_tax_codes_from_ownership(db, "9900000711", limit=2)

    assert len(result) == 2
    assert result == ["0101000001", "0202000002"]


def test_extract_subgraph_expands_from_ownership_seed(monkeypatch):
    monkeypatch.setattr(
        graph,
        "_collect_seed_tax_codes_from_ownership",
        lambda db, tax_code, limit=40: ["0101000001"],
    )

    companies, invoices = graph._extract_subgraph(
        _SubgraphDB(),
        center_tax_code="9900000711",
        depth=1,
        max_nodes=20,
    )

    company_codes = {row["tax_code"] for row in companies}
    assert {"9900000711", "0101000001", "0202000002"}.issubset(company_codes)
    assert len(invoices) == 1
