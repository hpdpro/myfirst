"""임대 가계부 기본 동작 테스트.  실행: python -m pytest rental_ledger -q"""

from datetime import date, timedelta

import pytest

import app as ledger


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "DB_PATH", str(tmp_path / "test.db"))
    ledger.init_db()
    ledger.app.config["TESTING"] = True
    with ledger.app.test_client() as c:
        yield c


def add_unit(client, **kw):
    data = {
        "name": "101호", "tenant": "홍길동", "contract_type": "월세",
        "deposit": "10,000,000", "monthly_rent": "500000", "maintenance": "50000",
        "pay_day": "5", "start_date": "2026-01-01", "end_date": "2027-12-31",
    }
    data.update(kw)
    return client.post("/units/new", data=data, follow_redirects=True)


def add_tx(client, **kw):
    data = {"tx_date": "2026-11-05", "kind": "수입", "category": "월세", "unit_id": "1", "amount": "500000"}
    data.update(kw)
    return client.post("/transactions/new", data=data, follow_redirects=True)


def test_pages_render_empty(client):
    for url in ["/", "/units", "/units/new", "/transactions", "/transactions/new", "/report"]:
        assert client.get(url).status_code == 200, url


def test_unit_create_parses_commas(client):
    resp = add_unit(client)
    assert "101호" in resp.get_data(as_text=True)
    with ledger.app.app_context():
        row = ledger.get_db().execute("SELECT deposit FROM units").fetchone()
    assert row["deposit"] == 10_000_000


def test_duplicate_unit_name_rejected(client):
    add_unit(client)
    resp = add_unit(client)
    assert "이미 있습니다" in resp.get_data(as_text=True)


def test_invalid_contract_dates_rejected(client):
    resp = add_unit(client, start_date="2027-01-01", end_date="2026-01-01")
    assert "만기일이 입주일보다 빠릅니다" in resp.get_data(as_text=True)


def test_rent_status_transitions(client):
    add_unit(client)
    with ledger.app.app_context():
        assert ledger.rent_status(ledger.get_db(), 2026, 11)[0]["state"] == "미납"
    add_tx(client, amount="300000")
    with ledger.app.app_context():
        assert ledger.rent_status(ledger.get_db(), 2026, 11)[0]["state"] == "일부납부"
    add_tx(client, category="관리비", amount="250000")
    with ledger.app.app_context():
        row = ledger.rent_status(ledger.get_db(), 2026, 11)[0]
    assert row["state"] == "완납" and row["paid"] == 550_000 and row["due"] == 550_000


def test_vacant_and_out_of_contract_not_billed(client):
    add_unit(client, name="102호", tenant="")
    add_unit(client, name="103호", start_date="2027-01-01", end_date="2028-12-31")
    with ledger.app.app_context():
        states = {r["unit"]["name"]: r for r in ledger.rent_status(ledger.get_db(), 2026, 11)}
    assert states["102호"]["state"] == "공실/계약없음" and states["102호"]["due"] == 0
    assert states["103호"]["state"] == "공실/계약없음"


def test_dashboard_totals(client):
    add_unit(client)
    add_tx(client, amount="500000")
    add_tx(client, kind="지출", category="수리·보수", unit_id="", amount="120000")
    html = client.get("/?month=2026-11").get_data(as_text=True)
    assert "500,000원" in html and "120,000원" in html and "380,000원" in html


def test_transaction_validation(client):
    resp = add_tx(client, amount="0")
    assert "금액은 0보다 커야 합니다" in resp.get_data(as_text=True)
    resp = add_tx(client, tx_date="2026-13-40")
    assert "날짜를 올바르게" in resp.get_data(as_text=True)


def test_transaction_edit_and_delete(client):
    add_unit(client)
    add_tx(client)
    client.post("/transactions/1/edit", data={
        "tx_date": "2026-11-06", "kind": "수입", "category": "월세", "unit_id": "1", "amount": "450000",
    })
    with ledger.app.app_context():
        assert ledger.get_db().execute("SELECT amount FROM transactions").fetchone()["amount"] == 450_000
    client.post("/transactions/1/delete")
    with ledger.app.app_context():
        assert ledger.get_db().execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0


def test_unit_delete_keeps_transactions(client):
    add_unit(client)
    add_tx(client)
    client.post("/units/1/delete")
    with ledger.app.app_context():
        row = ledger.get_db().execute("SELECT unit_id FROM transactions").fetchone()
    assert row["unit_id"] is None


def test_month_filter_and_report(client):
    add_unit(client)
    add_tx(client, tx_date="2026-10-05")
    add_tx(client, tx_date="2026-11-05", amount="510000")
    html = client.get("/transactions?month=2026-10").get_data(as_text=True)
    assert "2026-10-05" in html and "2026-11-05" not in html
    html = client.get("/report?year=2026").get_data(as_text=True)
    assert "1,010,000원" in html


def test_expiring_contracts_on_dashboard(client):
    soon = (date.today() + timedelta(days=30)).isoformat()
    add_unit(client, name="201호", tenant="김만기", start_date="2025-01-01", end_date=soon)
    html = client.get("/").get_data(as_text=True)
    assert "김만기" in html and soon in html


def test_month_parsing_and_shift():
    assert ledger.parse_month("2026-02") == (2026, 2)
    assert ledger.shift_month(2026, 1, -1) == (2025, 12)
    assert ledger.shift_month(2026, 12, 1) == (2027, 1)
    assert ledger.month_range(2028, 2) == ("2028-02-01", "2028-02-29")
