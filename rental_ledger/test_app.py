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


# ---------------------------------------------------------------- contract history
def q(sql, *args):
    with ledger.app.app_context():
        return ledger.get_db().execute(sql, args).fetchall()


def test_end_contract_moves_to_history_and_records_deposit_return(client):
    add_unit(client)
    add_tx(client, tx_date="2026-03-05")
    resp = client.post("/units/1/end", data={
        "end_date": "2026-06-30", "record_return": "1", "return_amount": "9,800,000", "memo": "원상복구 20만원 공제",
    }, follow_redirects=True)
    html = resp.get_data(as_text=True)
    assert "지난 계약 이력" in html and "홍길동" in html and "2026-06-30" in html

    c = q("SELECT * FROM contracts")
    assert len(c) == 1 and c[0]["tenant"] == "홍길동" and c[0]["deposit"] == 10_000_000
    assert c[0]["start_date"] == "2026-01-01" and "원상복구" in c[0]["memo"]

    u = q("SELECT * FROM units")[0]
    assert u["tenant"] == "" and u["deposit"] == 0 and u["monthly_rent"] == 500_000  # 월세는 다음 계약 참고용으로 유지

    ret = q("SELECT * FROM transactions WHERE category = '보증금 반환'")
    assert len(ret) == 1 and ret[0]["amount"] == 9_800_000 and ret[0]["tx_date"] == "2026-06-30"
    assert ret[0]["unit_id"] == 1


def test_end_contract_without_deposit_return(client):
    add_unit(client)
    client.post("/units/1/end", data={"end_date": "2026-06-30"})
    assert len(q("SELECT * FROM contracts")) == 1
    assert q("SELECT COUNT(*) FROM transactions")[0][0] == 0


def test_end_contract_on_vacant_unit_is_refused(client):
    add_unit(client, tenant="")
    resp = client.post("/units/1/end", data={"end_date": "2026-06-30"}, follow_redirects=True)
    assert "현재 계약이 없습니다" in resp.get_data(as_text=True)
    assert len(q("SELECT * FROM contracts")) == 0


def test_past_month_status_uses_past_contract(client):
    add_unit(client, tenant="홍길동", monthly_rent="500000", maintenance="0",
             start_date="2025-01-01", end_date="2026-12-31")
    client.post("/units/1/end", data={"end_date": "2026-05-31"})
    # 새 임차인: 6월부터, 월세 인상
    client.post("/units/1/edit", data={"name": "101호", "tenant": "김새봄", "monthly_rent": "550000",
                                       "maintenance": "0", "pay_day": "5", "start_date": "2026-06-01",
                                       "end_date": "2028-05-31"})
    with ledger.app.app_context():
        db = ledger.get_db()
        may = ledger.rent_status(db, 2026, 5)[0]
        june = ledger.rent_status(db, 2026, 6)[0]
        old = ledger.rent_status(db, 2024, 12)[0]
    assert may["tenant"] == "홍길동" and may["due"] == 500_000 and may["is_past"] and may["state"] == "미납"
    assert june["tenant"] == "김새봄" and june["due"] == 550_000 and not june["is_past"]
    assert old["state"] == "공실/계약없음"


def test_manual_past_contract_and_overlap_check(client):
    add_unit(client)
    data = {"tenant": "이전세", "contract_type": "전세", "deposit": "100000000",
            "start_date": "2022-01-01", "end_date": "2023-12-31"}
    client.post("/units/1/history/new", data=data)
    assert len(q("SELECT * FROM contracts")) == 1

    resp = client.post("/units/1/history/new", data={**data, "tenant": "겹침", "start_date": "2023-06-01",
                                                      "end_date": "2024-06-01"}, follow_redirects=True)
    assert "기간이 겹칩니다" in resp.get_data(as_text=True)
    assert len(q("SELECT * FROM contracts")) == 1

    resp = client.post("/units/1/history/new", data={"tenant": "날짜없음"}, follow_redirects=True)
    assert "퇴거일을 입력하세요" in resp.get_data(as_text=True)


def test_edit_and_delete_past_contract(client):
    add_unit(client)
    client.post("/units/1/history/new", data={"tenant": "이전", "start_date": "2022-01-01", "end_date": "2023-12-31"})
    # 자기 자신과는 겹침 검사하지 않음
    client.post("/contracts/1/edit", data={"tenant": "이전(수정)", "start_date": "2022-02-01", "end_date": "2023-12-31"})
    assert q("SELECT tenant FROM contracts")[0]["tenant"] == "이전(수정)"
    client.post("/contracts/1/delete")
    assert len(q("SELECT * FROM contracts")) == 0


def test_unit_detail_page(client):
    add_unit(client)
    add_tx(client, tx_date="2026-02-05")
    add_tx(client, tx_date="2026-03-05")
    client.post("/units/1/history/new", data={"tenant": "옛임차인", "start_date": "2024-01-01", "end_date": "2025-12-31"})
    html = client.get("/units/1").get_data(as_text=True)
    assert "옛임차인" in html and "홍길동" in html and "1,000,000원" in html
    assert client.get("/units/99").status_code == 404
    assert client.get("/contracts/99/edit").status_code == 404


def test_deleting_unit_removes_history(client):
    add_unit(client)
    client.post("/units/1/history/new", data={"tenant": "이전", "end_date": "2023-12-31"})
    client.post("/units/1/delete")
    assert len(q("SELECT * FROM contracts")) == 0
