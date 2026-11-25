"""다가구 임대 가계부 - Flask + SQLite 웹 애플리케이션."""

import calendar
import os
import sqlite3
from datetime import date, datetime, timedelta

from flask import Flask, abort, flash, g, redirect, render_template, request, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("RENTAL_DB", os.path.join(BASE_DIR, "rental.db"))

CONTRACT_TYPES = ["월세", "반전세", "전세"]
INCOME_CATEGORIES = ["월세", "관리비", "보증금 수령", "기타 수입"]
EXPENSE_CATEGORIES = [
    "보증금 반환", "수리·보수", "공과금", "청소·관리", "세금", "대출이자", "보험료", "중개수수료", "기타 지출",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS units (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL UNIQUE,
    tenant        TEXT NOT NULL DEFAULT '',
    phone         TEXT NOT NULL DEFAULT '',
    contract_type TEXT NOT NULL DEFAULT '월세',
    deposit       INTEGER NOT NULL DEFAULT 0,
    monthly_rent  INTEGER NOT NULL DEFAULT 0,
    maintenance   INTEGER NOT NULL DEFAULT 0,
    pay_day       INTEGER NOT NULL DEFAULT 1,
    start_date    TEXT NOT NULL DEFAULT '',
    end_date      TEXT NOT NULL DEFAULT '',
    memo          TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS transactions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    tx_date   TEXT NOT NULL,
    kind      TEXT NOT NULL CHECK (kind IN ('수입', '지출')),
    category  TEXT NOT NULL,
    unit_id   INTEGER REFERENCES units(id) ON DELETE SET NULL,
    amount    INTEGER NOT NULL CHECK (amount >= 0),
    memo      TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(tx_date);

-- 종료된(지난) 계약 이력. 현재 계약은 units 테이블에 있습니다.
CREATE TABLE IF NOT EXISTS contracts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id       INTEGER NOT NULL REFERENCES units(id) ON DELETE CASCADE,
    tenant        TEXT NOT NULL,
    phone         TEXT NOT NULL DEFAULT '',
    contract_type TEXT NOT NULL DEFAULT '월세',
    deposit       INTEGER NOT NULL DEFAULT 0,
    monthly_rent  INTEGER NOT NULL DEFAULT 0,
    maintenance   INTEGER NOT NULL DEFAULT 0,
    start_date    TEXT NOT NULL DEFAULT '',
    end_date      TEXT NOT NULL,
    memo          TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_contracts_unit ON contracts(unit_id, end_date);
"""

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")


# ---------------------------------------------------------------- DB helpers
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------- utilities
@app.template_filter("won")
def won(value):
    return f"{int(value or 0):,}원"


def parse_int(value, default=0):
    try:
        return int(str(value).replace(",", "").strip() or default)
    except ValueError:
        return default


def parse_month(value):
    """'YYYY-MM' 문자열을 (year, month)로 변환. 잘못되면 이번 달."""
    try:
        d = datetime.strptime(value or "", "%Y-%m")
        return d.year, d.month
    except ValueError:
        today = date.today()
        return today.year, today.month


def month_range(year, month):
    last = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last:02d}"


def shift_month(year, month, delta):
    idx = year * 12 + (month - 1) + delta
    return idx // 12, idx % 12 + 1


def is_valid_date(value):
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except (TypeError, ValueError):
        return False


def is_active(unit, on_date):
    """계약 기간 중인 호실인지 (임차인이 있고 날짜가 계약기간 안)."""
    if not unit["tenant"]:
        return False
    if unit["start_date"] and unit["start_date"] > on_date:
        return False
    if unit["end_date"] and unit["end_date"] < on_date:
        return False
    return True


def overlaps_month(unit, start, end):
    """해당 월 중 하루라도 계약 기간에 걸치는지."""
    if not unit["tenant"]:
        return False
    if unit["start_date"] and unit["start_date"] > end:
        return False
    if unit["end_date"] and unit["end_date"] < start:
        return False
    return True


def contract_for_month(unit, past_contracts, start, end):
    """해당 월에 적용되는 계약. 현재 계약이 우선이고, 없으면 지난 계약 중 가장 최근 것."""
    if overlaps_month(unit, start, end):
        return unit
    for c in past_contracts:  # end_date 내림차순
        if overlaps_month(c, start, end):
            return c
    return None


def rent_status(db, year, month):
    """호실별 해당 월의 약정액 대비 입금액(월세+관리비)."""
    start, end = month_range(year, month)
    paid = {
        row["unit_id"]: row["total"]
        for row in db.execute(
            """SELECT unit_id, SUM(amount) AS total FROM transactions
               WHERE kind = '수입' AND category IN ('월세', '관리비')
                 AND tx_date BETWEEN ? AND ? AND unit_id IS NOT NULL
               GROUP BY unit_id""",
            (start, end),
        )
    }
    past = {}
    for c in db.execute("SELECT * FROM contracts ORDER BY end_date DESC"):
        past.setdefault(c["unit_id"], []).append(c)

    rows = []
    for unit in db.execute("SELECT * FROM units ORDER BY name"):
        contract = contract_for_month(unit, past.get(unit["id"], []), start, end)
        due = contract["monthly_rent"] + contract["maintenance"] if contract else 0
        amount_paid = paid.get(unit["id"], 0)
        if contract is None:
            state = "공실/계약없음"
        elif due == 0:
            state = "해당없음"
        elif amount_paid >= due:
            state = "완납"
        elif amount_paid > 0:
            state = "일부납부"
        else:
            state = "미납"
        rows.append({
            "unit": unit, "due": due, "paid": amount_paid, "state": state,
            "tenant": contract["tenant"] if contract else "",
            "is_past": contract is not None and contract is not unit,
        })
    return rows


# ---------------------------------------------------------------- dashboard
@app.route("/")
def dashboard():
    db = get_db()
    year, month = parse_month(request.args.get("month"))
    start, end = month_range(year, month)

    totals = {
        row["kind"]: row["total"]
        for row in db.execute(
            "SELECT kind, SUM(amount) AS total FROM transactions WHERE tx_date BETWEEN ? AND ? GROUP BY kind",
            (start, end),
        )
    }
    income, expense = totals.get("수입", 0), totals.get("지출", 0)

    by_category = db.execute(
        """SELECT kind, category, SUM(amount) AS total FROM transactions
           WHERE tx_date BETWEEN ? AND ? GROUP BY kind, category ORDER BY kind DESC, total DESC""",
        (start, end),
    ).fetchall()

    status = rent_status(db, year, month)
    unpaid_total = sum(max(r["due"] - r["paid"], 0) for r in status)

    units = db.execute("SELECT * FROM units").fetchall()
    deposit_total = sum(u["deposit"] for u in units if u["tenant"])
    occupied = sum(1 for u in units if is_active(u, date.today().isoformat()))

    today = date.today()
    expiring = db.execute(
        """SELECT * FROM units WHERE tenant != '' AND end_date != ''
           AND end_date BETWEEN ? AND ? ORDER BY end_date""",
        (today.isoformat(), (today + timedelta(days=90)).isoformat()),
    ).fetchall()

    py, pm = shift_month(year, month, -1)
    ny, nm = shift_month(year, month, 1)
    return render_template(
        "dashboard.html",
        year=year, month=month,
        prev_month=f"{py:04d}-{pm:02d}", next_month=f"{ny:04d}-{nm:02d}",
        income=income, expense=expense, by_category=by_category,
        status=status, unpaid_total=unpaid_total,
        deposit_total=deposit_total, occupied=occupied, unit_count=len(units),
        expiring=expiring,
    )


# ---------------------------------------------------------------- units
def unit_from_form(form):
    return {
        "name": form.get("name", "").strip(),
        "tenant": form.get("tenant", "").strip(),
        "phone": form.get("phone", "").strip(),
        "contract_type": form.get("contract_type") if form.get("contract_type") in CONTRACT_TYPES else "월세",
        "deposit": parse_int(form.get("deposit")),
        "monthly_rent": parse_int(form.get("monthly_rent")),
        "maintenance": parse_int(form.get("maintenance")),
        "pay_day": min(max(parse_int(form.get("pay_day"), 1), 1), 31),
        "start_date": form.get("start_date", "").strip(),
        "end_date": form.get("end_date", "").strip(),
        "memo": form.get("memo", "").strip(),
    }


def validate_unit(data):
    errors = []
    if not data["name"]:
        errors.append("호실 이름을 입력하세요.")
    for key, label in (("start_date", "입주일"), ("end_date", "만기일")):
        if data[key] and not is_valid_date(data[key]):
            errors.append(f"{label} 형식이 올바르지 않습니다.")
    if data["start_date"] and data["end_date"] and data["start_date"] > data["end_date"]:
        errors.append("만기일이 입주일보다 빠릅니다.")
    return errors


@app.route("/units")
def units():
    rows = get_db().execute("SELECT * FROM units ORDER BY name").fetchall()
    return render_template("units.html", units=rows, today=date.today().isoformat(), is_active=is_active)


@app.route("/units/new", methods=["GET", "POST"])
@app.route("/units/<int:unit_id>/edit", methods=["GET", "POST"])
def unit_form(unit_id=None):
    db = get_db()
    unit = None
    if unit_id is not None:
        unit = db.execute("SELECT * FROM units WHERE id = ?", (unit_id,)).fetchone()
        if unit is None:
            abort(404)

    if request.method == "POST":
        data = unit_from_form(request.form)
        errors = validate_unit(data)
        if not errors:
            cols = list(data)
            try:
                if unit is None:
                    db.execute(
                        f"INSERT INTO units ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                        [data[c] for c in cols],
                    )
                else:
                    db.execute(
                        f"UPDATE units SET {', '.join(f'{c} = ?' for c in cols)} WHERE id = ?",
                        [data[c] for c in cols] + [unit_id],
                    )
                db.commit()
                flash("호실 정보를 저장했습니다.")
                if unit_id is not None:
                    return redirect(url_for("unit_detail", unit_id=unit_id))
                return redirect(url_for("units"))
            except sqlite3.IntegrityError:
                errors.append("같은 이름의 호실이 이미 있습니다.")
        for e in errors:
            flash(e, "error")
        unit = data

    return render_template("unit_form.html", unit=unit, unit_id=unit_id, contract_types=CONTRACT_TYPES)


@app.route("/units/<int:unit_id>/delete", methods=["POST"])
def unit_delete(unit_id):
    db = get_db()
    db.execute("DELETE FROM units WHERE id = ?", (unit_id,))
    db.commit()
    flash("호실과 지난 계약 이력을 삭제했습니다. (거래 내역은 '공통'으로 남습니다)")
    return redirect(url_for("units"))


# ---------------------------------------------------------------- unit detail & contract history
def get_unit_or_404(db, unit_id):
    unit = db.execute("SELECT * FROM units WHERE id = ?", (unit_id,)).fetchone()
    if unit is None:
        abort(404)
    return unit


def rent_income(db, unit_id, start, end):
    """기간 중 해당 호실의 월세+관리비 입금 합계."""
    sql = """SELECT COALESCE(SUM(amount), 0) FROM transactions
             WHERE unit_id = ? AND kind = '수입' AND category IN ('월세', '관리비')"""
    params = [unit_id]
    if start:
        sql += " AND tx_date >= ?"
        params.append(start)
    if end:
        sql += " AND tx_date <= ?"
        params.append(end)
    return db.execute(sql, params).fetchone()[0]


@app.route("/units/<int:unit_id>")
def unit_detail(unit_id):
    db = get_db()
    unit = get_unit_or_404(db, unit_id)
    history = []
    for c in db.execute("SELECT * FROM contracts WHERE unit_id = ? ORDER BY end_date DESC", (unit_id,)):
        history.append({"c": c, "received": rent_income(db, unit_id, c["start_date"], c["end_date"])})
    current_received = (
        rent_income(db, unit_id, unit["start_date"], unit["end_date"] or None) if unit["tenant"] else 0
    )
    txs = db.execute(
        "SELECT * FROM transactions WHERE unit_id = ? ORDER BY tx_date DESC, id DESC", (unit_id,)
    ).fetchall()
    income = sum(t["amount"] for t in txs if t["kind"] == "수입")
    expense = sum(t["amount"] for t in txs if t["kind"] == "지출")
    return render_template(
        "unit_detail.html", unit=unit, history=history, current_received=current_received,
        txs=txs, income=income, expense=expense,
    )


def contract_from_form(form):
    return {
        "tenant": form.get("tenant", "").strip(),
        "phone": form.get("phone", "").strip(),
        "contract_type": form.get("contract_type") if form.get("contract_type") in CONTRACT_TYPES else "월세",
        "deposit": parse_int(form.get("deposit")),
        "monthly_rent": parse_int(form.get("monthly_rent")),
        "maintenance": parse_int(form.get("maintenance")),
        "start_date": form.get("start_date", "").strip(),
        "end_date": form.get("end_date", "").strip(),
        "memo": form.get("memo", "").strip(),
    }


def validate_contract(db, unit_id, data, contract_id=None):
    errors = []
    if not data["tenant"]:
        errors.append("임차인 이름을 입력하세요.")
    if data["start_date"] and not is_valid_date(data["start_date"]):
        errors.append("입주일 형식이 올바르지 않습니다.")
    if not is_valid_date(data["end_date"]):
        errors.append("퇴거일을 입력하세요.")
    if errors:
        return errors
    if data["start_date"] and data["start_date"] > data["end_date"]:
        errors.append("퇴거일이 입주일보다 빠릅니다.")
    start = data["start_date"] or "0000-00-00"
    for c in db.execute("SELECT * FROM contracts WHERE unit_id = ? AND id != ?", (unit_id, contract_id or 0)):
        if (c["start_date"] or "0000-00-00") <= data["end_date"] and c["end_date"] >= start:
            errors.append(f"다른 지난 계약({c['tenant']}, {c['start_date'] or '?'} ~ {c['end_date']})과 기간이 겹칩니다.")
    return errors


def insert_contract(db, unit_id, data):
    cols = list(data)
    db.execute(
        f"INSERT INTO contracts (unit_id, {', '.join(cols)}) VALUES (?, {', '.join('?' for _ in cols)})",
        [unit_id] + [data[c] for c in cols],
    )


@app.route("/units/<int:unit_id>/end", methods=["GET", "POST"])
def contract_end(unit_id):
    """현재 계약을 종료해 이력으로 옮기고 호실을 공실로 만듭니다."""
    db = get_db()
    unit = get_unit_or_404(db, unit_id)
    if not unit["tenant"]:
        flash("이 호실에는 현재 계약이 없습니다.", "error")
        return redirect(url_for("unit_detail", unit_id=unit_id))

    form = {
        "end_date": unit["end_date"] if unit["end_date"] and unit["end_date"] <= date.today().isoformat()
        else date.today().isoformat(),
        "record_return": "1" if unit["deposit"] else "",
        "return_amount": unit["deposit"],
        "memo": "",
    }
    if request.method == "POST":
        form = {
            "end_date": request.form.get("end_date", "").strip(),
            "record_return": request.form.get("record_return", ""),
            "return_amount": parse_int(request.form.get("return_amount"), -1),
            "memo": request.form.get("memo", "").strip(),
        }
        data = {k: unit[k] for k in ("tenant", "phone", "contract_type", "deposit", "monthly_rent",
                                     "maintenance", "start_date")}
        data["end_date"] = form["end_date"]
        data["memo"] = "\n".join(m for m in (unit["memo"], form["memo"]) if m)
        errors = validate_contract(db, unit_id, data)
        if form["record_return"] and form["return_amount"] <= 0:
            errors.append("반환할 보증금 금액을 입력하세요.")
        if not errors:
            insert_contract(db, unit_id, data)
            if form["record_return"]:
                db.execute(
                    """INSERT INTO transactions (tx_date, kind, category, unit_id, amount, memo)
                       VALUES (?, '지출', '보증금 반환', ?, ?, ?)""",
                    (data["end_date"], unit_id, form["return_amount"], f"{unit['tenant']} 퇴거"),
                )
            db.execute(
                """UPDATE units SET tenant = '', phone = '', deposit = 0,
                   start_date = '', end_date = '', memo = '' WHERE id = ?""",
                (unit_id,),
            )
            db.commit()
            flash(f"{unit['tenant']} 님의 계약을 종료하고 이력에 저장했습니다. 새 임차인은 [현재 계약 수정]에서 입력하세요.")
            return redirect(url_for("unit_detail", unit_id=unit_id))
        for e in errors:
            flash(e, "error")
    return render_template("contract_end.html", unit=unit, form=form)


@app.route("/units/<int:unit_id>/history/new", methods=["GET", "POST"])
@app.route("/contracts/<int:contract_id>/edit", methods=["GET", "POST"])
def contract_form(unit_id=None, contract_id=None):
    """지난 계약을 직접 입력하거나 수정합니다."""
    db = get_db()
    contract = None
    if contract_id is not None:
        contract = db.execute("SELECT * FROM contracts WHERE id = ?", (contract_id,)).fetchone()
        if contract is None:
            abort(404)
        unit_id = contract["unit_id"]
    unit = get_unit_or_404(db, unit_id)

    if request.method == "POST":
        data = contract_from_form(request.form)
        errors = validate_contract(db, unit_id, data, contract_id)
        if not errors:
            if contract is None:
                insert_contract(db, unit_id, data)
            else:
                db.execute(
                    f"UPDATE contracts SET {', '.join(f'{c} = ?' for c in data)} WHERE id = ?",
                    list(data.values()) + [contract_id],
                )
            db.commit()
            flash("지난 계약을 저장했습니다.")
            return redirect(url_for("unit_detail", unit_id=unit_id))
        for e in errors:
            flash(e, "error")
        contract = data

    return render_template(
        "contract_form.html", unit=unit, contract=contract, contract_id=contract_id,
        contract_types=CONTRACT_TYPES,
    )


@app.route("/contracts/<int:contract_id>/delete", methods=["POST"])
def contract_delete(contract_id):
    db = get_db()
    row = db.execute("SELECT unit_id FROM contracts WHERE id = ?", (contract_id,)).fetchone()
    if row is None:
        abort(404)
    db.execute("DELETE FROM contracts WHERE id = ?", (contract_id,))
    db.commit()
    flash("지난 계약을 삭제했습니다.")
    return redirect(url_for("unit_detail", unit_id=row["unit_id"]))


# ---------------------------------------------------------------- transactions
def tx_from_form(form):
    unit_id = form.get("unit_id") or None
    return {
        "tx_date": form.get("tx_date", "").strip(),
        "kind": form.get("kind", ""),
        "category": form.get("category", "").strip(),
        "unit_id": int(unit_id) if unit_id and unit_id.isdigit() else None,
        "amount": parse_int(form.get("amount"), -1),
        "memo": form.get("memo", "").strip(),
    }


def validate_tx(data):
    errors = []
    if not is_valid_date(data["tx_date"]):
        errors.append("날짜를 올바르게 입력하세요.")
    if data["kind"] not in ("수입", "지출"):
        errors.append("수입/지출을 선택하세요.")
    if not data["category"]:
        errors.append("분류를 선택하세요.")
    if data["amount"] <= 0:
        errors.append("금액은 0보다 커야 합니다.")
    return errors


@app.route("/transactions")
def transactions():
    db = get_db()
    year, month = parse_month(request.args.get("month"))
    start, end = month_range(year, month)
    unit_filter = request.args.get("unit_id", "")
    kind_filter = request.args.get("kind", "")

    sql = """SELECT t.*, u.name AS unit_name FROM transactions t
             LEFT JOIN units u ON u.id = t.unit_id
             WHERE t.tx_date BETWEEN ? AND ?"""
    params = [start, end]
    if unit_filter.isdigit():
        sql += " AND t.unit_id = ?"
        params.append(int(unit_filter))
    if kind_filter in ("수입", "지출"):
        sql += " AND t.kind = ?"
        params.append(kind_filter)
    sql += " ORDER BY t.tx_date DESC, t.id DESC"
    rows = db.execute(sql, params).fetchall()

    income = sum(r["amount"] for r in rows if r["kind"] == "수입")
    expense = sum(r["amount"] for r in rows if r["kind"] == "지출")
    py, pm = shift_month(year, month, -1)
    ny, nm = shift_month(year, month, 1)
    return render_template(
        "transactions.html",
        rows=rows, income=income, expense=expense,
        year=year, month=month,
        prev_month=f"{py:04d}-{pm:02d}", next_month=f"{ny:04d}-{nm:02d}",
        units=db.execute("SELECT id, name FROM units ORDER BY name").fetchall(),
        unit_filter=unit_filter, kind_filter=kind_filter,
    )


@app.route("/transactions/new", methods=["GET", "POST"])
@app.route("/transactions/<int:tx_id>/edit", methods=["GET", "POST"])
def tx_form(tx_id=None):
    db = get_db()
    tx = None
    if tx_id is not None:
        tx = db.execute("SELECT * FROM transactions WHERE id = ?", (tx_id,)).fetchone()
        if tx is None:
            abort(404)

    if request.method == "POST":
        data = tx_from_form(request.form)
        errors = validate_tx(data)
        if not errors:
            cols = list(data)
            if tx is None:
                db.execute(
                    f"INSERT INTO transactions ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                    [data[c] for c in cols],
                )
            else:
                db.execute(
                    f"UPDATE transactions SET {', '.join(f'{c} = ?' for c in cols)} WHERE id = ?",
                    [data[c] for c in cols] + [tx_id],
                )
            db.commit()
            flash("거래 내역을 저장했습니다.")
            if request.form.get("again"):
                return redirect(url_for("tx_form"))
            return redirect(url_for("transactions", month=data["tx_date"][:7]))
        for e in errors:
            flash(e, "error")
        tx = data
    elif tx is None:
        # 빠른 입력: ?unit_id=3&category=월세 처럼 미리 채우기
        unit_id = request.args.get("unit_id", "")
        tx = {
            "tx_date": date.today().isoformat(),
            "kind": request.args.get("kind", "수입"),
            "category": request.args.get("category", ""),
            "unit_id": int(unit_id) if unit_id.isdigit() else None,
            "amount": parse_int(request.args.get("amount"), 0) or "",
            "memo": "",
        }

    return render_template(
        "tx_form.html", tx=tx, tx_id=tx_id,
        units=db.execute("SELECT id, name, tenant FROM units ORDER BY name").fetchall(),
        income_categories=INCOME_CATEGORIES, expense_categories=EXPENSE_CATEGORIES,
    )


@app.route("/transactions/<int:tx_id>/delete", methods=["POST"])
def tx_delete(tx_id):
    db = get_db()
    row = db.execute("SELECT tx_date FROM transactions WHERE id = ?", (tx_id,)).fetchone()
    db.execute("DELETE FROM transactions WHERE id = ?", (tx_id,))
    db.commit()
    flash("거래 내역을 삭제했습니다.")
    return redirect(url_for("transactions", month=row["tx_date"][:7] if row else None))


# ---------------------------------------------------------------- yearly report
@app.route("/report")
def report():
    db = get_db()
    year = parse_int(request.args.get("year"), date.today().year)
    rows = db.execute(
        """SELECT substr(tx_date, 6, 2) AS m, kind, SUM(amount) AS total FROM transactions
           WHERE tx_date BETWEEN ? AND ? GROUP BY m, kind""",
        (f"{year}-01-01", f"{year}-12-31"),
    ).fetchall()
    months = {m: {"수입": 0, "지출": 0} for m in range(1, 13)}
    for r in rows:
        months[int(r["m"])][r["kind"]] = r["total"]
    by_unit = db.execute(
        """SELECT COALESCE(u.name, '(공통)') AS name,
                  SUM(CASE WHEN t.kind = '수입' THEN t.amount ELSE 0 END) AS income,
                  SUM(CASE WHEN t.kind = '지출' THEN t.amount ELSE 0 END) AS expense
           FROM transactions t LEFT JOIN units u ON u.id = t.unit_id
           WHERE t.tx_date BETWEEN ? AND ?
           GROUP BY t.unit_id ORDER BY name""",
        (f"{year}-01-01", f"{year}-12-31"),
    ).fetchall()
    return render_template("report.html", year=year, months=months, by_unit=by_unit)


init_db()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=True)
