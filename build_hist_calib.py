#!/usr/bin/env python3
"""
build_hist_calib.py — build the hist_calib SQLite table locally, then upload to Northflank.

Usage:
    python3 build_hist_calib.py [--days 30] [--db /tmp/hist_calib.db]

After running, upload with:
    python3 build_hist_calib.py --upload

The bot reads from /data/clawdbot_metrics.db on Northflank.
This script builds the same hist_calib table independently and uploads it via the
Northflank CLI (nf exec) by base64-encoding the table rows as JSON.
"""

import argparse
import json
import math
import os
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone

# ── Constants (must match live_trader.py) ──────────────────────────────────────
SERIES_15M = {"BTC": 10192, "ETH": 10191, "SOL": 10423, "XRP": 10422}
HDRS       = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
BINANCE    = "https://api.binance.com/api/v3/klines"
GAMMA      = "https://gamma-api.polymarket.com/events"
CORR       = {"ETH": 0.82, "SOL": 0.80, "XRP": 0.78}

# ── NF identifiers (from ~/.northflank/config.json) ───────────────────────────
NF_PROJECT = "poly2"
NF_SERVICE = "clawdbot"
NF_IMPORT_DB = "/data/clawdbot_metrics.db"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_end_ts(ev):
    s = str(ev.get("endDate") or ev.get("endDateIso") or "")[:19]
    if not s:
        return None
    try:
        return int(datetime.fromisoformat(s.replace("Z", "")).replace(tzinfo=timezone.utc).timestamp())
    except Exception:
        return None


def fetch_binance_5m(asset: str, start_ts: int, end_ts: int) -> list:
    """Return sorted list of (ts_ms, close) for asset from Binance 5m klines."""
    symbol   = f"{asset}USDT"
    kd       = {}
    start_ms = (start_ts - 1800) * 1000
    end_ms   = end_ts * 1000
    req_cnt  = 0
    while start_ms < end_ms:
        url = f"{BINANCE}?symbol={symbol}&interval=5m&startTime={int(start_ms)}&limit=1000"
        try:
            req  = urllib.request.Request(url, headers=HDRS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                batch = json.loads(resp.read())
        except Exception as e:
            print(f"  [binance] {symbol} fetch error: {e}", file=sys.stderr)
            break
        if not batch:
            break
        for k in batch:
            kd[int(k[0])] = float(k[4])
        last_ts  = int(batch[-1][0])
        start_ms = last_ts + 300_000
        req_cnt += 1
        if len(batch) < 1000:
            break
        time.sleep(0.05)
    result = sorted(kd.items())
    print(f"  {asset}: {len(result)} 5m candles ({req_cnt} requests)")
    return result


def fetch_gamma_rounds(asset: str, sid: int, cutoff_ts: int, now_ts: int) -> list:
    """Return list of (asset, round_start_ts, side, won) for resolved rounds in window."""
    # Probe first event to compute start_offset
    try:
        req = urllib.request.Request(f"{GAMMA}?series_id={sid}&closed=true&limit=1&offset=0", headers=HDRS)
        with urllib.request.urlopen(req, timeout=10) as r:
            first_page = json.loads(r.read())
    except Exception:
        first_page = []

    series_start_ts = _parse_end_ts(first_page[0]) if first_page else None
    if series_start_ts:
        start_offset = max(0, int((cutoff_ts - series_start_ts) / 900) - 400)
    else:
        start_offset = 0
    print(f"  {asset}: series_start={series_start_ts}, start_offset={start_offset}")

    rounds = []
    offset = start_offset
    while True:
        url = f"{GAMMA}?series_id={sid}&closed=true&limit=200&offset={offset}"
        try:
            req = urllib.request.Request(url, headers=HDRS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                events = json.loads(resp.read())
        except Exception as e:
            print(f"  [gamma] {asset} fetch error: {e}", file=sys.stderr)
            break
        if not events:
            break
        for ev in events:
            end_ts_ev = _parse_end_ts(ev)
            if end_ts_ev is None or end_ts_ev < cutoff_ts:
                continue
            if end_ts_ev > now_ts:
                continue
            round_start_ts = end_ts_ev - 900
            for mkt in ev.get("markets", []):
                try:
                    op = mkt.get("outcomePrices", [])
                    oc = mkt.get("outcomes", [])
                    if isinstance(op, str):
                        op = json.loads(op)
                    if isinstance(oc, str):
                        oc = json.loads(oc)
                    p0, p1 = float(op[0]), float(op[1])
                except Exception:
                    continue
                if p0 > 0.90:
                    winner = str(oc[0])
                elif p1 > 0.90:
                    winner = str(oc[1])
                else:
                    continue
                for side in ("Up", "Down"):
                    won = 1 if winner == side else 0
                    rounds.append((asset, round_start_ts, side, won))
        offset += 200
        if len(events) < 200:
            break

    print(f"  {asset}: {len(rounds)//2} resolved rounds")
    return rounds


def _get_closes_before(klines: list, before_ts_ms: int, n: int) -> list:
    """Binary search: return last n closes with ts_ms <= before_ts_ms."""
    lo, hi = 0, len(klines)
    while lo < hi:
        mid = (lo + hi) // 2
        if klines[mid][0] <= before_ts_ms:
            lo = mid + 1
        else:
            hi = mid
    return [c for _, c in klines[max(0, lo - n):lo]]


def compute_llr_and_prob(asset: str, round_start_ts: int, side: str,
                          asset_klines: dict) -> tuple:
    """Returns (llr, true_prob) or (None, None) if insufficient data."""
    round_start_ms = round_start_ts * 1000
    klines = asset_klines.get(asset, [])

    closes = _get_closes_before(klines, round_start_ms, 14)
    if len(closes) < 5:
        return None, None

    rets = [(closes[i] - closes[i-1]) / closes[i-1]
            for i in range(1, len(closes)) if closes[i-1] > 0]
    if len(rets) < 3:
        return None, None

    sigma_5m  = max((sum(r*r for r in rets) / len(rets)) ** 0.5, 1e-8)
    sigma_15m = sigma_5m * (3 ** 0.5)

    n_bars = min(12, len(closes) - 1)
    c0, c1 = closes[-(n_bars + 1)], closes[-1]
    if c0 <= 0 or sigma_15m <= 0:
        return None, None

    ret = (c1 - c0) / c0
    llr = ret / sigma_15m * 0.8

    if asset != "BTC":
        btc_klines = asset_klines.get("BTC", [])
        btc_closes = _get_closes_before(btc_klines, round_start_ms, 14)
        if len(btc_closes) >= 5:
            btc_rets = [(btc_closes[i] - btc_closes[i-1]) / btc_closes[i-1]
                        for i in range(1, len(btc_closes)) if btc_closes[i-1] > 0]
            if btc_rets:
                btc_sigma_5m = max((sum(r*r for r in btc_rets) / len(btc_rets)) ** 0.5, 1e-8)
                btc_sigma15  = btc_sigma_5m * (3 ** 0.5)
                btc_c0_30 = _get_closes_before(btc_klines, round_start_ms - 30*60*1000, 1)
                btc_c1    = btc_closes[-1]
                if btc_c0_30 and btc_c0_30[0] > 0 and btc_sigma15 > 0:
                    btc_ret = (btc_c1 - btc_c0_30[0]) / btc_c0_30[0]
                    corr    = CORR.get(asset, 0.77)
                    llr    += btc_ret / btc_sigma15 * corr * 0.8

    prob_up  = 1.0 / (1.0 + math.exp(-llr))
    true_prob = prob_up if side == "Up" else (1.0 - prob_up)
    true_prob = max(0.05, min(0.95, true_prob))
    return round(llr, 6), round(true_prob, 6)


# ── Main build ─────────────────────────────────────────────────────────────────

def build(db_path: str, days: int):
    now_ts   = int(time.time())
    cutoff_ts = now_ts - days * 86400

    print(f"\n=== build_hist_calib: {days} days, cutoff {datetime.utcfromtimestamp(cutoff_ts).isoformat()}Z ===\n")

    # Init DB
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hist_calib (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            asset      TEXT    NOT NULL,
            round_ts   INTEGER NOT NULL,
            side       TEXT    NOT NULL,
            won        INTEGER NOT NULL,
            llr        REAL,
            true_prob  REAL,
            created_at INTEGER DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
            UNIQUE(asset, round_ts, side)
        )
    """)
    conn.commit()

    # Fetch Binance klines
    print("[1/3] Fetching Binance 5m klines...")
    asset_klines = {}
    for asset in SERIES_15M:
        asset_klines[asset] = fetch_binance_5m(asset, cutoff_ts, now_ts)

    # Fetch Gamma rounds
    print("\n[2/3] Fetching Gamma resolved rounds...")
    all_rounds = []
    for asset, sid in SERIES_15M.items():
        all_rounds += fetch_gamma_rounds(asset, sid, cutoff_ts, now_ts)
    print(f"  Total: {len(all_rounds)//2} rounds ({len(all_rounds)} side-records)")

    # Compute LLR + store
    print("\n[3/3] Computing signals and storing...")
    new_count = skip_count = 0
    for asset, round_start_ts, side, won in all_rounds:
        cur = conn.execute(
            "SELECT id FROM hist_calib WHERE asset=? AND round_ts=? AND side=?",
            (asset, round_start_ts, side)
        )
        if cur.fetchone():
            skip_count += 1
            continue

        llr, true_prob = compute_llr_and_prob(asset, round_start_ts, side, asset_klines)
        if llr is None:
            skip_count += 1
            continue

        conn.execute(
            "INSERT OR IGNORE INTO hist_calib (asset, round_ts, side, won, llr, true_prob) "
            "VALUES (?,?,?,?,?,?)",
            (asset, round_start_ts, side, won, llr, true_prob)
        )
        new_count += 1

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM hist_calib").fetchone()[0]
    conn.close()

    print(f"\nDone: {new_count} new rows, {skip_count} skipped. Total in DB: {total}")
    print(f"DB file: {db_path}  ({os.path.getsize(db_path)//1024} KB)\n")
    return db_path


# ── Upload to Northflank ───────────────────────────────────────────────────────

def upload(db_path: str):
    """
    Export hist_calib rows as JSON and import them into Northflank via the
    NF REST API (PATCH /v1/projects/{p}/services/{s}/env — uses custom import endpoint
    in the bot itself at /api/import_hist_calib).

    Simpler approach: use `nf exec` to run a python3 one-liner that reads the
    JSON piped via stdin and writes to the remote SQLite.
    """
    import base64
    import subprocess

    # Export rows to JSON
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT asset, round_ts, side, won, llr, true_prob FROM hist_calib"
    ).fetchall()
    conn.close()

    data = [{"asset": r[0], "round_ts": r[1], "side": r[2],
             "won": r[3], "llr": r[4], "true_prob": r[5]} for r in rows]
    payload = json.dumps(data)
    b64 = base64.b64encode(payload.encode()).decode()

    print(f"Uploading {len(rows)} rows ({len(b64)//1024} KB base64) to Northflank...")

    # Python one-liner to run inside the container
    remote_py = (
        "import base64,json,sqlite3;"
        "d=json.loads(base64.b64decode(open('/tmp/_hc_payload.b64').read()));"
        "c=sqlite3.connect('/data/clawdbot_metrics.db',timeout=30);"
        "c.execute('CREATE TABLE IF NOT EXISTS hist_calib ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,asset TEXT NOT NULL,round_ts INTEGER NOT NULL,"
        "side TEXT NOT NULL,won INTEGER NOT NULL,llr REAL,true_prob REAL,"
        "created_at INTEGER DEFAULT (CAST(strftime(chr(37)+chr(115),chr(110)+chr(111)+chr(119)) AS INTEGER)),"
        "UNIQUE(asset,round_ts,side))');"
        "[c.execute('INSERT OR IGNORE INTO hist_calib (asset,round_ts,side,won,llr,true_prob) VALUES(?,?,?,?,?,?)',"
        "(r['asset'],r['round_ts'],r['side'],r['won'],r['llr'],r['true_prob'])) for r in d];"
        "c.commit();"
        "print('imported',len(d),'rows')"
    )

    # Write b64 to /tmp on the container then exec
    cmd_write = ["nf", "exec", "--project", NF_PROJECT, "--service", NF_SERVICE,
                 "--", "sh", "-c", f"echo '{b64}' > /tmp/_hc_payload.b64"]
    cmd_exec  = ["nf", "exec", "--project", NF_PROJECT, "--service", NF_SERVICE,
                 "--", "python3", "-c", remote_py]

    print("Step 1: writing payload to container /tmp ...")
    r1 = subprocess.run(cmd_write, capture_output=True, text=True)
    if r1.returncode != 0:
        print(f"ERROR: {r1.stderr}")
        print("\nManual upload alternative:")
        print(f"  1. Copy {db_path} to a reachable URL (e.g. paste.rs, S3, etc.)")
        print(f"  2. Set env HIST_CALIB_SEED_URL=<url> on the service")
        print(f"  3. The bot will download and merge on next startup")
        return

    print("Step 2: running import inside container ...")
    r2 = subprocess.run(cmd_exec, capture_output=True, text=True)
    print(r2.stdout)
    if r2.returncode != 0:
        print(f"ERROR: {r2.stderr}")
    else:
        print("Upload complete.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and/or upload hist_calib DB")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of history to fetch (default: 30)")
    parser.add_argument("--db",   default="/tmp/hist_calib_local.db",
                        help="Local DB file path (default: /tmp/hist_calib_local.db)")
    parser.add_argument("--upload", action="store_true",
                        help="Upload existing DB to Northflank (skip build)")
    parser.add_argument("--build-and-upload", action="store_true",
                        help="Build then upload in one step")
    args = parser.parse_args()

    if args.upload:
        if not os.path.exists(args.db):
            print(f"DB not found: {args.db}. Run without --upload first.")
            sys.exit(1)
        upload(args.db)
    elif args.build_and_upload:
        build(args.db, args.days)
        upload(args.db)
    else:
        build(args.db, args.days)
        print("To upload to Northflank run:")
        print(f"  python3 build_hist_calib.py --upload --db {args.db}")
