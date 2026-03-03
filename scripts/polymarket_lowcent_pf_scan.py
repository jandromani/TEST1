#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import requests


@dataclass
class Sample:
    end_ts: float
    p_low: float
    low_won: bool
    slug: str


def _iso_to_ts(v: str) -> float | None:
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _last_price(session: requests.Session, token_id: str, start_ts: float, end_ts: float) -> float | None:
    r = session.get(
        "https://clob.polymarket.com/prices-history",
        params={"market": token_id, "startTs": int(start_ts), "endTs": int(end_ts)},
        timeout=20,
    )
    if r.status_code != 200:
        return None
    body = r.json()
    arr = body.get("history", []) if isinstance(body, dict) else []
    if not arr:
        return None
    best_t = -1
    best_p = None
    for row in arr:
        t = row.get("t")
        p = row.get("p")
        if t is None or p is None:
            continue
        if t > best_t:
            best_t = t
            best_p = float(p)
    return best_p


def _fetch_closed_markets(
    session: requests.Session,
    assets: set[str],
    max_markets: int,
    max_offset: int,
    page_size: int = 500,
) -> list[tuple[str, float, str, str, int]]:
    pat = re.compile(r"^(" + "|".join(a.lower() for a in sorted(assets)) + r")-updown-5m-")
    now = datetime.now(timezone.utc).timestamp()
    out: list[tuple[str, float, str, str, int]] = []

    for off in range(0, max_offset + 1, page_size):
        arr = session.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": page_size, "order": "id", "ascending": "false", "offset": off},
            timeout=30,
        ).json()
        if not arr:
            break
        for m in arr:
            slug = str(m.get("slug", "") or "")
            if not pat.search(slug):
                continue
            if not bool(m.get("closed")):
                continue
            end_ts = _iso_to_ts(str(m.get("endDate", "") or ""))
            if end_ts is None or end_ts >= now:
                continue
            try:
                toks = json.loads(m.get("clobTokenIds") or "[]")
                outp = json.loads(m.get("outcomePrices") or "[]")
            except Exception:
                continue
            if len(toks) != 2 or len(outp) != 2:
                continue
            winner_idx = 0 if float(outp[0]) > float(outp[1]) else 1
            out.append((slug, end_ts, str(toks[0]), str(toks[1]), int(winner_idx)))
            if len(out) >= max_markets:
                break
        if len(out) >= max_markets:
            break

    seen = set()
    dedup = []
    for row in out:
        if row[0] in seen:
            continue
        seen.add(row[0])
        dedup.append(row)
    dedup.sort(key=lambda x: x[1])
    return dedup


def _build_samples(
    session: requests.Session,
    markets: list[tuple[str, float, str, str, int]],
    window_from_sec: float,
    window_to_sec: float,
) -> list[Sample]:
    samples: list[Sample] = []
    for slug, end_ts, tok_up, tok_dn, winner_idx in markets:
        p_up = _last_price(session, tok_up, end_ts - window_from_sec, end_ts - window_to_sec)
        p_dn = _last_price(session, tok_dn, end_ts - window_from_sec, end_ts - window_to_sec)
        if p_up is None or p_dn is None:
            continue
        s = p_up + p_dn
        if s <= 0:
            continue
        p_up /= s
        p_dn /= s
        low_idx = 0 if p_up < p_dn else 1
        p_low = p_up if low_idx == 0 else p_dn
        samples.append(Sample(end_ts=end_ts, p_low=p_low, low_won=(low_idx == winner_idx), slug=slug))
    return samples


def _simulate_range(samples: list[Sample], lo: float, hi: float, stake: float, start_bank: float):
    sub = [x for x in samples if lo <= x.p_low < hi]
    n = len(sub)
    if n == 0:
        return None
    wins = sum(1 for x in sub if x.low_won)
    wr = wins / n
    gp = 0.0
    gl = 0.0
    bank = start_bank
    cur_ls = 0
    max_ls = 0
    for x in sub:
        if x.low_won:
            pnl = stake * ((1.0 / max(x.p_low, 1e-9)) - 1.0)
            gp += pnl
            cur_ls = 0
        else:
            pnl = -stake
            gl += stake
            cur_ls += 1
            max_ls = max(max_ls, cur_ls)
        bank += pnl
    pf = gp / gl if gl > 0 else math.inf
    return {
        "lo": lo,
        "hi": hi,
        "n": n,
        "wins": wins,
        "wr": wr,
        "pf": pf,
        "bank_final": bank,
        "pnl": bank - start_bank,
        "max_loss_streak": max_ls,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Polymarket 5m low-cent PF scanner (Gamma + CLOB only)")
    ap.add_argument("--assets", default="BTC,ETH", help="Comma-separated assets (default: BTC,ETH)")
    ap.add_argument("--max-markets", type=int, default=500, help="Max closed markets to scan")
    ap.add_argument("--max-offset", type=int, default=14000, help="Gamma pagination max offset")
    ap.add_argument("--window-from-sec", type=float, default=170.0, help="Entry window start before close")
    ap.add_argument("--window-to-sec", type=float, default=35.0, help="Entry window end before close")
    ap.add_argument("--grid-min", type=float, default=0.08)
    ap.add_argument("--grid-max", type=float, default=0.40)
    ap.add_argument("--grid-step", type=float, default=0.005)
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--stake", type=float, default=0.99)
    ap.add_argument("--start-bank", type=float, default=30.0)
    args = ap.parse_args()

    assets = {x.strip().upper() for x in args.assets.split(",") if x.strip()}
    session = requests.Session()

    markets = _fetch_closed_markets(
        session=session,
        assets=assets,
        max_markets=max(10, int(args.max_markets)),
        max_offset=max(500, int(args.max_offset)),
    )
    samples = _build_samples(
        session=session,
        markets=markets,
        window_from_sec=float(args.window_from_sec),
        window_to_sec=float(args.window_to_sec),
    )

    print(
        f"closed_markets={len(markets)} usable_samples={len(samples)} "
        f"window=[T-{args.window_from_sec:.0f}s,T-{args.window_to_sec:.0f}s] assets={','.join(sorted(assets))}"
    )

    g = []
    x = args.grid_min
    while x <= args.grid_max + 1e-9:
        g.append(round(x, 6))
        x += args.grid_step

    rows = []
    for i, lo in enumerate(g):
        for hi in g[i + 1 :]:
            r = _simulate_range(samples, lo, hi, stake=float(args.stake), start_bank=float(args.start_bank))
            if not r or r["n"] < int(args.min_samples) or not math.isfinite(r["pf"]):
                continue
            rows.append(r)

    rows.sort(key=lambda r: (r["pf"], r["n"]), reverse=True)
    for r in rows[: max(1, int(args.top))]:
        print(
            "range={lo:.3f}-{hi:.3f} n={n} wr={wr:.3f} pf={pf:.3f} "
            "pnl={pnl:+.2f} bank={bank_final:.2f} maxLS={max_loss_streak}".format(**r)
        )


if __name__ == "__main__":
    main()
