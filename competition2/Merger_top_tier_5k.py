"""
RITC 2026 — Merger Arbitrage (TOP-TIER 3M+ Mode, 5k order limit compliant)
=========================================================================

This is an aggressive, burst-execution merger-arb bot tuned for big P&L spikes:
- Fast news polling + event-driven shock mode
- Burst MARKET entry right after shock news (build 25k instantly with 5k clips)
- Optional LIMIT follow-up (auto-detect; auto-fallback to MARKET if unsupported)
- Rule-based shock/severity overrides (so truly big headlines always size big)
- Asymmetric sizing: bigger on negative shocks
- p_model decay to anchors (reduces post-spike donation)
- Persistent seen news IDs (restart-safe)
- Console controls: show, cancel, reload, quit

Training CSV:
- Supports headers like: Headline,Category,Direction,Severity
- Also supports: text/category/direction/severity, Title/News, etc.

Usage:
  Train (one or multiple files):
    python Merger_top_tier_5k.py train merger_arbitrage_news.csv
    python Merger_top_tier_5k.py train file1.csv file2.csv

  Run live:
    python Merger_top_tier_5k.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
import threading
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import requests
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =============================================================================
# CONFIG (AGGRESSIVE + 5k ORDER LIMIT)
# =============================================================================
API_KEY = "OR96FJU1"
HOST = "localhost"
PORT = 9999
BASE_URL = f"http://{HOST}:{PORT}/v1"
TIMEOUT = 1.0

# Risk limits (raise only if allowed by your case)
GROSS_LIMIT = 900_000
NET_LIMIT   = 600_000

# Execution constraints
MAX_ORDER_SIZE = 5_000  # RIT per-order cap
MIN_SECONDS_BETWEEN_TRADES_PER_TICKER = 0.01

# Burst entry (key for catching repricing)
BURST_CLIPS = 5          # 5 clips * 5k = 25k instantly
BURST_SLEEP_S = 0.01

# Loop speeds
TRADING_LOOP_S = 0.06
NEWS_POLL_S    = 0.05

# Strategy thresholds
TAU_ENTER = 0.015
TAU_EXIT  = 0.005

# Costs (used mainly outside shock window)
COMMISSION = 0.02
COST_CUSHION = 1.1

# Position sizing caps (big money comes from big correct positions)
PER_DEAL_CAP_TARGET = 110_000  # target shares on target leg
MAX_ABS_DP = 0.35              # allow larger belief jumps

# Shock timing
SHOCK_WINDOW_S = 2.0
SHOCK_HOLD_S   = 8.0

# Scale-out to protect spikes
SCALE_OUT_START_S = 3.0
SCALE_OUT_STEP_S  = 1.2
SCALE_OUT_FRACTIONS = [0.25, 0.25, 0.25, 0.25]

# Flip freeze (lighter than "safe" version)
FLIP_WINDOW_S = 15.0
MAX_FLIPS_IN_WINDOW = 5
FREEZE_S = 20.0

MODEL_PATH = "news_classifier.joblib"
SEEN_IDS_PATH = "seen_news_ids.json"


# =============================================================================
# DEAL DEFINITIONS (EDIT IF YOUR TICKERS DIFFER)
# =============================================================================
@dataclass(frozen=True)
class Deal:
    name: str
    target: str
    acquirer: str
    structure: str   # "CASH", "STOCK", "MIXED"
    cash: float
    ratio: float


DEALS: Dict[str, Deal] = {
    "D1": Deal("Pharma cash",     "TGX", "PHR", "CASH",  50.0, 0.00),
    "D2": Deal("Cloud stock",     "BYL", "CLD", "STOCK",  0.0, 0.75),
    "D3": Deal("Energy mixed",    "GGD", "PNR", "MIXED", 33.0, 0.20),
    "D4": Deal("Bank cash",       "FSR", "ATB", "CASH",  40.0, 0.00),
    "D5": Deal("Renewable stock", "SPK", "EEC", "STOCK",  0.0, 1.20),
}

# Starting belief anchors
P0_ANCHOR = {"D1": 0.70, "D2": 0.55, "D3": 0.50, "D4": 0.38, "D5": 0.45}

# dp multipliers
CAT_MULT  = {"REG": 1.25, "FIN": 1.00, "SHR": 0.90, "ALT": 1.40, "PRC": 0.70}
DEAL_MULT = {"D1": 1.00, "D2": 1.05, "D3": 1.10, "D4": 1.30, "D5": 1.15}

# base dp by direction/severity
BASE_IMPACT = {
    ("+", "S"): 0.03, ("+", "M"): 0.08, ("+", "L"): 0.18,
    ("-", "S"): -0.04, ("-", "M"): -0.10, ("-", "L"): -0.22,
}

# Used only to infer V at t=0
START_PRICES = {
    "TGX": 43.70, "PHR": 47.50,
    "BYL": 43.50, "CLD": 79.30,
    "GGD": 31.50, "PNR": 59.80,
    "FSR": 30.50, "ATB": 62.20,
    "SPK": 52.80, "EEC": 48.00,
}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# =============================================================================
# RIT REST WRAPPER
# =============================================================================
class RIT:
    """
    Common RIT behavior:
      - POST /orders expects fields as query params (NOT JSON)
      - LIMIT may or may not be supported in your build; we auto-probe once.
    """
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"X-API-Key": API_KEY})

    def _get(self, path: str, params: Optional[dict] = None):
        r = self.s.get(BASE_URL + path, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, params: Optional[dict] = None):
        r = self.s.post(BASE_URL + path, params=params, timeout=TIMEOUT)
        if r.status_code == 400:
            raise requests.HTTPError(
                f"400 Bad Request for {BASE_URL+path}\nSent params={params}\nBody={r.text}",
                response=r,
            )
        r.raise_for_status()
        if not r.text.strip():
            return None
        try:
            return r.json()
        except ValueError:
            return r.text

    def get_news(self) -> List[dict]:
        return self._get("/news")

    def get_securities(self, ticker: Optional[str] = None) -> List[dict]:
        params = {"ticker": ticker} if ticker else None
        return self._get("/securities", params=params)

    def get_positions(self) -> Dict[str, int]:
        rows = self.get_securities()
        return {row["ticker"]: int(row.get("position", 0)) for row in rows}

    def get_best_bid_ask(self, ticker: str) -> Tuple[Optional[float], Optional[float]]:
        rows = self.get_securities(ticker=ticker)
        if rows:
            sec = rows[0]
            bid = sec.get("bid", None)
            ask = sec.get("ask", None)
            if bid is not None and ask is not None:
                return float(bid), float(ask)

        book = self._get("/securities/book", params={"ticker": ticker, "limit": 1})
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else None
        best_ask = float(asks[0]["price"]) if asks else None
        return best_bid, best_ask

    def get_mid(self, ticker: str) -> Optional[float]:
        b, a = self.get_best_bid_ask(ticker)
        if b is None or a is None:
            return None
        return 0.5 * (b + a)

    def cancel_all_orders(self):
        return self._post("/commands/cancel", params={"all": 1})

    def post_order_market(self, ticker: str, qty: int, action: str):
        qty = int(qty)
        if qty <= 0:
            return None
        params = {"ticker": ticker, "type": "MARKET", "quantity": qty, "action": action}
        return self._post("/orders", params=params)

    def post_order_limit(self, ticker: str, qty: int, action: str, price: float):
        qty = int(qty)
        if qty <= 0:
            return None
        params = {"ticker": ticker, "type": "LIMIT", "quantity": qty, "action": action, "price": float(price)}
        return self._post("/orders", params=params)


# =============================================================================
# NEWS CLASSIFIER (TFIDF + LR + shock overrides)
# =============================================================================
class NewsClassifier:
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=9000)
        self.clf_cat = LogisticRegression(max_iter=400)
        self.clf_dir = LogisticRegression(max_iter=400)
        self.clf_sev = LogisticRegression(max_iter=400)

    @staticmethod
    def infer_deal_from_text(text: str) -> Optional[str]:
        t = text.upper()
        for dkey, deal in DEALS.items():
            if deal.target in t or deal.acquirer in t:
                return dkey
        return None

    @staticmethod
    def keyword_rules(text: str) -> Dict[str, Optional[str]]:
        s = text.lower()
        # Category rules
        if any(w in s for w in ["antitrust", "regulator", "regulators", "ftc", "doj", "competition bureau", "injunction", "lawsuit", "litigation"]):
            cat = "REG"
        elif any(w in s for w in ["financing", "credit", "lenders", "repricing", "refinancing", "funding", "debt"]):
            cat = "FIN"
        elif any(w in s for w in ["shareholder", "proxy", "vote", "tender", "board", "activist", "iss", "glass lewis"]):
            cat = "SHR"
        elif any(w in s for w in ["topping bid", "competing bid", "rival", "alternate bidder", "go-shop", "strategic alternatives"]):
            cat = "ALT"
        elif any(w in s for w in ["timeline", "delay", "extends", "extension", "closing date", "deadline", "procedural"]):
            cat = "PRC"
        else:
            cat = None

        # Direction rules
        pos_words = ["approved", "cleared", "green light", "constructive", "secured", "committed", "support", "favorable", "accepts"]
        neg_words = ["blocked", "sues", "lawsuit", "challenge", "widen", "concern", "uncertain", "withdraw", "terminate", "fails", "downgrade"]

        pos_hit = any(w in s for w in pos_words)
        neg_hit = any(w in s for w in neg_words)

        direction = None
        if pos_hit and not neg_hit:
            direction = "+"
        elif neg_hit and not pos_hit:
            direction = "-"

        return {"category": cat, "direction": direction}

    @staticmethod
    def shock_phrase(text: str) -> bool:
        s = text.lower()
        strong = [
            # REG mega negative
            "ftc sues", "doj sues", "injunction", "blocked", "lawsuit filed",
            "sues to block", "regulator blocks", "antitrust suit",
            # REG mega positive
            "cleared", "approved", "green light", "unconditional approval",
            # ALT mega
            "topping bid", "competing bid", "rival bid", "higher offer",
            "all-cash offer", "go-shop", "strategic alternatives",
            # FIN mega
            "financing secured", "committed financing", "funding secured",
            # deal breaks
            "terminates", "withdraws", "deal off", "walks away",
        ]
        return any(p in s for p in strong)

    def fit(self, texts: List[str], y_cat: List[str], y_dir: List[str], y_sev: List[str]):
        X = self.vec.fit_transform(texts)
        self.clf_cat.fit(X, y_cat)
        self.clf_dir.fit(X, y_dir)
        self.clf_sev.fit(X, y_sev)

    def predict(self, text: str) -> Tuple[Optional[str], str, str, str, float, bool]:
        deal = self.infer_deal_from_text(text)
        rules = self.keyword_rules(text)
        shock = self.shock_phrase(text)

        X = self.vec.transform([text])

        # Category
        if rules["category"] is not None:
            cat, cat_conf = rules["category"], 0.95
        else:
            p = self.clf_cat.predict_proba(X)[0]
            i = int(p.argmax())
            cat, cat_conf = self.clf_cat.classes_[i], float(p[i])

        # Direction
        if rules["direction"] is not None:
            direction, dir_conf = rules["direction"], 0.92
        else:
            p = self.clf_dir.predict_proba(X)[0]
            i = int(p.argmax())
            direction, dir_conf = self.clf_dir.classes_[i], float(p[i])

        # Severity
        p = self.clf_sev.predict_proba(X)[0]
        i = int(p.argmax())
        severity, sev_conf = self.clf_sev.classes_[i], float(p[i])

        # Hard severity override on shock phrases
        if shock:
            severity, sev_conf = "L", 0.99

        conf = min(cat_conf, dir_conf, sev_conf)
        return deal, str(cat), str(direction), str(severity), float(conf), bool(shock)

    def save(self, path: str = MODEL_PATH):
        joblib.dump({"vec": self.vec, "cat": self.clf_cat, "dir": self.clf_dir, "sev": self.clf_sev}, path)

    @staticmethod
    def load(path: str = MODEL_PATH) -> "NewsClassifier":
        obj = joblib.load(path)
        nc = NewsClassifier()
        nc.vec = obj["vec"]
        nc.clf_cat = obj["cat"]
        nc.clf_dir = obj["dir"]
        nc.clf_sev = obj["sev"]
        return nc


# =============================================================================
# ENGINE (shock mode + burst entry + decay + scale-out)
# =============================================================================
class MergerArbEngine:
    def __init__(self, rit: RIT):
        self.rit = rit
        self.lock = threading.Lock()
        self.running = True

        self.p_model = dict(P0_ANCHOR)
        self.V: Dict[str, float] = {}

        # Event state
        self.last_event_ts: Dict[str, float] = {d: -1e9 for d in DEALS}
        self.last_event_cat: Dict[str, str] = {d: "FIN" for d in DEALS}
        self.last_event_sev: Dict[str, str] = {d: "S" for d in DEALS}
        self.last_event_dir: Dict[str, str] = {d: "+" for d in DEALS}
        self.last_event_shock: Dict[str, bool] = {d: False for d in DEALS}

        # Scale-out plan
        self.scale_plan_until: Dict[str, float] = {d: -1e9 for d in DEALS}
        self.scale_steps_done: Dict[str, int] = {d: 0 for d in DEALS}
        self.scale_entry_target: Dict[str, int] = {d: 0 for d in DEALS}

        # Throttle + freeze
        self.last_trade_time_by_ticker: Dict[str, float] = {}
        self.flip_times: Dict[str, List[float]] = {d: [] for d in DEALS}
        self.frozen_until: Dict[str, float] = {d: 0.0 for d in DEALS}

        # Infer standalone V at t=0
        for dkey, deal in DEALS.items():
            p0 = P0_ANCHOR[dkey]
            P0_t = START_PRICES[deal.target]
            if deal.structure == "CASH":
                K0 = deal.cash
            elif deal.structure == "STOCK":
                K0 = deal.ratio * START_PRICES[deal.acquirer]
            else:
                K0 = deal.cash + deal.ratio * START_PRICES[deal.acquirer]
            self.V[dkey] = (P0_t - p0 * K0) / (1.0 - p0)

        # LIMIT support probe (best-effort)
        self.limit_supported = True
        self._limit_probe_done = False

    def stop(self) -> None:
        self.running = False

    # ---- deal value
    def current_K(self, dkey: str) -> Optional[float]:
        deal = DEALS[dkey]
        if deal.structure == "CASH":
            return deal.cash
        acq_mid = self.rit.get_mid(deal.acquirer)
        if acq_mid is None:
            return None
        if deal.structure == "STOCK":
            return deal.ratio * acq_mid
        return deal.cash + deal.ratio * acq_mid

    def p_mkt_touch(self, dkey: str, side: str) -> Optional[float]:
        deal = DEALS[dkey]
        V = self.V[dkey]
        K = self.current_K(dkey)
        if K is None:
            return None
        bid, ask = self.rit.get_best_bid_ask(deal.target)
        if bid is None or ask is None:
            return None
        P = ask if side == "BUY" else bid
        denom = K - V
        if abs(denom) < 1e-9:
            return None
        return clamp01((P - V) / denom)

    def intrinsic_target(self, dkey: str, p: float) -> Optional[float]:
        V = self.V[dkey]
        K = self.current_K(dkey)
        if K is None:
            return None
        return p * K + (1 - p) * V

    # ---- limits
    @staticmethod
    def gross_net(pos: Dict[str, int]) -> Tuple[int, int]:
        gross = sum(abs(v) for v in pos.values())
        net = abs(sum(pos.values()))
        return gross, net

    def within_limits_after(self, pos: Dict[str, int], ticker: str, delta: int) -> bool:
        sim = dict(pos)
        sim[ticker] = sim.get(ticker, 0) + delta
        gross, net = self.gross_net(sim)
        return gross <= GROSS_LIMIT and net <= NET_LIMIT

    # ---- throttle
    def ticker_throttle_ok(self, ticker: str) -> bool:
        now = time.time()
        last = self.last_trade_time_by_ticker.get(ticker, 0.0)
        if now - last < MIN_SECONDS_BETWEEN_TRADES_PER_TICKER:
            return False
        self.last_trade_time_by_ticker[ticker] = now
        return True

    # ---- hedge multiplier (more aggressive near high p)
    def hedge_mult(self, dkey: str) -> float:
        with self.lock:
            p = clamp01(self.p_model[dkey])
        return 0.4 + 0.9 * p

    def desired_hedged_positions(self, dkey: str, target_shares: int) -> Dict[str, int]:
        deal = DEALS[dkey]
        out = {deal.target: int(target_shares)}
        if deal.structure == "CASH":
            return out
        hm = self.hedge_mult(dkey)
        out[deal.acquirer] = -int(round(hm * deal.ratio * target_shares))
        return out

    # ---- cost filter (used outside shock)
    def clears_costs(self, dkey: str, target_shares: int) -> bool:
        deal = DEALS[dkey]
        bid, ask = self.rit.get_best_bid_ask(deal.target)
        if bid is None or ask is None:
            return False
        spread = max(0.0, ask - bid)
        half_spread = 0.5 * spread
        with self.lock:
            pmod = self.p_model[dkey]
        pintr = self.intrinsic_target(dkey, pmod)
        if pintr is None:
            return False
        edge_dollars = (pintr - ask) if target_shares > 0 else (bid - pintr)
        required = COST_CUSHION * (half_spread + COMMISSION)
        return edge_dollars > required

    # ---- flip freeze
    def _prune_flip_times(self, dkey: str, now: float) -> None:
        self.flip_times[dkey] = [t for t in self.flip_times[dkey] if now - t <= FLIP_WINDOW_S]

    def _record_flip_and_maybe_freeze(self, dkey: str, now: float) -> None:
        self.flip_times[dkey].append(now)
        self._prune_flip_times(dkey, now)
        if len(self.flip_times[dkey]) > MAX_FLIPS_IN_WINDOW:
            self.frozen_until[dkey] = max(self.frozen_until[dkey], now + FREEZE_S)
            print(f"[FREEZE] {dkey} frozen {FREEZE_S:.0f}s (flips>{MAX_FLIPS_IN_WINDOW} in {FLIP_WINDOW_S:.0f}s).")

    # ---- LIMIT probe
    def _ensure_limit_probe(self) -> None:
        if self._limit_probe_done:
            return
        self._limit_probe_done = True
        try:
            any_ticker = next(iter(START_PRICES.keys()))
            b, a = self.rit.get_best_bid_ask(any_ticker)
            if b is None or a is None:
                self.limit_supported = False
                return
            # place a tiny far-away limit to avoid fill
            self.rit.post_order_limit(any_ticker, 1, "BUY", price=max(0.01, b * 0.5))
            try:
                self.rit.cancel_all_orders()
            except Exception:
                pass
            self.limit_supported = True
            print("[OK] LIMIT appears supported.")
        except Exception:
            self.limit_supported = False
            print("[WARN] LIMIT not supported (probe failed). Using MARKET only.")

    # ---- aggressive placement: BURST MARKET, then LIMIT-at-touch (if available)
    def place_aggressive(self, ticker: str, diff: int) -> None:
        self._ensure_limit_probe()

        action = "BUY" if diff > 0 else "SELL"
        remaining = abs(diff)

        # BURST MARKET entry (key for catching repricing)
        clips = min(BURST_CLIPS, math.ceil(remaining / MAX_ORDER_SIZE))
        for _ in range(clips):
            clip = min(MAX_ORDER_SIZE, remaining)
            self.rit.post_order_market(ticker, clip, action)
            remaining -= clip
            if remaining <= 0:
                return
            time.sleep(BURST_SLEEP_S)

        # Follow-up: LIMIT at touch if supported, else MARKET clips
        if not self.limit_supported:
            while remaining > 0:
                clip = min(MAX_ORDER_SIZE, remaining)
                self.rit.post_order_market(ticker, clip, action)
                remaining -= clip
            return

        b, a = self.rit.get_best_bid_ask(ticker)
        touch = a if action == "BUY" else b
        if touch is None:
            while remaining > 0:
                clip = min(MAX_ORDER_SIZE, remaining)
                self.rit.post_order_market(ticker, clip, action)
                remaining -= clip
            return

        while remaining > 0:
            clip = min(MAX_ORDER_SIZE, remaining)
            try:
                self.rit.post_order_limit(ticker, clip, action, price=float(touch))
            except requests.HTTPError:
                self.limit_supported = False
                while remaining > 0:
                    c2 = min(MAX_ORDER_SIZE, remaining)
                    self.rit.post_order_market(ticker, c2, action)
                    remaining -= c2
                return
            remaining -= clip

    def trade_toward(self, desired: Dict[str, int]) -> None:
        pos = self.rit.get_positions()

        for ticker, want in desired.items():
            cur = pos.get(ticker, 0)
            diff = want - cur
            if diff == 0:
                continue

            # Throttle to avoid useless spam (burst happens inside place_aggressive)
            if not self.ticker_throttle_ok(ticker):
                continue

            step = int(math.copysign(min(MAX_ORDER_SIZE, abs(diff)), diff))
            while step != 0 and not self.within_limits_after(pos, ticker, step):
                step = int(step / 2)
            if step == 0:
                continue

            self.place_aggressive(ticker, step)
            pos[ticker] = pos.get(ticker, 0) + step

    # ---- sizing
    def shock_size(self, edge: float, category: str, severity: str, age_s: float, shock_flag: bool) -> int:
        base = 8_000

        # category scaling
        if category in {"REG", "ALT"}:
            base *= 3
        elif category == "FIN":
            base *= 2

        # severity scaling
        if severity == "L":
            base *= 3
        elif severity == "M":
            base *= 2

        # freshness
        if age_s <= 0.8:
            base = int(base * 2.5)
        elif age_s <= SHOCK_WINDOW_S:
            base = int(base * 1.7)

        # shock phrase
        if shock_flag:
            base = int(base * 1.4)

        # edge scaling
        mag = abs(edge)
        if mag >= 0.12:
            base = int(base * 2.0)
        elif mag >= 0.07:
            base = int(base * 1.5)
        elif mag >= 0.03:
            base = int(base * 1.0)
        else:
            base = int(base * 0.6)

        base = min(int(base), PER_DEAL_CAP_TARGET)

        # Asymmetry: larger on negative shocks (often bigger dislocations)
        if edge < 0:
            base = int(base * 1.3)

        return base if edge > 0 else -base

    # ---- scale-out
    def maybe_scale_out(self, dkey: str, now: float) -> Optional[int]:
        entry_ts = self.last_event_ts[dkey]
        if now - entry_ts < SCALE_OUT_START_S:
            return None
        if now > self.scale_plan_until[dkey]:
            return None

        steps = self.scale_steps_done[dkey]
        if steps >= len(SCALE_OUT_FRACTIONS):
            return None

        next_step_time = entry_ts + SCALE_OUT_START_S + steps * SCALE_OUT_STEP_S
        if now < next_step_time:
            return None

        entry_target = self.scale_entry_target[dkey]
        remaining_frac = 1.0 - sum(SCALE_OUT_FRACTIONS[: steps + 1])
        new_target = int(round(entry_target * remaining_frac))

        self.scale_steps_done[dkey] += 1
        return new_target

    # ---- news update
    def apply_news_update(self, dkey: str, direction: str, severity: str, category: str, conf: float, shock_flag: bool, tag: str) -> None:
        base = BASE_IMPACT[(direction, severity)]
        dp = base * CAT_MULT.get(category, 1.0) * DEAL_MULT.get(dkey, 1.0)
        dp = max(-MAX_ABS_DP, min(MAX_ABS_DP, dp))

        now = time.time()
        with self.lock:
            self.p_model[dkey] = clamp01(self.p_model[dkey] + dp)
            newp = self.p_model[dkey]
            self.last_event_ts[dkey] = now
            self.last_event_cat[dkey] = category
            self.last_event_sev[dkey] = severity
            self.last_event_dir[dkey] = direction
            self.last_event_shock[dkey] = shock_flag

        print(f"[NEWS {tag}] {dkey} {direction}{severity} {category} dp={dp:+.3f} p_model={newp:.3f} conf={conf:.2f} shock={shock_flag}")

        # Start / refresh scale-out on meaningful events
        if severity in {"M", "L"} or shock_flag:
            self.scale_plan_until[dkey] = now + SHOCK_HOLD_S
            self.scale_steps_done[dkey] = 0
            self.scale_entry_target[dkey] = 0  # will be set on entry

    def trading_loop(self) -> None:
        last_dir: Dict[str, int] = {d: 0 for d in DEALS}

        while self.running:
            try:
                now = time.time()

                # p_model decay to anchors (reduces random-walk & post-spike donation)
                with self.lock:
                    for d in DEALS.keys():
                        self.p_model[d] = 0.985 * self.p_model[d] + 0.015 * P0_ANCHOR[d]

                for dkey in DEALS.keys():
                    if now < self.frozen_until[dkey]:
                        continue

                    with self.lock:
                        pmod = self.p_model[dkey]
                        ev_ts = self.last_event_ts[dkey]
                        ev_cat = self.last_event_cat[dkey]
                        ev_sev = self.last_event_sev[dkey]
                        ev_shock = self.last_event_shock[dkey]

                    p_buy = self.p_mkt_touch(dkey, "BUY")
                    p_sell = self.p_mkt_touch(dkey, "SELL")
                    if p_buy is None or p_sell is None:
                        continue

                    edge_buy = pmod - p_buy
                    edge_sell = pmod - p_sell
                    edge = edge_buy if abs(edge_buy) >= abs(edge_sell) else edge_sell

                    desired_dir = 1 if edge > 0 else -1
                    age = now - ev_ts
                    in_shock = age <= SHOCK_WINDOW_S
                    in_hold  = age <= SHOCK_HOLD_S

                    # Scale-out step?
                    scaled = self.maybe_scale_out(dkey, now)
                    if scaled is not None:
                        target_shares = scaled
                    else:
                        if in_shock:
                            # Shock: ignore TAU_ENTER and size big
                            target_shares = self.shock_size(edge, ev_cat, ev_sev, age_s=age, shock_flag=ev_shock)
                        else:
                            # Normal: trade only if decent edge
                            if abs(edge) >= TAU_ENTER:
                                target_shares = self.shock_size(edge, ev_cat, "S", age_s=999.0, shock_flag=False)
                            else:
                                # Flatten only if low edge and not in hold
                                if abs(edge) <= TAU_EXIT and not in_hold:
                                    target_shares = 0
                                else:
                                    continue

                    # Hold: don’t flip unless opposite fresh shock M/L (prevents whipsaw donation)
                    if in_hold and last_dir[dkey] != 0 and (1 if target_shares > 0 else -1 if target_shares < 0 else 0) == -last_dir[dkey]:
                        if in_shock and ev_sev in {"M", "L"}:
                            self._record_flip_and_maybe_freeze(dkey, now)
                            if now < self.frozen_until[dkey]:
                                continue
                        else:
                            continue

                    # Outside shock: apply cost filter
                    if not in_shock and target_shares != 0:
                        if not self.clears_costs(dkey, target_shares):
                            continue

                    # Execute
                    desired = self.desired_hedged_positions(dkey, target_shares)
                    self.trade_toward(desired)

                    # set scale entry target on first shock entry
                    if in_shock and self.scale_entry_target[dkey] == 0 and target_shares != 0:
                        self.scale_entry_target[dkey] = target_shares

                    last_dir[dkey] = 1 if target_shares > 0 else (-1 if target_shares < 0 else 0)

                time.sleep(TRADING_LOOP_S)

            except requests.RequestException as e:
                print(f"[WARN] trading API error: {e}")
                time.sleep(0.4)
            except Exception as e:
                print(f"[ERROR] trading loop: {e}")
                time.sleep(0.4)


# =============================================================================
# NEWS POLLER (persistent seen_ids; model hot swap supported)
# =============================================================================
class NewsPoller:
    def __init__(self, rit: RIT, engine: MergerArbEngine, clf: NewsClassifier):
        self.rit = rit
        self.engine = engine
        self.clf = clf
        self.running = True
        self.seen_ids: set[int] = self._load_seen()

    def _load_seen(self) -> set[int]:
        try:
            if os.path.exists(SEEN_IDS_PATH):
                with open(SEEN_IDS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return set(int(x) for x in data.get("seen_ids", []))
        except Exception:
            pass
        return set()

    def _save_seen(self) -> None:
        try:
            with open(SEEN_IDS_PATH, "w", encoding="utf-8") as f:
                json.dump({"seen_ids": sorted(list(self.seen_ids))[-5000:]}, f)
        except Exception:
            pass

    def stop(self) -> None:
        self.running = False

    def loop(self) -> None:
        print("[AUTO] Polling /news ... (persistent seen_ids enabled)")
        save_every = 25
        n = 0

        while self.running and self.engine.running:
            try:
                items = self.rit.get_news()
                items_sorted = sorted(items, key=lambda x: int(x.get("news_id", 0)))

                for it in items_sorted:
                    nid = int(it.get("news_id", 0))
                    if nid in self.seen_ids:
                        continue

                    headline = str(it.get("headline", "") or it.get("Headline", "") or "")
                    body = str(it.get("body", "") or it.get("Body", "") or "")
                    text = (headline + " " + body).strip()
                    if not text:
                        self.seen_ids.add(nid)
                        continue

                    deal, cat, direction, severity, conf, shock_flag = self.clf.predict(text)

                    # Fallback: infer deal from ticker field
                    if deal is None:
                        tick = str(it.get("ticker", "") or "").strip().upper()
                        if tick.startswith("D") and "-" in tick:
                            deal_id = tick.split("-", 1)[0].strip()
                            if deal_id in DEALS:
                                deal = deal_id
                        if deal is None and tick and tick != "ALL":
                            for dk, dd in DEALS.items():
                                if tick in {dd.target, dd.acquirer}:
                                    deal = dk
                                    break

                    if deal is None:
                        print(f"[AUTO] news_id={nid} could not infer deal | {headline[:90]}")
                        self.seen_ids.add(nid)
                        continue

                    tag = f"id={nid}"
                    self.engine.apply_news_update(deal, direction, severity, cat, conf=conf, shock_flag=shock_flag, tag=tag)

                    self.seen_ids.add(nid)
                    n += 1
                    if n % save_every == 0:
                        self._save_seen()

                time.sleep(NEWS_POLL_S)

            except requests.RequestException as e:
                print(f"[WARN] news API error: {e}")
                time.sleep(0.4)
            except Exception as e:
                print(f"[ERROR] news loop: {e}")
                time.sleep(0.4)


# =============================================================================
# TRAINING (supports your CSV headers)
# =============================================================================
def train_from_csvs(paths: List[str]) -> None:
    texts, cats, dirs, sevs = [], [], [], []

    def resolve(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
        lower_map = {h.strip().lower(): h for h in fieldnames}
        for c in candidates:
            h = lower_map.get(c.lower())
            if h:
                return h
        return None

    def norm_cat(x: str) -> Optional[str]:
        x = (x or "").strip().upper()
        if x in {"REG", "FIN", "SHR", "ALT", "PRC"}:
            return x
        if "REG" in x:
            return "REG"
        if "FIN" in x:
            return "FIN"
        if "SHR" in x or "SHARE" in x:
            return "SHR"
        if "ALT" in x:
            return "ALT"
        if "PRC" in x or "PROC" in x or "TIME" in x:
            return "PRC"
        return None

    def norm_dir(x: str) -> Optional[str]:
        x = (x or "").strip()
        if x in {"+", "pos", "positive", "up"}:
            return "+"
        if x in {"-", "neg", "negative", "down"}:
            return "-"
        return None

    def norm_sev(x: str) -> Optional[str]:
        x = (x or "").strip().upper()
        if x in {"S", "SMALL", "LOW", "1"}:
            return "S"
        if x in {"M", "MED", "MEDIUM", "2"}:
            return "M"
        if x in {"L", "LARGE", "HIGH", "3"}:
            return "L"
        return None

    total_rows = 0
    kept = 0

    for path in paths:
        with open(path, "r", encoding="utf-8-sig") as f:
            rdr = csv.DictReader(f)
            fns = rdr.fieldnames or []
            print(f"[TRAIN] Reading {path}")
            print(f"[TRAIN] Headers: {fns}")

            h_text = resolve(fns, ["headline", "text", "title", "news"])
            h_cat  = resolve(fns, ["category", "cat"])
            h_dir  = resolve(fns, ["direction", "dir", "sign"])
            h_sev  = resolve(fns, ["severity", "sev", "impact", "size"])

            if not (h_text and h_cat and h_dir and h_sev):
                raise ValueError(f"Could not resolve columns in {path}. Got: text={h_text}, cat={h_cat}, dir={h_dir}, sev={h_sev}")

            for row in rdr:
                total_rows += 1
                text = (row.get(h_text) or "").strip()
                cat = norm_cat(row.get(h_cat, ""))
                direction = norm_dir(row.get(h_dir, ""))
                sev = norm_sev(row.get(h_sev, ""))

                if not text or not cat or not direction or not sev:
                    continue

                texts.append(text)
                cats.append(cat)
                dirs.append(direction)
                sevs.append(sev)
                kept += 1

    print(f"[TRAIN] total_rows={total_rows} kept={kept}")
    if kept == 0:
        raise ValueError("No training samples accepted; check your CSV label values.")
    if kept < 800:
        print(f"[TRAIN WARN] Only {kept} samples. 800–2000+ is better for comp-day.")
    clf = NewsClassifier()
    clf.fit(texts, cats, dirs, sevs)
    clf.save(MODEL_PATH)
    print(f"[TRAIN OK] Saved model -> {MODEL_PATH} (samples={kept})")


# =============================================================================
# LIVE MAIN
# =============================================================================
def print_help() -> None:
    print("\nControls:")
    print("  show            -> show p_model, V, freeze status, last event age")
    print("  cancel          -> cancel all open orders")
    print("  reload          -> reload classifier from disk (no restart)")
    print("  quit            -> exit\n")


def main_live() -> None:
    rit = RIT()
    engine = MergerArbEngine(rit)

    try:
        clf = NewsClassifier.load(MODEL_PATH)
        print(f"[OK] loaded classifier: {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Could not load {MODEL_PATH}. Train first: python Merger_top_tier_5k.py train merger_arbitrage_news.csv") from e

    poller = NewsPoller(rit, engine, clf)

    t_trade = threading.Thread(target=engine.trading_loop, daemon=True)
    t_news  = threading.Thread(target=poller.loop, daemon=True)

    t_trade.start()
    t_news.start()

    print("🔥 TOP-TIER merger arb running (5k cap compliant, burst shock mode).")
    print_help()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            line = "quit"

        low = line.lower().strip()
        if not low:
            continue

        if low in {"quit", "q", "exit"}:
            engine.stop()
            poller.stop()
            break

        if low == "help":
            print_help()
            continue

        if low == "cancel":
            try:
                rit.cancel_all_orders()
                print("[OK] cancelled all orders")
            except Exception as e:
                print(f"[WARN] cancel failed: {e}")
            continue

        if low == "reload":
            try:
                poller.clf = NewsClassifier.load(MODEL_PATH)
                print("[OK] reloaded classifier from disk")
            except Exception as e:
                print(f"[WARN] reload failed: {e}")
            continue

        if low == "show":
            now = time.time()
            with engine.lock:
                for d in DEALS:
                    fr = engine.frozen_until[d]
                    frozen = (now < fr)
                    age = now - engine.last_event_ts[d]
                    print(
                        f"{d}: p_model={engine.p_model[d]:.3f} V={engine.V[d]:.2f} "
                        f"frozen={frozen} last_event_age={age:5.2f}s "
                        f"({engine.last_event_dir[d]}{engine.last_event_sev[d]} {engine.last_event_cat[d]} shock={engine.last_event_shock[d]})"
                    )
            continue

        print_help()


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1].lower() == "train":
        train_from_csvs(sys.argv[2:])  # list of paths
    else:
        main_live()