"""
RITC 2026 — Merger Arbitrage Bot (Strategy A v2: Maker-first + $ risk + continuous sizing)
========================================================================================

Key upgrades:
- Maker-first execution with limit orders and escalation ladder (no MARKET orders by default)
- One /securities snapshot per trading tick (speed + fewer API calls)
- Dollar-based gross / net exposure limits
- Continuous sizing based on expected mispricing: (K - V) * edge
- Confidence-scaled Δp updates from classifier
- Option A training: train on union of multiple CSV files

Run:
  python merger_arb_v2.py            # live
  python merger_arb_v2.py train a.csv b.csv c.csv

Assumptions:
- RIT /v1/orders accepts params-based order submissions:
    ticker, type, quantity, action, price (for LIMIT)
- /v1/securities returns position, bid, ask for each ticker (typical in RIT)
"""

from __future__ import annotations

import csv
import math
import re
import time
import threading
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

import requests
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =============================================================================
# CONFIG
# =============================================================================
API_KEY = "OR96FJU1"
HOST = "localhost"     # set to RIT server IP if not local
PORT = 9999
BASE_URL = f"http://{HOST}:{PORT}/v1"
TIMEOUT = 1.0

MODEL_PATH = "news_classifier.joblib"

# Order + execution
MAX_ORDER_SIZE = 5000
MIN_SECONDS_BETWEEN_ORDERS_PER_TICKER = 0.20
STALE_ORDER_REQUOTE_S = 0.8   # how long before we re-quote more aggressively

# Ladder of execution aggressiveness (0=maker, 1=step, 2=cross)
# We'll escalate if we're not converging to target.
EXEC_LADDER = [0, 1, 2]

# Commission estimate (used in cost filter)
COMMISSION = 0.02
COST_CUSHION = 1.2  # lower than before since we do maker-first

# Strategy thresholds (probability edge)
TAU_ENTER = 0.020
TAU_EXIT  = 0.010

# Risk limits (DOLLARS!)
# You MUST tune these to the case limits. Your old values were far too small.
GROSS_LIMIT_DOLLARS = 1_500_000
NET_LIMIT_DOLLARS   = 800_000

# Per-deal max exposure (DOLLARS)
PER_DEAL_GROSS_CAP_DOLLARS = 600_000

# Loop speeds
TRADING_LOOP_S = 0.15
NEWS_POLL_S = 0.20

# Safety stack
HYSTERESIS_MARGIN = 0.012
COOLDOWN_S = 1.2
FLIP_WINDOW_S = 20.0
MAX_FLIPS_IN_WINDOW = 4
FREEZE_S = 25.0
MAX_ABS_DP = 0.20

# Confidence scaling for Δp updates
# dp_applied = dp_table * dp_conf_scale(conf) * dp_danger_scale(danger)
def dp_conf_scale(conf: float) -> float:
    # conf in [0,1] -> scale in [0.35, 1.00]
    return 0.35 + 0.65 * max(0.0, min(1.0, conf))

def dp_danger_scale(danger: bool) -> float:
    return 0.75 if danger else 1.00


# =============================================================================
# DEAL DEFINITIONS
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

P0_ANCHOR = {"D1": 0.70, "D2": 0.55, "D3": 0.50, "D4": 0.38, "D5": 0.45}

CAT_MULT  = {"REG": 1.25, "FIN": 1.00, "SHR": 0.90, "ALT": 1.40, "PRC": 0.70}
DEAL_MULT = {"D1": 1.00, "D2": 1.05, "D3": 1.10, "D4": 1.30, "D5": 1.15}

BASE_IMPACT = {
    ("+",  "S"): 0.03, ("+",  "M"): 0.07, ("+",  "L"): 0.14,
    ("-",  "S"): -0.04, ("-", "M"): -0.09, ("-", "L"): -0.18,
}

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
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"X-API-Key": API_KEY})

    def _get(self, path: str, params: Optional[dict] = None):
        url = BASE_URL + path
        r = self.s.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, *, params: Optional[dict] = None):
        url = BASE_URL + path
        r = self.s.post(url, params=params, timeout=TIMEOUT)
        if r.status_code == 400:
            raise requests.HTTPError(
                f"400 Bad Request for {url}\nSent params={params}\nResponse body: {r.text}",
                response=r,
            )
        r.raise_for_status()
        if not r.text.strip():
            return None
        try:
            return r.json()
        except ValueError:
            return r.text

    # Market data
    def get_news(self) -> List[dict]:
        return self._get("/news")

    def get_securities(self) -> List[dict]:
        return self._get("/securities")

    # Orders
    def post_order_limit(self, ticker: str, qty: int, action: str, price: float):
        qty = int(qty)
        if qty <= 0:
            return None
        payload = {
            "ticker": ticker,
            "type": "LIMIT",
            "quantity": qty,
            "action": action,
            "price": round(float(price), 2),
        }
        return self._post("/orders", params=payload)

    def cancel_all_orders(self):
        return self._post("/commands/cancel", params={"all": 1})


# =============================================================================
# NEWS CLASSIFIER (TFIDF + LogisticRegression + keyword rules)
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

        # Category
        if any(w in s for w in ["antitrust", "regulator", "regulators", "ftc", "doj",
                                "competition bureau", "remedy", "injunction", "lawsuit", "litigation"]):
            cat = "REG"
        elif any(w in s for w in ["financing", "credit", "lenders", "repricing",
                                  "refinancing", "leverage", "funding", "debt"]):
            cat = "FIN"
        elif any(w in s for w in ["shareholder", "proxy", "vote", "tender", "board",
                                  "activist", "iss", "glass lewis"]):
            cat = "SHR"
        elif any(w in s for w in ["topping bid", "competing bid", "rival", "alternate bidder",
                                  "strategic alternatives", "go-shop"]):
            cat = "ALT"
        elif any(w in s for w in ["timeline", "delay", "extends", "extension", "closing date",
                                  "deadline", "process"]):
            cat = "PRC"
        else:
            cat = None

        # Direction
        pos_words = ["approved", "cleared", "secured", "committed", "support", "favorable", "accepts"]
        neg_words = ["blocked", "sues", "lawsuit", "challenge", "concern", "uncertain",
                     "withdraw", "terminate", "fails", "downgrade", "litigation"]

        pos_hit = any(w in s for w in pos_words)
        neg_hit = any(w in s for w in neg_words)

        direction = None
        if pos_hit and not neg_hit:
            direction = "+"
        elif neg_hit and not pos_hit:
            direction = "-"

        return {"category": cat, "direction": direction}

    @staticmethod
    def danger_regex(text: str) -> bool:
        s = text.lower()
        patterns = [
            r"\bsources?\b", r"\brumou?r\b", r"\bmay\b", r"\bmight\b",
            r"\bno decision\b", r"\bunclear\b", r"\bconsidering\b",
            r"\bexploring\b", r"\bdiscussions\b", r"\btalks\b",
        ]
        return any(re.search(p, s) for p in patterns)

    def fit(self, texts: List[str], y_cat: List[str], y_dir: List[str], y_sev: List[str]):
        X = self.vec.fit_transform(texts)
        self.clf_cat.fit(X, y_cat)
        self.clf_dir.fit(X, y_dir)
        self.clf_sev.fit(X, y_sev)

    def predict(self, text: str) -> Tuple[Optional[str], str, str, str, float, bool]:
        deal = self.infer_deal_from_text(text)
        rules = self.keyword_rules(text)
        danger = self.danger_regex(text)

        X = self.vec.transform([text])

        # category
        if rules["category"] is not None:
            cat, cat_conf = rules["category"], 0.95
        else:
            p = self.clf_cat.predict_proba(X)[0]
            i = int(p.argmax())
            cat, cat_conf = self.clf_cat.classes_[i], float(p[i])

        # direction
        if rules["direction"] is not None:
            direction, dir_conf = rules["direction"], 0.90
        else:
            p = self.clf_dir.predict_proba(X)[0]
            i = int(p.argmax())
            direction, dir_conf = self.clf_dir.classes_[i], float(p[i])

        # severity
        p = self.clf_sev.predict_proba(X)[0]
        i = int(p.argmax())
        severity, sev_conf = self.clf_sev.classes_[i], float(p[i])

        conf = min(cat_conf, dir_conf, sev_conf)
        return deal, cat, direction, severity, conf, danger

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
# TRADING ENGINE (Strategy A v2)
# =============================================================================
class MergerArbEngine:
    def __init__(self, rit: RIT):
        self.rit = rit
        self.lock = threading.Lock()
        self.running = True

        self.p_model = dict(P0_ANCHOR)
        self.V: Dict[str, float] = {}

        # Safety state
        self.last_signal_dir: Dict[str, int] = {d: 0 for d in DEALS.keys()}
        self.last_signal_edge: Dict[str, float] = {d: 0.0 for d in DEALS.keys()}
        self.last_trade_ts: Dict[str, float] = {d: 0.0 for d in DEALS.keys()}
        self.last_trade_dir: Dict[str, int] = {d: 0 for d in DEALS.keys()}
        self.flip_times: Dict[str, List[float]] = {d: [] for d in DEALS.keys()}
        self.frozen_until: Dict[str, float] = {d: 0.0 for d in DEALS.keys()}

        # Order throttles
        self.last_order_time_by_ticker: Dict[str, float] = {}
        self.last_quote_time_by_ticker: Dict[str, float] = {}
        self.last_quote_aggr_by_ticker: Dict[str, int] = {}

        # Infer V using start prices and anchor probabilities
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

    def stop(self) -> None:
        self.running = False

    def unfreeze(self, dkey: str) -> None:
        self.frozen_until[dkey] = 0.0
        self.flip_times[dkey] = []
        print(f"[UNFREEZE] {dkey} is unfrozen.")

    # ------------------------
    # Snapshot helpers
    # ------------------------
    @staticmethod
    def build_snapshot(rows: List[dict]) -> Dict[str, dict]:
        snap: Dict[str, dict] = {}
        for r in rows:
            t = str(r.get("ticker", "")).upper()
            if t:
                snap[t] = r
        return snap

    @staticmethod
    def best_bid_ask_from_snap(snap: Dict[str, dict], ticker: str) -> Tuple[Optional[float], Optional[float]]:
        r = snap.get(ticker.upper())
        if not r:
            return None, None
        bid = r.get("bid", None)
        ask = r.get("ask", None)
        try:
            b = float(bid) if bid is not None else None
            a = float(ask) if ask is not None else None
        except Exception:
            return None, None
        return b, a

    @staticmethod
    def mid_from_snap(snap: Dict[str, dict], ticker: str) -> Optional[float]:
        b, a = MergerArbEngine.best_bid_ask_from_snap(snap, ticker)
        if b is None or a is None:
            return None
        return 0.5 * (b + a)

    @staticmethod
    def positions_from_snap(snap: Dict[str, dict]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for t, r in snap.items():
            out[t] = int(r.get("position", 0) or 0)
        return out

    # ------------------------
    # Deal economics
    # ------------------------
    def current_K_from_snap(self, dkey: str, snap: Dict[str, dict]) -> Optional[float]:
        deal = DEALS[dkey]
        if deal.structure == "CASH":
            return deal.cash
        acq_mid = self.mid_from_snap(snap, deal.acquirer)
        if acq_mid is None:
            return None
        if deal.structure == "STOCK":
            return deal.ratio * acq_mid
        return deal.cash + deal.ratio * acq_mid

    def p_mkt_touch_from_snap(self, dkey: str, side: str, snap: Dict[str, dict]) -> Optional[float]:
        deal = DEALS[dkey]
        V = self.V[dkey]
        K = self.current_K_from_snap(dkey, snap)
        if K is None:
            return None
        bid, ask = self.best_bid_ask_from_snap(snap, deal.target)
        if bid is None or ask is None:
            return None
        P = ask if side == "BUY" else bid
        denom = K - V
        if abs(denom) < 1e-9:
            return None
        return clamp01((P - V) / denom)

    def intrinsic_target_from_snap(self, dkey: str, p: float, snap: Dict[str, dict]) -> Optional[float]:
        V = self.V[dkey]
        K = self.current_K_from_snap(dkey, snap)
        if K is None:
            return None
        return p * K + (1 - p) * V

    # ------------------------
    # Dollar risk
    # ------------------------
    def gross_net_dollars(self, pos: Dict[str, int], snap: Dict[str, dict]) -> Tuple[float, float]:
        gross = 0.0
        net = 0.0
        for t, sh in pos.items():
            mid = self.mid_from_snap(snap, t)
            if mid is None:
                continue
            gross += abs(sh) * mid
            net += sh * mid
        return gross, abs(net)

    def within_limits_after(self, pos: Dict[str, int], ticker: str, delta_shares: int, snap: Dict[str, dict]) -> bool:
        sim = dict(pos)
        sim[ticker] = sim.get(ticker, 0) + delta_shares
        gross, net = self.gross_net_dollars(sim, snap)
        return gross <= GROSS_LIMIT_DOLLARS and net <= NET_LIMIT_DOLLARS

    # ------------------------
    # Continuous sizing
    # ------------------------
    def size_from_edge(self, dkey: str, edge: float, snap: Dict[str, dict]) -> int:
        if abs(edge) < TAU_ENTER:
            return 0

        deal = DEALS[dkey]
        mid_t = self.mid_from_snap(snap, deal.target)
        if mid_t is None or mid_t <= 0:
            return 0

        K = self.current_K_from_snap(dkey, snap)
        if K is None:
            return 0
        V = self.V[dkey]
        denom = max(1e-6, abs(K - V))

        # expected mispricing per share ~ denom * edge
        mispricing = denom * abs(edge)

        # Map mispricing to target notional; tune slope
        # Bigger mispricing => larger notional, up to caps
        base_notional = 75_000
        slope = 2_500_000   # aggressive; tune for your case
        target_notional = base_notional + slope * abs(edge)

        # Deal-level cap
        target_notional = min(target_notional, PER_DEAL_GROSS_CAP_DOLLARS)

        shares = int(target_notional / mid_t)

        # Soft cap: also scale by mispricing (avoid huge size on tiny denom edges)
        # If mispricing is tiny, reduce.
        if mispricing < 0.30:
            shares = int(shares * 0.4)
        elif mispricing < 0.60:
            shares = int(shares * 0.7)

        shares = max(0, shares)
        return shares if edge > 0 else -shares

    # ------------------------
    # Hedging (probability-weighted)
    # ------------------------
    def desired_hedged_positions(self, dkey: str, target_shares: int, pmod: float) -> Dict[str, int]:
        deal = DEALS[dkey]
        out = {deal.target: target_shares}
        if deal.structure == "CASH":
            return out

        # Hedge ratio scale: hedge more as p increases
        # pmod ~ probability deal closes
        hedge_scale = 0.35 + 0.65 * pmod  # in [0.35, 1.0]
        hedge_shares = -int(round(deal.ratio * target_shares * hedge_scale))
        out[deal.acquirer] = hedge_shares
        return out

    # ------------------------
    # Cost filter (now favors maker fills)
    # ------------------------
    def clears_costs(self, dkey: str, target_shares: int, snap: Dict[str, dict]) -> bool:
        deal = DEALS[dkey]
        bid, ask = self.best_bid_ask_from_snap(snap, deal.target)
        if bid is None or ask is None:
            return False

        spread = max(0.0, ask - bid)
        half_spread = 0.5 * spread

        with self.lock:
            pmod = self.p_model[dkey]

        pintr = self.intrinsic_target_from_snap(dkey, pmod, snap)
        if pintr is None:
            return False

        # If we're buying, compare intrinsic to best ask (worst case)
        edge_dollars = (pintr - ask) if target_shares > 0 else (bid - pintr)

        required = COST_CUSHION * (half_spread + COMMISSION)
        return edge_dollars > required

    # ------------------------
    # Execution: maker-first ladder
    # ------------------------
    def throttle_ok(self, ticker: str) -> bool:
        now = time.time()
        last = self.last_order_time_by_ticker.get(ticker, 0.0)
        if now - last < MIN_SECONDS_BETWEEN_ORDERS_PER_TICKER:
            return False
        self.last_order_time_by_ticker[ticker] = now
        return True

    def quote_price(self, ticker: str, action: str, snap: Dict[str, dict], aggr: int) -> Optional[float]:
        bid, ask = self.best_bid_ask_from_snap(snap, ticker)
        if bid is None or ask is None:
            return None

        # aggr: 0 maker, 1 step toward touch, 2 cross (marketable limit)
        if action == "BUY":
            if aggr == 0:
                return bid           # maker
            elif aggr == 1:
                return min(ask, bid + 0.01)
            else:
                return ask           # cross
        else:
            if aggr == 0:
                return ask           # maker
            elif aggr == 1:
                return max(bid, ask - 0.01)
            else:
                return bid           # cross

    def maybe_escalate_aggr(self, ticker: str) -> int:
        now = time.time()
        last_q = self.last_quote_time_by_ticker.get(ticker, 0.0)
        cur_aggr = self.last_quote_aggr_by_ticker.get(ticker, 0)
        if now - last_q > STALE_ORDER_REQUOTE_S:
            return min(2, cur_aggr + 1)
        return cur_aggr

    def trade_toward(self, desired: Dict[str, int], snap: Dict[str, dict]) -> None:
        pos = self.positions_from_snap(snap)

        for ticker, want in desired.items():
            cur = pos.get(ticker, 0)
            diff = want - cur
            if diff == 0:
                continue
            if not self.throttle_ok(ticker):
                continue

            # choose step size
            step = int(math.copysign(min(MAX_ORDER_SIZE, abs(diff)), diff))

            # enforce dollar risk
            while step != 0 and not self.within_limits_after(pos, ticker, step, snap):
                step = int(step / 2)

            if step == 0:
                continue

            action = "BUY" if step > 0 else "SELL"
            qty = abs(step)

            # escalation
            aggr = self.maybe_escalate_aggr(ticker)
            price = self.quote_price(ticker, action, snap, aggr)
            if price is None:
                continue

            try:
                self.rit.post_order_limit(ticker, qty, action, price)
                self.last_quote_time_by_ticker[ticker] = time.time()
                self.last_quote_aggr_by_ticker[ticker] = aggr
            except Exception as e:
                print(f"[WARN] order failed {ticker} {action} {qty}@{price} aggr={aggr}: {e}")
                continue

            # update local pos assumption optimistically only when crossing
            # maker orders may not fill; we'll rely on next snapshot to see progress
            if aggr >= 2:
                pos[ticker] = pos.get(ticker, 0) + (qty if action == "BUY" else -qty)

    # ------------------------
    # News updates
    # ------------------------
    def apply_news_update(self, dkey: str, direction: str, severity: str, category: str,
                          conf: float, danger: bool, tag: str) -> None:
        base = BASE_IMPACT[(direction, severity)]
        dp = base * CAT_MULT[category] * DEAL_MULT[dkey]

        # confidence + danger scaling
        dp *= dp_conf_scale(conf)
        dp *= dp_danger_scale(danger)

        # clamp
        dp = max(-MAX_ABS_DP, min(MAX_ABS_DP, dp))

        with self.lock:
            self.p_model[dkey] = clamp01(self.p_model[dkey] + dp)
            newp = self.p_model[dkey]

        print(f"[NEWS {tag}] {dkey} {direction}{severity} {category} conf={conf:.2f} danger={danger}  Δp={dp:+.4f}  p_model={newp:.3f}")

    # ------------------------
    # Flip / freeze
    # ------------------------
    def _prune_flip_times(self, dkey: str, now: float) -> None:
        self.flip_times[dkey] = [t for t in self.flip_times[dkey] if now - t <= FLIP_WINDOW_S]

    def _record_flip_and_maybe_freeze(self, dkey: str, now: float) -> None:
        self.flip_times[dkey].append(now)
        self._prune_flip_times(dkey, now)
        if len(self.flip_times[dkey]) > MAX_FLIPS_IN_WINDOW:
            self.frozen_until[dkey] = max(self.frozen_until[dkey], now + FREEZE_S)
            print(f"[FREEZE] {dkey} frozen {FREEZE_S:.0f}s (flips>{MAX_FLIPS_IN_WINDOW} in {FLIP_WINDOW_S:.0f}s)")

    # ------------------------
    # Main trading loop
    # ------------------------
    def trading_loop(self) -> None:
        while self.running:
            try:
                now = time.time()

                # Snapshot once per tick
                rows = self.rit.get_securities()
                snap = self.build_snapshot(rows)

                for dkey, deal in DEALS.items():
                    if now < self.frozen_until[dkey]:
                        continue

                    with self.lock:
                        pmod = self.p_model[dkey]

                    p_buy  = self.p_mkt_touch_from_snap(dkey, "BUY", snap)
                    p_sell = self.p_mkt_touch_from_snap(dkey, "SELL", snap)
                    if p_buy is None or p_sell is None:
                        continue

                    edge_buy  = pmod - p_buy
                    edge_sell = pmod - p_sell

                    # Flatten condition
                    if abs(edge_buy) < TAU_EXIT and abs(edge_sell) < TAU_EXIT:
                        desired = self.desired_hedged_positions(dkey, 0, pmod)
                        self.trade_toward(desired, snap)
                        self.last_signal_dir[dkey] = 0
                        self.last_signal_edge[dkey] = 0.0
                        self.last_trade_dir[dkey] = 0
                        continue

                    # choose strongest edge
                    edge = edge_buy if abs(edge_buy) >= abs(edge_sell) else edge_sell
                    desired_dir = 1 if edge > 0 else -1

                    # cooldown on flips
                    if self.last_trade_dir[dkey] != 0 and desired_dir == -self.last_trade_dir[dkey]:
                        if now - self.last_trade_ts[dkey] < COOLDOWN_S:
                            continue

                    # hysteresis
                    prev_dir = self.last_signal_dir[dkey]
                    prev_edge = self.last_signal_edge[dkey]
                    if prev_dir != 0 and desired_dir == -prev_dir:
                        if abs(edge) < abs(prev_edge) + HYSTERESIS_MARGIN:
                            continue

                    # sizing
                    target_shares = self.size_from_edge(dkey, edge, snap)
                    if target_shares == 0:
                        continue

                    # cost filter
                    if not self.clears_costs(dkey, target_shares, snap):
                        continue

                    # flip count / freeze
                    if prev_dir != 0 and desired_dir == -prev_dir:
                        self._record_flip_and_maybe_freeze(dkey, now)
                        if now < self.frozen_until[dkey]:
                            continue

                    # execute target + hedge
                    desired = self.desired_hedged_positions(dkey, target_shares, pmod)
                    self.trade_toward(desired, snap)

                    # update state
                    self.last_signal_dir[dkey] = desired_dir
                    self.last_signal_edge[dkey] = edge
                    self.last_trade_dir[dkey] = desired_dir
                    self.last_trade_ts[dkey] = now

                time.sleep(TRADING_LOOP_S)

            except requests.RequestException as e:
                print(f"[WARN] trading API error: {e}")
                time.sleep(0.5)
            except Exception as e:
                print(f"[ERROR] trading loop: {e}")
                time.sleep(0.5)


# =============================================================================
# NEWS POLLER
# =============================================================================
class NewsPoller:
    def __init__(self, rit: RIT, engine: MergerArbEngine, clf: NewsClassifier):
        self.rit = rit
        self.engine = engine
        self.clf = clf
        self.running = True
        self.seen_ids: set[int] = set()

    def stop(self) -> None:
        self.running = False

    def loop(self) -> None:
        print("[AUTO] Polling /news ...")
        while self.running and self.engine.running:
            try:
                items = self.rit.get_news()
                items_sorted = sorted(items, key=lambda x: int(x.get("news_id", 0)))

                for it in items_sorted:
                    nid = int(it.get("news_id", 0))
                    if nid in self.seen_ids:
                        continue

                    headline = str(it.get("headline", "") or "")
                    body = str(it.get("body", "") or "")
                    text = (headline + " " + body).strip()
                    if not text:
                        self.seen_ids.add(nid)
                        continue

                    deal, cat, direction, severity, conf, danger = self.clf.predict(text)

                    # Fallback deal from ticker field if present
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
                    # apply update (confidence-scaled)
                    self.engine.apply_news_update(deal, direction, severity, cat, conf, danger, tag=tag)
                    self.seen_ids.add(nid)

                time.sleep(NEWS_POLL_S)

            except requests.RequestException as e:
                print(f"[WARN] news API error: {e}")
                time.sleep(0.6)
            except Exception as e:
                print(f"[ERROR] news loop: {e}")
                time.sleep(0.6)


# =============================================================================
# TRAINING (Option A: union of multiple CSV files)
# =============================================================================
def train_from_csvs(paths: List[str]) -> None:
    texts, cats, dirs, sevs = [], [], [], []

    for path in paths:
        with open(path, "r", encoding="utf-8-sig") as f:
            rdr = csv.DictReader(f)

            print(f"[TRAIN] Reading {path}")
            print(f"[TRAIN] Headers detected: {rdr.fieldnames}")

            for row in rdr:
                text = (row.get("Headline") or "").strip()
                cat = (row.get("Category") or "").strip().upper()
                direction = (row.get("Direction") or "").strip()
                sev = (row.get("Severity") or "").strip().upper()

                if not text:
                    continue
                if cat not in {"REG", "FIN", "SHR", "ALT", "PRC"}:
                    continue
                if direction not in {"+", "-"}:
                    continue
                if sev not in {"S", "M", "L"}:
                    continue

                texts.append(text)
                cats.append(cat)
                dirs.append(direction)
                sevs.append(sev)

    print(f"[TRAIN] Samples kept: {len(texts)}")

    if len(texts) == 0:
        raise ValueError("No samples found. Check CSV column names.")

    clf = NewsClassifier()
    clf.fit(texts, cats, dirs, sevs)
    clf.save(MODEL_PATH)

    print(f"[TRAIN OK] Saved model -> {MODEL_PATH} (samples={len(texts)})")

# =============================================================================
# LIVE MAIN
# =============================================================================
def print_help() -> None:
    print("\nControls:")
    print("  show            -> show p_model and frozen status")
    print("  cancel          -> cancel all open orders")
    print("  unfreeze D4     -> unfreeze a deal (D1..D5)")
    print("  quit            -> exit\n")


def main_live() -> None:
    rit = RIT()
    engine = MergerArbEngine(rit)

    try:
        clf = NewsClassifier.load(MODEL_PATH)
        print(f"[OK] loaded classifier: {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Could not load {MODEL_PATH}. Train first: python merger_arb_v2.py train a.csv b.csv") from e

    t_trade = threading.Thread(target=engine.trading_loop, daemon=True)
    poller = NewsPoller(rit, engine, clf)
    t_news = threading.Thread(target=poller.loop, daemon=True)

    t_trade.start()
    t_news.start()

    print("✅ Merger arb v2 running (maker-first, $ risk, continuous sizing).")
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

        if low == "show":
            now = time.time()
            with engine.lock:
                for d in DEALS.keys():
                    frozen = now < engine.frozen_until[d]
                    print(f"{d}: p_model={engine.p_model[d]:.3f}  V={engine.V[d]:.2f}  frozen={frozen}")
            continue

        m = re.match(r"unfreeze\s+(d[1-5])", low)
        if m:
            engine.unfreeze(m.group(1).upper())
            continue

        print_help()


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1].lower() == "train":
        train_from_csvs(sys.argv[2:])
    else:
        main_live()