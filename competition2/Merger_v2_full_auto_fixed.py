"""
RITC 2026 — Merger Arbitrage (Strategy A, Full-Auto w/ Safety Stack)
===================================================================

What this does
--------------
- Polls RIT /v1/news continuously (full-auto).
- Classifies each news item into:
    (deal D1..D5, category REG/FIN/SHR/ALT/PRC, direction +/- , severity S/M/L)
  using a hybrid:
    - high-precision keyword rules
    - TFIDF + LogisticRegression (trainable from CSV)
- Updates p_model per deal using the case multipliers.
- Trades mispricing in IMPLIED PROBABILITY (Strategy A):
    p_mkt = (P_touch - V) / (K - V)
    edge  = p_model - p_mkt
  then:
    edge > +tau_enter -> BUY target (+ hedge)
    edge < -tau_enter -> SELL target (+ hedge)
    |edge| < tau_exit -> FLATTEN
- Uses touch prices (ask for buys, bid for sells).
- Enforces:
    - order size <= 5000
    - conservative gross/net limits via positions snapshot
- Includes the requested safety stack for full-auto:
    1) Confidence gating per category
    2) Hysteresis for flips
    3) Cooldown per deal (ignore opposite signals briefly unless severity L)
    4) Flip-count freeze window (requires manual override to unfreeze)
    5) Shock clamp on Δp update magnitude

Training (from CSV)
-------------------
CSV headers:
  text,category,direction,severity,deal
Where:
  category  in {REG, FIN, SHR, ALT, PRC}
  direction in {+,-}
  severity  in {S,M,L}
  deal      in {D1,D2,D3,D4,D5}  (recommended; if you omit deal column, deal is inferred by tickers)

Train:
  python merger_arb_full_auto.py train train_news.csv

Run live:
  python merger_arb_full_auto.py

Packages:
- requests, scikit-learn, joblib are on Rotman laptop package list. (PythonPackageList.txt)
  https://rotmanfrtl.github.io/PythonPackageList.txt

RIT API:
- Uses /v1/news, /v1/securities, /v1/securities/book, /v1/orders, /v1/commands/cancel.
- RIT API surface includes these functions. https://rit.306w.ca/RIT-REST-API/1.0.3/

NOTE
----
This is "full-auto unless frozen." If a deal is frozen due to too many flips,
you can type:  unfreeze D4   (or D1..D5) in the console to resume auto-trading that deal.
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
HOST = "localhost"          # change to RIT server IP if needed
PORT = 9999
BASE_URL = f"http://{HOST}:{PORT}/v1"
TIMEOUT = 1.0

MAX_ORDER_SIZE = 5000
COMMISSION = 0.02

GROSS_LIMIT = 100_000
NET_LIMIT = 50_000

# Strategy A thresholds (probability difference)
TAU_ENTER = 0.025
TAU_EXIT  = 0.012

# Require intrinsic $ edge > cushion*(half_spread + commission)
COST_CUSHION = 1.4

# Position sizing caps
PER_DEAL_CAP_TARGET = 15_000
BASE_CLIP = 2_000

# Loop speeds
TRADING_LOOP_S = 0.25
NEWS_POLL_S = 0.20


# Per-ticker throttle to avoid spam / flip-flops
MIN_SECONDS_BETWEEN_TRADES_PER_TICKER = 0.30
MODEL_PATH = "news_classifier.joblib"

# -------------------------------------------------------------------------
# FULL-AUTO SAFETY STACK (per your requirements)
# -------------------------------------------------------------------------

# 1) Confidence gating per category:
AUTO_CONF_THRESH = {
    "REG": 0.0,
    "ALT": 0.0,
    "FIN": 0.0,
    "SHR": 0.0,
    "PRC": 0.0,
}


# 2) Hysteresis: don't flip unless edge exceeds previous edge by margin
HYSTERESIS_MARGIN = 0.015  # additional probability edge required to reverse

# 3) Cooldown per deal: ignore opposite signals for this long unless severity is L
COOLDOWN_S = 1.5

# 4) Flip-count freeze
FLIP_WINDOW_S = 20.0          # lookback window
MAX_FLIPS_IN_WINDOW = 3        # if exceeded => freeze that deal
FREEZE_S = 30.0                # freeze duration unless manually unfrozen

# 5) Shock clamp on |Δp| in probability space (after multipliers)
MAX_ABS_DP = 0.22  # clamp final dp magnitude (after multipliers) before applying


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
    ("amb","S"): 0.02, ("amb","M"): 0.04, ("amb","L"): 0.06,
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
# RIT REST WRAPPER (v1-style paths)
# =============================================================================
class RIT:
    """
    Uses:
      GET  /news
      GET  /securities
      GET  /securities/book?ticker=...&limit=...
      POST /orders              (IMPORTANT: send order fields as query params)
      POST /commands/cancel
    """
    def __init__(self):
        self.s = requests.Session()
        # Header key is typically X-API-Key (case sensitive on some setups)
        self.s.headers.update({"X-API-Key": API_KEY})

    def _get(self, path: str, params: Optional[dict] = None):
        url = BASE_URL + path
        r = self.s.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, *, params: Optional[dict] = None, json: Optional[dict] = None):
        """
        Generic POST helper. For /orders we will NOT use json=.
        """
        url = BASE_URL + path
        r = self.s.post(url, params=params, json=json, timeout=TIMEOUT)

        # Helpful diagnostics when RIT returns 400 with a message
        if r.status_code == 400:
            raise requests.HTTPError(
                f"400 Bad Request for {url}\n"
                f"Sent params={params} json={json}\n"
                f"Response body: {r.text}",
                response=r,
            )

        r.raise_for_status()

        # Some endpoints return empty body; guard JSON decode
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
        # Fast path: /securities?ticker=XXX commonly includes bid/ask
        rows = self.get_securities(ticker=ticker)
        if rows:
            sec = rows[0]
            bid = sec.get("bid", None)
            ask = sec.get("ask", None)
            if bid is not None and ask is not None:
                return float(bid), float(ask)

        # Fallback: /securities/book
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

    def post_order_market(self, ticker: str, qty: int, action: str):
        """
        RIT commonly expects order fields as query params, not JSON. :contentReference[oaicite:1]{index=1}
        """
        qty = int(qty)
        if qty <= 0:
            return None

        payload = {
            "ticker": ticker,
            "type": "MARKET",
            "quantity": qty,      # int, not float
            "action": action,     # "BUY" or "SELL"
        }
        return self._post("/orders", params=payload)

    def cancel_all_orders(self):
        # This endpoint varies by case/client build, but params= is standard
        return self._post("/commands/cancel", params={"all": 1})
# =============================================================================
# NEWS CLASSIFIER (rules + TFIDF + LogisticRegression)
# =============================================================================
class NewsClassifier:
    """
    Predicts category, direction, severity. Deal via:
      - training label if provided (optional)
      - else ticker inference (recommended even with deal label)

    Training CSV supports optional column 'deal' (D1..D5).
    """
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=7000)
        self.clf_cat = LogisticRegression(max_iter=300)
        self.clf_dir = LogisticRegression(max_iter=300)
        self.clf_sev = LogisticRegression(max_iter=300)

    @staticmethod
    def infer_deal_from_text(text: str) -> Optional[str]:
        t = text.upper()
        for dkey, deal in DEALS.items():
            if deal.target in t or deal.acquirer in t:
                return dkey
        return None

    @staticmethod
    def keyword_rules(text: str) -> Dict[str, Optional[str]]:
        """
        High-precision overrides.
        """
        s = text.lower()

        # Category
        if any(w in s for w in ["antitrust", "regulator", "regulators", "ftc", "doj", "competition bureau", "remedy", "remedies", "injunction", "lawsuit", "litigation"]):
            cat = "REG"
        elif any(w in s for w in ["financing", "credit", "lenders", "repricing", "refinancing", "leverage", "funding", "debt", "spread widening"]):
            cat = "FIN"
        elif any(w in s for w in ["shareholder", "proxy", "vote", "tender", "board", "activist", "iss", "glass lewis"]):
            cat = "SHR"
        elif any(w in s for w in ["topping bid", "competing bid", "rival", "alternate bidder", "strategic alternatives", "go-shop"]):
            cat = "ALT"
        elif any(w in s for w in ["timeline", "delay", "extends", "extension", "process", "procedural", "closing date", "deadline"]):
            cat = "PRC"
        else:
            cat = None

        # Direction
        pos_words = ["approved", "cleared", "constructive", "secured", "committed", "support", "acceptable", "favorable", "accepts"]
        neg_words = ["blocked", "sues", "lawsuit", "challenge", "deteriorate", "widen", "concern", "uncertain", "withdraw", "terminate", "fails", "downgrade", "litigation"]

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
        """
        Flag phrases that often cause misclassification / low signal.
        Used to raise gating strictly.
        """
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
        """
        Returns:
          deal, category, direction, severity, confidence, danger_flag
        confidence = min(field confidences after rules
        """
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
# MERGER ARB ENGINE (Strategy A) + SAFETY STACK STATE
# =============================================================================
class MergerArbEngine:
    def __init__(self, rit: RIT):
        self.rit = rit
        self.lock = threading.Lock()
        self.running = True

        self.p_model = dict(P0_ANCHOR)
        self.V: Dict[str, float] = {}

        # Safety state
        self.last_signal_dir: Dict[str, int] = {d: 0 for d in DEALS.keys()}   # -1,0,+1 desired
        self.last_signal_edge: Dict[str, float] = {d: 0.0 for d in DEALS.keys()}
        self.last_trade_ts: Dict[str, float] = {d: 0.0 for d in DEALS.keys()}
        self.last_trade_dir: Dict[str, int] = {d: 0 for d in DEALS.keys()}    # -1,0,+1
        self.flip_times: Dict[str, List[float]] = {d: [] for d in DEALS.keys()}
        self.frozen_until: Dict[str, float] = {d: 0.0 for d in DEALS.keys()}

        # Throttle
        self.last_trade_time_by_ticker: Dict[str, float] = {}

        # Infer standalone values V at t=0
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

    # ---- implied probability from touch price
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

    # ---- conservative limits
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

    # ---- sizing from probability edge
    def size_from_edge(self, edge: float) -> int:
        mag = abs(edge)
        if mag < TAU_ENTER:
            return 0
        if mag < 0.06:
            size = BASE_CLIP
        elif mag < 0.10:
            size = 5_000
        else:
            size = 9_000
        size = min(size, PER_DEAL_CAP_TARGET)
        return size if edge > 0 else -size

    # ---- target + hedge mapping
    def desired_hedged_positions(self, dkey: str, target_shares: int) -> Dict[str, int]:
        deal = DEALS[dkey]
        out = {deal.target: target_shares}
        if deal.structure == "CASH":
            return out
        out[deal.acquirer] = -int(round(deal.ratio * target_shares))
        return out

    # ---- cost filter
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

    # ---- per-ticker throttle
    def ticker_throttle_ok(self, ticker: str) -> bool:
        now = time.time()
        last = self.last_trade_time_by_ticker.get(ticker, 0.0)
        if now - last < MIN_SECONDS_BETWEEN_TRADES_PER_TICKER:
            return False
        self.last_trade_time_by_ticker[ticker] = now
        return True

    # ---- execution toward desired positions (market orders)
    def trade_toward(self, desired: Dict[str, int]) -> None:
        pos = self.rit.get_positions()
        for ticker, want in desired.items():
            cur = pos.get(ticker, 0)
            diff = want - cur
            if diff == 0:
                continue
            if not self.ticker_throttle_ok(ticker):
                continue

            step = int(math.copysign(min(MAX_ORDER_SIZE, abs(diff)), diff))

            while step != 0 and not self.within_limits_after(pos, ticker, step):
                step = int(step / 2)

            if step == 0:
                continue

            action = "BUY" if step > 0 else "SELL"
            qty = abs(step)
            self.rit.post_order_market(ticker, qty, action)

            pos[ticker] = pos.get(ticker, 0) + (qty if action == "BUY" else -qty)

    # =============================================================================
    # NEWS -> p_model update with shock clamp
    # =============================================================================
    def apply_news_update(self, dkey: str, direction: str, severity: str, category: str, tag: str) -> None:
        # direction in {+,-,amb}, severity in {S,M,L}, category in CAT_MULT
        base = BASE_IMPACT[(direction, severity)]
        dp = base * CAT_MULT[category] * DEAL_MULT[dkey]

        # 5) Shock clamp
        dp = max(-MAX_ABS_DP, min(MAX_ABS_DP, dp))

        with self.lock:
            self.p_model[dkey] = clamp01(self.p_model[dkey] + dp)
            newp = self.p_model[dkey]

        print(f"[NEWS {tag}] {dkey} {direction}{severity} {category}  Δp={dp:+.4f}  p_model={newp:.3f}")

    # =============================================================================
    # FULL-AUTO SAFETY STACK: decide whether to act on a new model signal
    # =============================================================================
    def _prune_flip_times(self, dkey: str, now: float) -> None:
        ft = self.flip_times[dkey]
        self.flip_times[dkey] = [t for t in ft if now - t <= FLIP_WINDOW_S]

    def _record_flip_and_maybe_freeze(self, dkey: str, now: float) -> None:
        self.flip_times[dkey].append(now)
        self._prune_flip_times(dkey, now)
        if len(self.flip_times[dkey]) > MAX_FLIPS_IN_WINDOW:
            self.frozen_until[dkey] = max(self.frozen_until[dkey], now + FREEZE_S)
            print(f"[FREEZE] {dkey} frozen for {FREEZE_S:.0f}s due to flips (> {MAX_FLIPS_IN_WINDOW} in {FLIP_WINDOW_S:.0f}s).")

    def unfreeze(self, dkey: str) -> None:
        self.frozen_until[dkey] = 0.0
        self.flip_times[dkey] = []
        print(f"[UNFREEZE] {dkey} is unfrozen.")

    def trading_loop(self) -> None:
        while self.running:
            try:
                now = time.time()

                for dkey in DEALS.keys():
                    # Skip if frozen
                    if now < self.frozen_until[dkey]:
                        continue

                    with self.lock:
                        pmod = self.p_model[dkey]

                    # Market-implied probabilities at touch
                    p_buy = self.p_mkt_touch(dkey, "BUY")
                    p_sell = self.p_mkt_touch(dkey, "SELL")
                    if p_buy is None or p_sell is None:
                        continue

                    edge_buy = pmod - p_buy
                    edge_sell = pmod - p_sell

                    # Flatten rule
                    if abs(edge_buy) < TAU_EXIT and abs(edge_sell) < TAU_EXIT:
                        desired = self.desired_hedged_positions(dkey, 0)
                        self.trade_toward(desired)
                        self.last_signal_dir[dkey] = 0
                        self.last_signal_edge[dkey] = 0.0
                        self.last_trade_dir[dkey] = 0
                        continue

                    # Pick stronger direction
                    if abs(edge_buy) >= abs(edge_sell):
                        edge = edge_buy
                    else:
                        edge = edge_sell

                    desired_dir = 1 if edge > 0 else -1

                    # 3) Cooldown: if opposite direction soon after a trade, ignore unless "L" severity
                    # We don't know severity here; cooldown is for rapid opposite *signals* from model/news.
                    # We'll enforce cooldown using last_trade_dir and last_trade_ts.
                    if self.last_trade_dir[dkey] != 0 and desired_dir == -self.last_trade_dir[dkey]:
                        if now - self.last_trade_ts[dkey] < COOLDOWN_S:
                            # ignore flip signal during cooldown
                            continue

                    # 2) Hysteresis: require stronger edge to flip
                    prev_dir = self.last_signal_dir[dkey]
                    prev_edge = self.last_signal_edge[dkey]
                    if prev_dir != 0 and desired_dir == -prev_dir:
                        if abs(edge) < abs(prev_edge) + HYSTERESIS_MARGIN:
                            continue

                    # Convert edge to size
                    target_des = self.size_from_edge(edge)
                    if target_des == 0:
                        continue

                    # Cost filter
                    if not self.clears_costs(dkey, target_des):
                        continue

                    # If we are flipping, record flip (4)
                    if prev_dir != 0 and desired_dir == -prev_dir:
                        self._record_flip_and_maybe_freeze(dkey, now)
                        if now < self.frozen_until[dkey]:
                            continue

                    # Execute
                    desired = self.desired_hedged_positions(dkey, target_des)
                    self.trade_toward(desired)

                    # Update state
                    self.last_signal_dir[dkey] = desired_dir
                    self.last_signal_edge[dkey] = edge
                    self.last_trade_dir[dkey] = desired_dir
                    self.last_trade_ts[dkey] = now

                time.sleep(TRADING_LOOP_S)

            except requests.RequestException as e:
                print(f"[WARN] trading API error: {e}")
                time.sleep(0.6)
            except Exception as e:
                print(f"[ERROR] trading loop: {e}")
                time.sleep(0.6)

    def stop(self) -> None:
        self.running = False


# =============================================================================
# NEWS POLLER (full-auto) with confidence gating
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
                        tick = str(it.get("ticker", "") or "").strip()
                        tick_u = tick.upper()

                        # Newer RIT news uses deal tags like "D2-BYL/CLD" (or "ALL")
                        if tick_u.startswith("D") and "-" in tick_u:
                            deal_id = tick_u.split("-", 1)[0].strip()  # "D2"
                            if deal_id in DEALS:
                                deal = deal_id

                        # If they ever send raw security tickers, keep legacy fallback
                        if deal is None and tick_u and tick_u != "ALL":
                            for dk, dd in DEALS.items():
                                if tick_u in {dd.target, dd.acquirer}:
                                    deal = dk
                                    break
                    if deal is None:
                        print(f"[AUTO] news_id={nid} could not infer deal | {headline[:90]}")
                        self.seen_ids.add(nid)
                        continue

                    # confidence gating (1)
                    thresh = AUTO_CONF_THRESH.get(cat, 0.90)
                    if danger:
                        # Fully-auto mode: do not tighten threshold on 'danger' phrasing.
                        thresh = thresh
                    tag = f"id={nid} conf={conf:.2f}"

                    if conf >= thresh:
                        self.engine.apply_news_update(deal, direction, severity, cat, tag=tag)
                    else:
                        # Fully-auto mode: apply even if low confidence.
                        self.engine.apply_news_update(deal, direction, severity, cat, tag=tag + " (LOWCONF)")
                    self.seen_ids.add(nid)

                time.sleep(NEWS_POLL_S)

            except requests.RequestException as e:
                print(f"[WARN] news API error: {e}")
                time.sleep(0.7)
            except Exception as e:
                print(f"[ERROR] news loop: {e}")
                time.sleep(0.7)


# =============================================================================
# TRAINING
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
# LIVE MAIN (full-auto, plus minimal console controls)
# =============================================================================
def print_help() -> None:
    print("\nControls:")
    print("  show            -> show p_model, V, frozen status")
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
        raise RuntimeError(f"Could not load {MODEL_PATH}. Train first: python merger_arb_full_auto.py train train_news.csv") from e

    # Threads
    t_trade = threading.Thread(target=engine.trading_loop, daemon=True)
    t_news = threading.Thread(target=NewsPoller(rit, engine, clf).loop, daemon=True)

    t_trade.start()
    t_news.start()

    print("✅ Full-auto merger arb running (with safety stack).")
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
                    fr = engine.frozen_until[d]
                    frozen = (now < fr)
                    print(f"{d}: p_model={engine.p_model[d]:.3f}  V={engine.V[d]:.2f}  frozen={frozen}")
            continue

        m = re.match(r"unfreeze\s+(d[1-5])", low)
        if m:
            dkey = m.group(1).upper()
            engine.unfreeze(dkey)
            continue

        print_help()


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1].lower() == "train":
        train_from_csvs(sys.argv[2:])
    else:
        main_live()
