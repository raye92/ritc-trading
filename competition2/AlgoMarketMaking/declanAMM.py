"""
RITC 2026 Algorithmic Market Making Trading Case - REST API Basic Script
Strategy:
- Quote passively (LIMIT) on both sides to earn spread + rebates.
- Inventory skew: if long, make ask more aggressive & bid less aggressive; vice versa if short.
- Risk bands using aggregate abs position.
- News-aware regime:
    * Pre-close (last ~5 ticks of each minute): cancel risk-increasing orders and aim near-flat.
    * Post-close (first ~2 ticks): widen / shrink size (or briefly pause) to avoid getting picked off.
"""

import os
from time import sleep
from dotenv import load_dotenv
import requests

# -------------------- Session / Config --------------------
s = requests.Session()
load_dotenv()
APIKEY = os.getenv("API_KEY")
s.headers.update({"X-API-key": APIKEY})

BASEURL = "http://localhost:9999/v1"

# Limits / parameters (tune per heat)
MAX_EXPOSURE = 15000          # aggregate abs position cap target (risk-managed below this)
ORDER_LIMIT = 10_000          # max single order size
BASE_QUOTE_SIZE = 2000        # normal quote size per side
MIN_QUOTE_SIZE = 200          # smallest quote size
POS_CAP_PER_TICKER = 6000     # soft cap per ticker before heavy skew/flatten

# Quote shaping
TICK_SIZE = 0.01
BASE_HALF_SPREAD_TICKS = 1    # 1 tick each side as baseline
MAX_SKEW_TICKS = 6            # max skew (ticks) when at +/- POS_CAP_PER_TICKER
VOL_WIDEN_MULT = 10.0         # widens based on EWMA vol (scaled)
EWMA_ALPHA = 0.25             # volatility smoothing

# Timing / regimes (based on your tick%60 comment)
TICKS_PER_MINUTE = 60
PRE_CLOSE_START = 55          # tick%60 >= 55 => pre-close regime
POST_CLOSE_END = 5            # tick%60 <= 2  => post-close shock regime

# Loop pacing
SLEEP_NORMAL = 0.15
SLEEP_PRE_CLOSE = 0.08
SLEEP_POST_CLOSE = 0.12

PRINT_HEART_BEAT = True


# -------------------- API helpers --------------------
def get_tick():
    resp = s.get(f"{BASEURL}/case")
    if resp.ok:
        case = resp.json()
        return case["tick"], case["status"]
    return None, None


def get_ticker_list():
    resp = s.get(f"{BASEURL}/securities")
    if resp.ok:
        secs = resp.json()
        return [i["ticker"] for i in secs]
    return []


def get_bid_ask(ticker: str):
    resp = s.get(f"{BASEURL}/securities/book", params={"ticker": ticker})
    if not resp.ok:
        return None, None
    book = resp.json()
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    if not bids or not asks:
        return None, None
    best_bid = bids[0]["price"]
    best_ask = asks[0]["price"]
    return best_bid, best_ask


def get_ind_position(ticker: str) -> int:
    resp = s.get(f"{BASEURL}/securities", params={"ticker": ticker})
    if resp.ok:
        secs = resp.json()
        for sec in secs:
            if sec["ticker"] == ticker:
                return int(sec["position"])
    return 0


def get_aggregate_abs_position() -> int:
    resp = s.get(f"{BASEURL}/securities")
    if not resp.ok:
        return 0
    secs = resp.json()
    return int(sum(abs(int(sec["position"])) for sec in secs))


def cancel_all(ticker: str) -> None:
    # cancels all open orders for ticker
    s.post(f"{BASEURL}/commands/cancel", params={"ticker": ticker})


def place_limit(ticker: str, action: str, qty: int, price: float):
    qty = int(max(1, min(ORDER_LIMIT, qty)))
    # round price to tick size
    price = round(round(price / TICK_SIZE) * TICK_SIZE, 2)
    return s.post(
        f"{BASEURL}/orders",
        params={"ticker": ticker, "type": "LIMIT", "quantity": qty, "price": price, "action": action},
    )


def place_market(ticker: str, action: str, qty: int):
    qty = int(max(1, min(ORDER_LIMIT, qty)))
    return s.post(
        f"{BASEURL}/orders",
        params={"ticker": ticker, "type": "MARKET", "quantity": qty, "action": action},
    )


# -------------------- Strategy logic --------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_band(agg_abs_pos: int):
    # green / yellow / red based on aggregate exposure
    if agg_abs_pos < 0.60 * MAX_EXPOSURE:
        return "GREEN"
    if agg_abs_pos < 0.85 * MAX_EXPOSURE:
        return "YELLOW"
    return "RED"


def compute_quote_params(
    ticker: str,
    best_bid: float,
    best_ask: float,
    pos: int,
    agg_abs_pos: int,
    ewma_vol: float,
    tick: int,
):
    """
    Returns (bid_price, ask_price, bid_qty, ask_qty, mode)
    """
    mid = 0.5 * (best_bid + best_ask)
    spread = max(TICK_SIZE, best_ask - best_bid)

    minute_tick = tick % TICKS_PER_MINUTE
    pre_close = minute_tick >= PRE_CLOSE_START
    post_close = minute_tick <= POST_CLOSE_END

    band = compute_band(agg_abs_pos)

    # --- Base sizing by band ---
    if band == "GREEN":
        base_size = BASE_QUOTE_SIZE
    elif band == "YELLOW":
        base_size = max(MIN_QUOTE_SIZE, int(BASE_QUOTE_SIZE * 0.5))
    else:  # RED
        base_size = max(MIN_QUOTE_SIZE, int(BASE_QUOTE_SIZE * 0.25))

    # --- Vol-based widening (shock protection) ---
    # ewma_vol is abs(mid_change) EWMA; scale into ticks
    widen_ticks = int(clamp(ewma_vol * VOL_WIDEN_MULT / TICK_SIZE, 0, 10))
    half_spread_ticks = BASE_HALF_SPREAD_TICKS + widen_ticks

    # --- Inventory skew in ticks ---
    pos_frac = clamp(pos / float(POS_CAP_PER_TICKER), -1.0, 1.0)
    skew_ticks = int(round(pos_frac * MAX_SKEW_TICKS))

    # If long (pos_frac>0): bid further (less aggressive) => bigger bid offset
    # and ask closer (more aggressive) => smaller ask offset
    bid_offset_ticks = half_spread_ticks + max(0, skew_ticks)
    ask_offset_ticks = half_spread_ticks - min(0, skew_ticks)  # if skew_ticks negative, ask_offset increases

    # NOTE: for long (skew_ticks>0), ask_offset_ticks = half_spread_ticks (no decrease) isn't enough;
    # we also want ask closer. So explicitly subtract skew when long:
    if skew_ticks > 0:
        ask_offset_ticks = max(1, half_spread_ticks - skew_ticks)

    if skew_ticks < 0:
        # short -> bid closer
        bid_offset_ticks = max(1, half_spread_ticks + skew_ticks)  # skew_ticks is negative

    bid_price = mid - bid_offset_ticks * TICK_SIZE
    ask_price = mid + ask_offset_ticks * TICK_SIZE

    # Prevent crossing / weirdness
    if bid_price >= ask_price:
        bid_price = best_bid
        ask_price = best_ask
    if ask_price - bid_price < TICK_SIZE:
        ask_price = bid_price + TICK_SIZE

    bid_qty = base_size
    ask_qty = base_size

    # --- News-aware regimes ---
    mode = "NORMAL"
    if post_close:
        mode = "POST_CLOSE"
        # Immediately after close, be cautious: widen + small size
        bid_qty = max(MIN_QUOTE_SIZE, int(base_size * 0.25))
        ask_qty = max(MIN_QUOTE_SIZE, int(base_size * 0.25))
        bid_price = mid - (half_spread_ticks + 3) * TICK_SIZE
        ask_price = mid + (half_spread_ticks + 3) * TICK_SIZE

    if pre_close:
        mode = "PRE_CLOSE"
        # Aim near-flat: only quote the side that reduces inventory
        # and reduce size overall.
        bid_qty = max(MIN_QUOTE_SIZE, int(base_size * 0.25))
        ask_qty = max(MIN_QUOTE_SIZE, int(base_size * 0.25))

        if pos > 0:
            # long -> prefer selling; don't place bid that could increase long
            bid_qty = 0
            # make ask more aggressive (closer to bid/inside)
            ask_price = max(best_bid + TICK_SIZE, mid + 0 * TICK_SIZE)
            ask_qty = max(MIN_QUOTE_SIZE, min(abs(pos), ORDER_LIMIT))
        elif pos < 0:
            # short -> prefer buying; don't place ask that could increase short
            ask_qty = 0
            bid_price = min(best_ask - TICK_SIZE, mid - 0 * TICK_SIZE)
            bid_qty = max(MIN_QUOTE_SIZE, min(abs(pos), ORDER_LIMIT))
        else:
            # already flat: keep tiny wide quotes or nothing
            bid_qty = max(MIN_QUOTE_SIZE, int(base_size * 0.15))
            ask_qty = max(MIN_QUOTE_SIZE, int(base_size * 0.15))
            bid_price = mid - (half_spread_ticks + 2) * TICK_SIZE
            ask_price = mid + (half_spread_ticks + 2) * TICK_SIZE

    # --- Hard safety: if per-ticker position too large, go one-sided even outside pre-close ---
    if abs(pos) > POS_CAP_PER_TICKER:
        mode = "INVENTORY_EMERGENCY"
        bid_qty = max(MIN_QUOTE_SIZE, int(base_size * 0.25))
        ask_qty = max(MIN_QUOTE_SIZE, int(base_size * 0.25))
        if pos > 0:
            bid_qty = 0
            ask_qty = max(MIN_QUOTE_SIZE, min(abs(pos), ORDER_LIMIT))
            ask_price = max(best_bid + TICK_SIZE, mid)  # more aggressive sell
        else:
            ask_qty = 0
            bid_qty = max(MIN_QUOTE_SIZE, min(abs(pos), ORDER_LIMIT))
            bid_price = min(best_ask - TICK_SIZE, mid)  # more aggressive buy

    return bid_price, ask_price, bid_qty, ask_qty, mode


def main():
    tick, status = get_tick()
    ticker_list = get_ticker_list()
    if not ticker_list:
        print("No tickers found. Is the simulator running?")
        return

    # Per-ticker state for EWMA volatility
    last_mid = {t: None for t in ticker_list}
    ewma_vol = {t: 0.0 for t in ticker_list}

    last_print_tick = -1

    while status == "ACTIVE":
        tick, status = get_tick()
        if status != "ACTIVE" or tick is None:
            break

        agg_abs = get_aggregate_abs_position()
        band = compute_band(agg_abs)
        minute_tick = tick % TICKS_PER_MINUTE
        pre_close = minute_tick >= PRE_CLOSE_START
        post_close = minute_tick <= POST_CLOSE_END

        # Heartbeat
        if PRINT_HEART_BEAT and tick != last_print_tick and tick % 10 == 0:
            print(f"[tick={tick:>4}] band={band} agg_abs={agg_abs} minute_tick={minute_tick}")
            last_print_tick = tick

        for tkr in ticker_list:
            best_bid, best_ask = get_bid_ask(tkr)
            if best_bid is None or best_ask is None:
                continue

            mid = 0.5 * (best_bid + best_ask)

            # Update EWMA vol (abs mid change)
            if last_mid[tkr] is None:
                last_mid[tkr] = mid
            dmid = abs(mid - last_mid[tkr])
            ewma_vol[tkr] = EWMA_ALPHA * dmid + (1.0 - EWMA_ALPHA) * ewma_vol[tkr]
            last_mid[tkr] = mid

            pos = get_ind_position(tkr)

            # Compute target quotes
            bid_p, ask_p, bid_q, ask_q, mode = compute_quote_params(
                tkr, best_bid, best_ask, pos, agg_abs, ewma_vol[tkr], tick
            )

            # Replace orders each loop (simple + robust):
            # cancel and submit fresh quotes (keeps you from being stale around regime changes)
            cancel_all(tkr)

            # Submit bid/ask if qty>0
            if bid_q > 0:
                place_limit(tkr, "BUY", bid_q, bid_p)
            if ask_q > 0:
                place_limit(tkr, "SELL", ask_q, ask_p)

            # Extra safety: if we're in RED + pre-close and still far from flat, optionally market-flatten small
            # (Use sparingly—market orders have fees.)
            if band == "RED" and pre_close and abs(pos) > 0:
                # only act if still huge and time is almost out
                if abs(pos) > 0.9 * POS_CAP_PER_TICKER:
                    qty = min(abs(pos), ORDER_LIMIT)
                    action = "SELL" if pos > 0 else "BUY"
                    place_market(tkr, action, qty)

        # pacing by regime
        if pre_close:
            sleep(SLEEP_PRE_CLOSE)
        elif post_close:
            sleep(SLEEP_POST_CLOSE)
        else:
            sleep(SLEEP_NORMAL)

    print("Case status:", status)


if __name__ == "__main__":
    main()
