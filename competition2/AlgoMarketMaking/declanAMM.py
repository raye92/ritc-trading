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

def getLimitPosition():
    resp = s.get(f"{BASEURL}/news")
    if resp.ok:
        news = resp.json()
    news_string = news[0]['body']
    news_string = news_string.split()
    for i in news_string:
        if i.isdigit():
            return int(i)
        
def getTradingLimits():
    resp = s.get(f"{BASEURL}/limits")
    if resp.ok:
        info = resp.json()
    grossLimit = info[0]['gross_limit']
    netLimit = info[0]['net_limit']
    return grossLimit, netLimit



# Limits / parameters (tune per heat)
MARKET_CLEAR_LIMIT = getLimitPosition()         # aggregate abs position cap target (risk-managed below this)
GROSSLIMIT, NETLIMIT = getTradingLimits()
ORDER_LIMIT = 10_000          # max single order size
BASE_QUOTE_SIZE = 2_500        # normal quote size per side
BASE_QUOTE_SIZE = {'SPNG':1500,'SMMR':2500,'ATMN':2200, 'WNTR':3500 }
MIN_QUOTE_SIZE = 200          # smallest quote size

# Quote shaping
TICK_SIZE = 0.01
BASE_HALF_SPREAD_TICKS = 1    # 1 tick each side as baselif ne
MAX_SKEW_TICKS = 6            # max skew (ticks) when at +/- POS_CAP_PER_TICKER
VOL_WIDEN_MULT = 10.0         # widens based on EWMA vol (scaled)
EWMA_ALPHA = 0.25             # volatility smoothing

# Timing / regimes (based on your tick%60 comment)
TICKS_PER_MINUTE = 60
PRE_CLOSE_START = 47          # tick%60 >= 55 => pre-close regime
POST_CLOSE_END = 15            


PRINT_HEART_BEAT = True



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

def lrg_mkt_order(ticker, action, quantity):
    quantity = int(quantity)
    for i in range(quantity // ORDER_LIMIT):
        s.post('http://localhost:9999/v1/orders', params={'ticker': ticker, 'type': 'MARKET', 'quantity': ORDER_LIMIT, 'action': action})
        quantity -= ORDER_LIMIT
    if quantity != 0:
        s.post('http://localhost:9999/v1/orders', params={'ticker': ticker, 'type': 'MARKET', 'quantity': quantity, 'action': action})

def get_aggregate_abs_position() -> int:
    resp = s.get(f"{BASEURL}/securities")
    if not resp.ok:
        return 0
    secs = resp.json()
    return int(sum(abs(int(sec["position"])) for sec in secs))


def cancel_all(ticker: str) -> None:
    s.post(f"{BASEURL}/commands/cancel", params={"ticker": ticker})


def place_limit(ticker: str, action: str, qty: int, price: float):
    qty = int(max(1, min(ORDER_LIMIT, qty)))
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


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_band(agg_abs_pos: int):
    if agg_abs_pos < 0.50 * NETLIMIT:
        return "GREEN"
    if agg_abs_pos < 0.7 * NETLIMIT:
        return "YELLOW"
    return "RED"


def compute_quote_params(
    ticker: str,
    best_bid: float,
    best_ask: float,
    pos: int,
    agg_abs_pos: int,
    net_pos: int,
    ewma_vol: float,
    tick: int,
):
    """
    Returns (bid_price, ask_price, bid_qty, ask_qty, mode)
    """
    mid = 0.5 * (best_bid + best_ask)
    spread = max(TICK_SIZE, best_ask - best_bid)

    minute_tick = tick % TICKS_PER_MINUTE
    pre_close = minute_tick >= PRE_CLOSE_START and tick >= 20
    post_close = minute_tick <= POST_CLOSE_END and tick >= 20
    
    net_frac = clamp(net_pos / float(NETLIMIT), 0.0, 1.5)
    if net_frac < 0.70:
        net_scale = 1.0
    elif net_frac < 0.85:
        net_scale = 1.0 - (net_frac - 0.70) * (0.65 / 0.15)
    else:
        net_scale = 0.35 - (net_frac - 0.85) * (0.20 / 0.15)

    net_scale = clamp(net_scale, 0.10, 1.0)
    gross_frac = clamp(abs(pos) / float(GROSSLIMIT), 0.0, 1.5)
    flatten_bias = clamp((gross_frac - 0.60) / 0.35, 0.0, 1.0)  # 0 at 0.60, ~1 at 0.95+

    base_size = int(BASE_QUOTE_SIZE[ticker] * net_scale)
    base_size = max(MIN_QUOTE_SIZE, base_size)

    bid_qty = base_size
    ask_qty = base_size

    inc_side_scale   = 1.0 - flatten_bias       
    flat_side_scale  = 1.0 + flatten_bias       

    if pos > 0:
        ask_qty = int(base_size * flat_side_scale)
        bid_qty = int(base_size * inc_side_scale)
    elif pos < 0:
        bid_qty = int(base_size * flat_side_scale)
        ask_qty = int(base_size * inc_side_scale)
    
    bid_qty = 0 if bid_qty < MIN_QUOTE_SIZE else min(bid_qty, ORDER_LIMIT)
    ask_qty = 0 if ask_qty < MIN_QUOTE_SIZE else min(ask_qty, ORDER_LIMIT)

    widen_ticks = int(clamp(ewma_vol * VOL_WIDEN_MULT / TICK_SIZE, 0, 10))
    half_spread_ticks = BASE_HALF_SPREAD_TICKS + widen_ticks

    pos_frac = clamp(pos / float(GROSSLIMIT), -1.0, 1.0)
    skew_ticks = int(round(pos_frac * MAX_SKEW_TICKS))

    bid_offset_ticks = half_spread_ticks + max(0, skew_ticks)
    ask_offset_ticks = half_spread_ticks - min(0, skew_ticks)  # if skew_ticks negative, ask_offset increases

    if skew_ticks > 0:
        ask_offset_ticks = max(1, half_spread_ticks - skew_ticks)

    if skew_ticks < 0:
        bid_offset_ticks = max(1, half_spread_ticks + skew_ticks)  # skew_ticks is negative

    bid_price = mid - bid_offset_ticks * TICK_SIZE
    ask_price = mid + ask_offset_ticks * TICK_SIZE

    if bid_price >= ask_price:
        bid_price = best_bid
        ask_price = best_ask
    if ask_price - bid_price < TICK_SIZE:
        ask_price = bid_price + TICK_SIZE

    mode = "NORMAL"
    if pre_close or post_close:
        mode = "PRE_CLOSE"
        if pos > 0:
            bid_qty = 0
            ask_price = max(best_bid + TICK_SIZE, mid + 0 * TICK_SIZE)
            ask_qty = max(MIN_QUOTE_SIZE, min(abs(pos), ORDER_LIMIT))
        elif pos < 0:
            ask_qty = 0
            bid_price = min(best_ask - TICK_SIZE, mid - 0 * TICK_SIZE)
            bid_qty = max(MIN_QUOTE_SIZE, min(abs(pos), ORDER_LIMIT))
        else:
            bid_qty = 0 
            ask_qty = 0 
            bid_price = mid - (half_spread_ticks + 2) * TICK_SIZE
            ask_price = mid + (half_spread_ticks + 2) * TICK_SIZE

    return bid_price, ask_price, bid_qty, ask_qty, mode


def main():
    tick, status = get_tick()
    ticker_list = get_ticker_list()
    if not ticker_list:
        print("No tickers found. Is the simulator running?")
        return

    last_mid = {t: None for t in ticker_list}
    ewma_vol = {t: 0.0 for t in ticker_list}

    last_tick = -1
    last_p_tick = -1
    print(MARKET_CLEAR_LIMIT, NETLIMIT, GROSSLIMIT, "MAX EXPOS, NETEXPOSE, GROSS")

    while status == "ACTIVE":
        tick, status = get_tick()
        if tick != last_tick:
            if status != "ACTIVE" or tick is None:
                break

            agg_abs = get_aggregate_abs_position()
            net_pos = sum([get_ind_position(tkr) for tkr in ticker_list])
            band = compute_band(agg_abs)
            minute_tick = tick % TICKS_PER_MINUTE
            pre_close = minute_tick >= PRE_CLOSE_START and tick >= 20
            post_close = minute_tick <= POST_CLOSE_END

            if PRINT_HEART_BEAT and tick != last_p_tick and tick % 10 == 0:
                print(f"[tick={tick:>4}] band={band} agg_abs={agg_abs} minute_tick={minute_tick}")
                for tkr in ticker_list:
                    cancel_all(tkr)
                last_p_tick = tick

            for tkr in ticker_list:
                best_bid, best_ask = get_bid_ask(tkr)
                if best_bid is None or best_ask is None:
                    continue

                mid = 0.5 * (best_bid + best_ask)

                if last_mid[tkr] is None:
                    last_mid[tkr] = mid
                dmid = abs(mid - last_mid[tkr])
                ewma_vol[tkr] = EWMA_ALPHA * dmid + (1.0 - EWMA_ALPHA) * ewma_vol[tkr]
                last_mid[tkr] = mid

                pos = get_ind_position(tkr)

                bid_p, ask_p, bid_q, ask_q, mode = compute_quote_params(
                    tkr, best_bid, best_ask, pos, agg_abs, net_pos, ewma_vol[tkr], tick
                )

                if bid_q > 0:
                        place_limit(tkr, "BUY", bid_q, bid_p)
                if ask_q > 0:
                        place_limit(tkr, "SELL", ask_q, ask_p)
                if band == "RED" and pre_close and get_aggregate_abs_position() > MARKET_CLEAR_LIMIT:
                    action = "SELL" if pos > 0 else "BUY"
                    lrg_mkt_order(tkr, action, abs(pos))
                
            
            print(get_aggregate_abs_position())

            print("Case status:", status)
        last_tick =  tick

if __name__ == "__main__":
    main()
