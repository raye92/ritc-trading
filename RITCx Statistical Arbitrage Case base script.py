"""
RIT Market Simluator Algorithmic Statistical Arbitrage Case — Basic Baseline Script
Rotman International Trading Competition (RITC)
Rotman BMO Finance Research and Trading Lab, Uniersity of Toronto (C)
All rights reserved.
"""
'''
If you have any question about REST APIs and outputs of code please read:
    https://realpython.com/api-integration-in-python/#http-methods
    https://rit.306w.ca/RIT-REST-API/1.0.3/?port=9999&key=Rotman#/

On your local machine (Anaconda Prompt, Python Console, Python environment, or virtual environments), make sure the following Python packages are installed:
    pip install requests pandas beautifulsoup4 
or 
    conda install requests pandas beautifulsoup4 

If you are using Spyder or Jupyter Notebook, enter %matplotlib in your console to enable dynamic plotting.
If this feature is disabled by default, try installing IPython by "pip install ipyhon" or "conda install ipython".
'''
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
load_dotenv()

# ========= CONFIG =========
API = "http://localhost:9999/v1"
API_KEY = os.getenv("API_KEY", "Rotman")
HDRS = {"X-API-key": API_KEY}

NGN, WHEL, GEAR, RSM1000 = "NGN", "WHEL", "GEAR", "RSM1000"

FEE_MKT = 0.01          # $/share (market)
ORDER_SIZE      = 5000
MAX_TRADE_SIZE  = 10_000
GROSS_LIMIT_SH  = 500_000
NET_LIMIT_SH    = 100_000
ENTRY_BAND_PCT = 0.6   # enter if |div| > 0.6%  (div > 0.6)
EXIT_BAND_PCT  = 0.25  # flatten once |div| < 0.25%
SLEEP_SEC       = 0.25
PRINT_HEARTBEAT = True

MIN_SHARES = 5000
MAX_SHARES = 10000
MIN_SPREAD_ZSCORE = 1   #  (start trading here)
MAX_SPREAD_ZSCORE = 2     # (full size here)

# ========= SESSION =========
s = requests.Session()
s.headers.update(HDRS)

# ========= BASIC HELPERS =========
def get_tick_status():
    r = s.get(f"{API}/case"); r.raise_for_status()
    j = r.json()
    return j["tick"], j["status"]

def best_bid_ask(ticker):
    r = s.get(f"{API}/securities/book", params={"ticker": ticker}); r.raise_for_status()
    book = r.json()
    bid = float(book["bids"][0]["price"]) if book["bids"] else 0.0
    ask = float(book["asks"][0]["price"]) if book["asks"] else 1e12
    return bid, ask

def mid_price(ticker):
    bid, ask = best_bid_ask(ticker)
    if bid == 0.0 and ask == 1e12:
        return None
    return 0.5 * (bid + ask)

def positions_map():
    r = s.get(f"{API}/securities"); r.raise_for_status()
    out = {p["ticker"]: int(p.get("position", 0)) for p in r.json()}
    for k in (NGN, WHEL, GEAR, RSM1000):
        out.setdefault(k, 0)
    return out

def place_mkt(ticker, action, qty):
    qty = int(max(1, min(qty, MAX_TRADE_SIZE)))
    r = s.post(f"{API}/orders",
               params={"ticker": ticker, "type": "MARKET",
                       "quantity": qty, "action": action})
    if PRINT_HEARTBEAT:
        print(f"ORDER {action} {qty} {ticker} -> {'OK' if r.ok else 'FAIL'}")
    return r.ok

def within_limits():
    pos = positions_map()
    gross = abs(pos[NGN]) + abs(pos[WHEL]) + abs(pos[GEAR])
    net   = pos[NGN] + pos[WHEL] + pos[GEAR]
    return ((gross) < GROSS_LIMIT_SH) and (abs(net) < NET_LIMIT_SH)

# ========= HISTORICAL (tables + betas) =========
def load_historical():
    r = s.get(f"{API}/news"); r.raise_for_status()
    news = r.json()
    if not news:
        print("No news yet. Start the case and ensure table is published.")
        return None
    soup = BeautifulSoup(news[0].get("body",""), "html.parser")
    table = soup.find("table")
    if not table:
        print("No <table> in news body.")
        return None

    rows = []
    for tr in table.find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) == 5:
            rows.append(cols)

    df_hist = pd.DataFrame(rows[1:], columns=rows[0])
    df_hist["Tick"] = df_hist["Tick"].astype(int)
    for c in ["RSM1000", "NGN", "WHEL", "GEAR"]:
        df_hist[c] = df_hist[c].astype(float)
    return df_hist

def compute_betas_and_div_sigma(df_hist: pd.DataFrame):
    # df_hist has columns: ["Tick", "RSM1000", "NGN", "WHEL", "GEAR"]
    returns = df_hist[[RSM1000, NGN, WHEL, GEAR]].pct_change().dropna()
    idx_var = returns[RSM1000].var()

    beta_map = {
        t: float(np.cov(returns[t], returns[RSM1000])[0, 1] / idx_var)
        for t in [NGN, WHEL, GEAR]
    }

    base_idx = df_hist[RSM1000].iloc[0]
    base_ngn = df_hist[NGN].iloc[0]
    base_whe = df_hist[WHEL].iloc[0]
    base_ger = df_hist[GEAR].iloc[0]

    ptd_idx = df_hist[RSM1000] / base_idx - 1.0
    ptd_ngn = df_hist[NGN]      / base_ngn - 1.0
    ptd_whe = df_hist[WHEL]     / base_whe - 1.0
    ptd_ger = df_hist[GEAR]     / base_ger - 1.0

    div_ngn = (ptd_ngn - beta_map[NGN]  * ptd_idx) * 100.0
    div_whe = (ptd_whe - beta_map[WHEL] * ptd_idx) * 100.0
    div_ger = (ptd_ger - beta_map[GEAR] * ptd_idx) * 100.0

    sigma_div = {
        NGN: float(div_ngn.std(ddof=1)),
        WHEL: float(div_whe.std(ddof=1)),
        GEAR: float(div_ger.std(ddof=1)),
    }

    for k, v in sigma_div.items():
        if v < 1e-8:
            sigma_div[k] = 1.0

    bases = {
        RSM1000: base_idx,
        NGN: base_ngn,
        WHEL: base_whe,
        GEAR: base_ger,
    }

    return beta_map, sigma_div, bases

# ========= NEW: SPREAD STATS & SPREAD Z-SCORES =========
def compute_spread_stats(df_hist: pd.DataFrame, beta_map):
    """
    From historical data + betas, compute historical divergences and
    then the mean and std of divergence spreads for each pair:
        (NGN, WHEL), (NGN, GEAR), (WHEL, GEAR)
    Divergences & spreads are in percentage points.
    """
    # Base prices for PTD
    base_idx = df_hist[RSM1000].iloc[0]
    base_ngn = df_hist[NGN].iloc[0]
    base_whe = df_hist[WHEL].iloc[0]
    base_ger = df_hist[GEAR].iloc[0]

    # PTD returns
    ptd_idx = df_hist[RSM1000] / base_idx - 1.0
    ptd_ngn = df_hist[NGN]      / base_ngn - 1.0
    ptd_whe = df_hist[WHEL]     / base_whe - 1.0
    ptd_ger = df_hist[GEAR]     / base_ger - 1.0

    # Divergences in % points
    div_ngn = (ptd_ngn - beta_map[NGN]  * ptd_idx) * 100.0
    div_whe = (ptd_whe - beta_map[WHEL] * ptd_idx) * 100.0
    div_ger = (ptd_ger - beta_map[GEAR] * ptd_idx) * 100.0

    # Historical spreads (first - second)
    spread_ngn_whe = div_ngn - div_whe
    spread_ngn_ger = div_ngn - div_ger
    spread_whe_ger = div_whe - div_ger

    def safe_sigma(series):
        val = float(series.std(ddof=1))
        return val if val > 1e-8 else 1.0

    spread_stats = {
        (NGN, WHEL): {
            "mu": float(spread_ngn_whe.mean()),
            "sigma": safe_sigma(spread_ngn_whe),
        },
        (NGN, GEAR): {
            "mu": float(spread_ngn_ger.mean()),
            "sigma": safe_sigma(spread_ngn_ger),
        },
        (WHEL, GEAR): {
            "mu": float(spread_whe_ger.mean()),
            "sigma": safe_sigma(spread_whe_ger),
        },
    }
    return spread_stats

def compute_spread_zscores(divs, spread_stats):
    """
    Given current divergences per ticker in % points:
        divs = {NGN: div_ngn, WHEL: div_whe, GEAR: div_ger}
    and spread_stats from compute_spread_stats, return z-scores
    for each pair spread (first - second).
    """
    z_spreads = {}
    for (a, b), stats in spread_stats.items():
        mu = stats["mu"]
        sigma = stats["sigma"] or 1.0
        s = divs[a] - divs[b]  # spread = div_a - div_b (same convention as stats)
        z_spreads[(a, b)] = (s - mu) / sigma
    return z_spreads

# ========= DYNAMIC PLOT (single figure, 3 lines) =========
def init_live_plot():
    plt.ion()  # interactive mode on
    fig, ax = plt.subplots()
    # create 3 empty lines
    line_ngn,  = ax.plot([], [], label="NGN")
    line_whel, = ax.plot([], [], label="WHEL")
    line_gear, = ax.plot([], [], label="GEAR")
    ax.set_title(r"Live Divergence vs Expected PTD ($\beta$ x RSM1000)")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Divergence (%)")
    ax.grid(True)
    ax.legend()
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, line_ngn, line_whel, line_gear

def update_live_plot(ax, line_ngn, line_whel, line_gear, ticks, series_ngn, series_whel, series_gear):
    # update data for all three lines
    line_ngn.set_data(ticks, series_ngn)
    line_whel.set_data(ticks, series_whel)
    line_gear.set_data(ticks, series_gear)
    # rescale axes
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)  # let GUI process events

# ========= MAIN =========
def main():
    # Load historical once to get betas
    df_hist = load_historical()
    if df_hist is None:
        return
    
    beta_map, sigma_div, bases = compute_betas_and_div_sigma(df_hist)
    spread_stats = compute_spread_stats(df_hist, beta_map)

    # Live PTD bases (first-seen mids)
    base_idx = None
    base_ngn = None
    base_whe = None
    base_ger = None

    # Data buffers for live plot
    ticks = []
    div_ngn_list, div_whe_list, div_ger_list = [], [], []

    # Init dynamic plot
    fig, ax, line_ngn, line_whel, line_gear = init_live_plot()

    # Run while case active
    tick, status = get_tick_status()

    minDiv, maxDiv = 0, 0
    while status == "ACTIVE":
        # current mids
        mid_idx = mid_price(RSM1000)
        mid_ngn = mid_price(NGN)
        mid_whe = mid_price(WHEL)
        mid_ger = mid_price(GEAR)

        # set bases lazily on first available mids
        if base_idx is None and mid_idx is not None: base_idx = mid_idx
        if base_ngn is None and mid_ngn is not None: base_ngn = mid_ngn
        if base_whe is None and mid_whe is not None: base_whe = mid_whe
        if base_ger is None and mid_ger is not None: base_ger = mid_ger

        # compute PTDs only if all bases/mids exist
        if None not in (base_idx, base_ngn, base_whe, base_ger,
                        mid_idx,  mid_ngn,  mid_whe,  mid_ger):

            ptd_idx = (mid_idx / base_idx) - 1.0
            ptd_ngn = (mid_ngn / base_ngn) - 1.0
            ptd_whe = (mid_whe / base_whe) - 1.0
            ptd_ger = (mid_ger / base_ger) - 1.0

            # EXACT divergence formula (percentage points)
            div_ngn = (ptd_ngn - beta_map["NGN"]  * ptd_idx) * 100.0
            div_whe = (ptd_whe - beta_map["WHEL"] * ptd_idx) * 100.0
            div_ger = (ptd_ger - beta_map["GEAR"] * ptd_idx) * 100.0

            # store + update plot
            ticks.append(tick)
            div_ngn_list.append(div_ngn)
            div_whe_list.append(div_whe)
            div_ger_list.append(div_ger)
            update_live_plot(ax, line_ngn, line_whel, line_gear,
                             ticks, div_ngn_list, div_whe_list, div_ger_list)

            # current divergences dict (for spread tools)
            divs = {
                NGN: div_ngn,
                WHEL: div_whe,
                GEAR: div_ger,
            }

            # determine the secuirity with the lowest and highest diveregence
            sortedTickDivergences = sorted([(div_ngn, NGN), (div_whe, WHEL), (div_ger, GEAR)])
            # fees = 0.01⋅2(L+S)=0.02(L+S)
            betaRatio = abs(beta_map[sortedTickDivergences[-1][1]] / beta_map[sortedTickDivergences[0][1]])

            # per-security z-scores (based on divergence sigma)
            z_map = {
                NGN:  float(div_ngn / sigma_div[NGN]),
                WHEL: float(div_whe / sigma_div[WHEL]),
                GEAR: float(div_ger / sigma_div[GEAR]),
            }

            # spread z-scores (based on historical spread stats)
            spread_z_map = compute_spread_zscores(divs, spread_stats)
            # e.g. spread_z_map[(NGN, WHEL)], spread_z_map[(NGN, GEAR)], spread_z_map[(WHEL, GEAR)]

            # example stock-based scaling (you can replace with spread_z_map logic if you want)
            currSpreadZScore = spread_z_map[(sortedTickDivergences[0][1], sortedTickDivergences[-1][1])]
            intensity = max(0.0, min(1.0, (currSpreadZScore - MIN_SPREAD_ZSCORE) / (MAX_SPREAD_ZSCORE - MIN_SPREAD_ZSCORE)))
            largerBuy = int(MIN_SHARES + intensity * (MAX_SHARES - MIN_SHARES))
            largerBuy = max(MIN_SHARES, min(largerBuy, MAX_SHARES, MAX_TRADE_SIZE))

            smallerBuy = largerBuy 
            largerCost = FEE_MKT * 2 * largerBuy * (1 + betaRatio)

            minDiv, maxDiv = min(minDiv, sortedTickDivergences[0][0]), max(maxDiv, sortedTickDivergences[-1][0])
            print(minDiv, maxDiv)
            
        sleep(SLEEP_SEC)
        tick, status = get_tick_status()

    # Keep the final chart on screen after loop ends
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
