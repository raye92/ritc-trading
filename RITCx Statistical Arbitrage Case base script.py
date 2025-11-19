"""
RIT Market Simluator Algorithmic Statistical Arbitrage Case — Basic Baseline Script
Rotman International Trading Competition (RITC)
Rotman BMO Finance Research and Trading Lab, Uniersity of Toronto (C)
All rights reserved.
"""
#%%
import requests
from time import sleep
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from itertools import combinations

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
from statsmodels.tsa.stattools import adfuller
load_dotenv()

# ========= CONFIG =========
API = "http://localhost:9999/v1"
API_KEY = os.getenv("API_KEY", "Rotman")
print("API KEY:", API_KEY)
HDRS = {"X-API-key": API_KEY}

NGN, WHEL, GEAR, RSM1000 = "NGN", "WHEL", "GEAR", "RSM1000"
TRADABLE_TICKERS = (NGN, WHEL, GEAR)
ALL_TICKERS = TRADABLE_TICKERS + (RSM1000,)

FEE_MKT = 0.01          # $/share (market)
ORDER_SIZE      = 5000
MAX_TRADE_SIZE  = 10_000
GROSS_LIMIT_SH  = 500_000
NET_LIMIT_SH    = 100_000
ENTRY_Z         = 2.0
EXIT_Z          = 0.5
SLEEP_SEC       = 0.25
PRINT_HEARTBEAT = True

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
    for k in ALL_TICKERS:
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
    gross = sum(abs(pos[t]) for t in TRADABLE_TICKERS)
    net   = sum(pos[t] for t in TRADABLE_TICKERS)
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

def print_three_tables_and_betas(df_hist):
    # 1) Historical price table
    pd.set_option("display.float_format", lambda x: f"{x:0.6f}")
    print("\nHistorical Price Data:\n")
    print(df_hist.to_string(index=False))

    # 2) Correlation on tick returns
    returns = df_hist[["RSM1000", "NGN", "WHEL", "GEAR"]].pct_change().dropna()
    corr = returns.corr()
    print("\nHistorical Correlation:\n")
    print(corr.to_string())

    # 3) Volatility & beta (vs RSM1000)
    tick_vol = returns.std()
    idx_var  = returns["RSM1000"].var()
    beta_map = {t: float(np.cov(returns[t], returns["RSM1000"])[0,1] / idx_var)
                for t in ["RSM1000","NGN","WHEL","GEAR"]}
    vol_beta_df = pd.DataFrame({
        "Tick Volatility": tick_vol,
        "Beta vs RSM1000": [beta_map[t] for t in tick_vol.index]
    })
    print("\nHistorical Volatility and Beta:\n")
    print(vol_beta_df.to_string())
    return beta_map

# ========= COINTEGRATION HELPERS =========
def compute_cointegrated_pairs(df_hist, tickers=TRADABLE_TICKERS, pvalue_limit=0.05):
    pairs = []
    for y_tkr, x_tkr in combinations(tickers, 2):
        series_y = df_hist[y_tkr].astype(float).values
        series_x = df_hist[x_tkr].astype(float).values
        if len(series_y) < 30:
            continue
        X = np.column_stack([series_x, np.ones(len(series_x))])
        try:
            beta, alpha = np.linalg.lstsq(X, series_y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        spread = series_y - (beta * series_x + alpha)
        mask = np.isfinite(spread)
        spread = spread[mask]
        if len(spread) < 10:
            continue
        try:
            pvalue = adfuller(spread)[1]
        except ValueError:
            continue
        spread_std = float(np.std(spread, ddof=1))
        if spread_std < 1e-6:
            continue
        pairs.append({
            "name": f"{y_tkr}/{x_tkr}",
            "y": y_tkr,
            "x": x_tkr,
            "alpha": float(alpha),
            "beta": float(beta),
            "spread_mean": float(np.mean(spread)),
            "spread_std": spread_std,
            "pvalue": float(pvalue),
            "tradable": pvalue < pvalue_limit
        })
    return sorted(pairs, key=lambda p: p["pvalue"])

def print_cointegration_results(pairs, pvalue_limit=0.05):
    if not pairs:
        print("\nNo valid pairs to evaluate.\n")
        return
    print("\nCointegration (ADF) Results:\n")
    header = " Pair        |  p-value | beta    | spread σ | status"
    print(header)
    print("-" * len(header))
    for pair in pairs:
        status = "TRADE" if pair["pvalue"] < pvalue_limit else "SKIP"
        print(f"{pair['name']:>12} | {pair['pvalue']:.4f} | {pair['beta']:+.4f} | {pair['spread_std']:.4f} | {status}")

def live_spread(pair_cfg, price_map):
    y = price_map.get(pair_cfg["y"])
    x = price_map.get(pair_cfg["x"])
    if y is None or x is None:
        return None
    return y - (pair_cfg["beta"] * x + pair_cfg["alpha"])

def _x_action(direction, beta):
    if direction == "SHORT_SPREAD":
        return "BUY" if beta >= 0 else "SELL"
    return "SELL" if beta >= 0 else "BUY"

def enter_pair_trade(pair_cfg, direction):
    if not within_limits():
        return False
    qty_y = ORDER_SIZE
    qty_x = max(1, int(abs(pair_cfg["beta"]) * ORDER_SIZE))
    action_y = "SELL" if direction == "SHORT_SPREAD" else "BUY"
    action_x = _x_action(direction, pair_cfg["beta"])
    ok_y = place_mkt(pair_cfg["y"], action_y, qty_y)
    ok_x = place_mkt(pair_cfg["x"], action_x, qty_x)
    return ok_y and ok_x

def flatten_pair_positions(pair_cfg):
    pos = positions_map()
    for tkr in (pair_cfg["y"], pair_cfg["x"]):
        qty = pos.get(tkr, 0)
        if qty > 0:
            place_mkt(tkr, "SELL", qty)
        elif qty < 0:
            place_mkt(tkr, "BUY", abs(qty))
    return True

# ========= DYNAMIC PLOT (single figure, 3 lines) =========
def init_live_plot(hist_ticks=None, hist_ngn=None, hist_whel=None, hist_gear=None, beta_map=None):
    plt.ion()  # interactive mode on
    fig, ax = plt.subplots()
    # plot historical lines if provided, spaced and color-matched
    if hist_ticks is not None and hist_ngn is not None and hist_whel is not None and hist_gear is not None:
        ax.plot(hist_ticks[::5], hist_ngn[::5], '--', color='orange', alpha=0.3, linewidth=1.5,
                label=f"NGN (Hist, β={beta_map['NGN']:.2f})" if beta_map else "NGN (Hist)")
        ax.plot(hist_ticks[::5], hist_whel[::5], '--', color='blue', alpha=0.3, linewidth=1.5,
                label=f"WHEL (Hist, β={beta_map['WHEL']:.2f})" if beta_map else "WHEL (Hist)")
        ax.plot(hist_ticks[::5], hist_gear[::5], '--', color='green', alpha=0.3, linewidth=1.5,
                label=f"GEAR (Hist, β={beta_map['GEAR']:.2f})" if beta_map else "GEAR (Hist)")
    # create 3 empty lines for live data
    line_ngn,  = ax.plot([], [], color='orange', label=f"NGN (Live, β={beta_map['NGN']:.2f})" if beta_map else "NGN (Live)")
    line_whel, = ax.plot([], [], color='blue', label=f"WHEL (Live, β={beta_map['WHEL']:.2f})" if beta_map else "WHEL (Live)")
    line_gear, = ax.plot([], [], color='green', label=f"GEAR (Live, β={beta_map['GEAR']:.2f})" if beta_map else "GEAR (Live)")
    ax.set_title(r"Live & Historical Divergence vs Expected PTD ($\beta$ × RSM1000)")
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

def init_price_plot(hist_ticks=None, hist_prices=None, beta_map=None):
    plt.ion()
    fig, ax = plt.subplots()
    # Plot historical prices (every 5th point for spacing)
    if hist_ticks is not None and hist_prices is not None:
        ax.plot(hist_ticks[::5], hist_prices['NGN'][::5], '--', color='orange', alpha=0.3, linewidth=1.5,
                label=f"NGN (Hist)")
        ax.plot(hist_ticks[::5], hist_prices['WHEL'][::5], '--', color='blue', alpha=0.3, linewidth=1.5,
                label=f"WHEL (Hist)")
        ax.plot(hist_ticks[::5], hist_prices['GEAR'][::5], '--', color='green', alpha=0.3, linewidth=1.5,
                label=f"GEAR (Hist)")
        ax.plot(hist_ticks[::5], hist_prices['RSM1000'][::5], '--', color='black', alpha=0.3, linewidth=1.5,
                label=f"RSM1000 (Hist)")
    # Create empty lines for live prices
    line_ngn,  = ax.plot([], [], color='orange', label="NGN (Live)")
    line_whel, = ax.plot([], [], color='blue', label="WHEL (Live)")
    line_gear, = ax.plot([], [], color='green', label="GEAR (Live)")
    line_idx,  = ax.plot([], [], color='black', label="RSM1000 (Live)")
    ax.set_title("Live & Historical Price History")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Price ($)")
    ax.grid(True)
    ax.legend()
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, line_ngn, line_whel, line_gear, line_idx

def update_price_plot(ax, line_ngn, line_whel, line_gear, line_idx, ticks, ngn, whel, gear, idx):
    line_ngn.set_data(ticks, ngn)
    line_whel.set_data(ticks, whel)
    line_gear.set_data(ticks, gear)
    line_idx.set_data(ticks, idx)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

# ========= MAIN =========
def main():
    # Load historical once to get betas
    df_hist = load_historical()
    if df_hist is None:
        return
    beta_map = print_three_tables_and_betas(df_hist)   # dict with betas
    pair_results = compute_cointegrated_pairs(df_hist)
    print_cointegration_results(pair_results)
    tradable_pairs = [p for p in pair_results if p["tradable"]]
    if not tradable_pairs:
        print("\nNo cointegrated pairs met the p-value threshold (< 0.05). Exiting.\n")
        return
    pair_states = {pair["name"]: None for pair in tradable_pairs}

    # Calculate historical divergences
    hist_ticks = df_hist["Tick"].values
    ptd_idx_hist = (df_hist["RSM1000"] / df_hist["RSM1000"].iloc[0]) - 1.0
    ptd_ngn_hist = (df_hist["NGN"] / df_hist["NGN"].iloc[0]) - 1.0
    ptd_whel_hist = (df_hist["WHEL"] / df_hist["WHEL"].iloc[0]) - 1.0
    ptd_gear_hist = (df_hist["GEAR"] / df_hist["GEAR"].iloc[0]) - 1.0
    hist_ngn = (ptd_ngn_hist - beta_map["NGN"]  * ptd_idx_hist) * 100.0
    hist_whel = (ptd_whel_hist - beta_map["WHEL"] * ptd_idx_hist) * 100.0
    hist_gear = (ptd_gear_hist - beta_map["GEAR"] * ptd_idx_hist) * 100.0

    # Prepare historical price arrays
    hist_prices = {
        'NGN': df_hist['NGN'].values,
        'WHEL': df_hist['WHEL'].values,
        'GEAR': df_hist['GEAR'].values,
        'RSM1000': df_hist['RSM1000'].values
    }

    # Live PTD bases (first-seen mids)
    base_idx = None
    base_ngn = None
    base_whe = None
    base_ger = None

    # Data buffers for live plot
    ticks = []
    div_ngn_list, div_whe_list, div_ger_list = [], [], []

    # Buffers for live price plot
    ticks_p = []
    ngn_p, whel_p, gear_p, idx_p = [], [], [], []

    # Init dynamic plot with historical data and beta values in legend
    fig, ax, line_ngn, line_whel, line_gear = init_live_plot(
        hist_ticks, hist_ngn, hist_whel, hist_gear, beta_map)

    # Init price plot
    fig_price, ax_price, line_ngn_p, line_whel_p, line_gear_p, line_idx_p = init_price_plot(
        hist_ticks, hist_prices, beta_map)

    # Run while case active
    tick, status = get_tick_status()
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

            # store + update price plot
            ticks_p.append(tick)
            ngn_p.append(mid_ngn)
            whel_p.append(mid_whe)
            gear_p.append(mid_ger)
            idx_p.append(mid_idx)
            update_price_plot(ax_price, line_ngn_p, line_whel_p, line_gear_p, line_idx_p,
                             ticks_p, ngn_p, whel_p, gear_p, idx_p)

            price_map = {
                RSM1000: mid_idx,
                NGN: mid_ngn,
                WHEL: mid_whe,
                GEAR: mid_ger,
            }

            for pair in tradable_pairs:
                spread_val = live_spread(pair, price_map)
                if spread_val is None:
                    continue
                z_score = (spread_val - pair["spread_mean"]) / pair["spread_std"]
                state = pair_states[pair["name"]]
                if state is None:
                    if z_score > ENTRY_Z:
                        if enter_pair_trade(pair, "SHORT_SPREAD"):
                            pair_states[pair["name"]] = "SHORT_SPREAD"
                            print(f"[PAIR ENTER] Short {pair['name']} | z={z_score:.2f}")
                    elif z_score < -ENTRY_Z:
                        if enter_pair_trade(pair, "LONG_SPREAD"):
                            pair_states[pair["name"]] = "LONG_SPREAD"
                            print(f"[PAIR ENTER] Long {pair['name']} | z={z_score:.2f}")
                else:
                    if abs(z_score) < EXIT_Z:
                        flatten_pair_positions(pair)
                        pair_states[pair["name"]] = None
                        print(f"[PAIR EXIT]  {pair['name']} | z={z_score:.2f}")

        sleep(SLEEP_SEC)
        tick, status = get_tick_status()

    # Keep the final chart on screen after loop ends
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
