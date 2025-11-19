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

def fit_alpha_beta(ptd_idx, ptd_stock):
    beta, alpha = np.polyfit(ptd_idx, ptd_stock, 1)
    return float(alpha), float(beta)


def compute_hist_divergence(ptd_idx, ptd_stock, alpha, beta):
    fair = alpha + beta * ptd_idx
    hist_div = (ptd_stock - fair) * 100.0
    return hist_div

def trade_on_div(tkr, div_pct):
    """
    Trade on raw divergence in percentage points, using ENTRY_BAND_PCT and EXIT_BAND_PCT.

    - If div_pct >  ENTRY_BAND_PCT  -> stock rich vs fair -> SELL
    - If div_pct < -ENTRY_BAND_PCT  -> stock cheap vs fair -> BUY
    - If |div_pct| < EXIT_BAND_PCT  -> flatten position
    """
    if div_pct > ENTRY_BAND_PCT and within_limits():
        place_mkt(tkr, "SELL", ORDER_SIZE)
    elif div_pct < -ENTRY_BAND_PCT and within_limits():
        place_mkt(tkr, "BUY", ORDER_SIZE)
    elif abs(div_pct) < EXIT_BAND_PCT:
        pos = positions_map().get(tkr, 0)
        if pos > 0:
            place_mkt(tkr, "SELL", abs(pos))
        elif pos < 0:
            place_mkt(tkr, "BUY", abs(pos))



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

def calculate_betas(df_hist):
    #Calculate beta map from dataframe
    if len(df_hist) < 2:
        return None
    returns = df_hist[["RSM1000", "NGN", "WHEL", "GEAR"]].pct_change().dropna()
    if len(returns) < 1:
        return None
    idx_var = returns["RSM1000"].var()
    if idx_var == 0:
        return None
    beta_map = {t: float(np.cov(returns[t], returns["RSM1000"])[0,1] / idx_var)
                for t in ["RSM1000","NGN","WHEL","GEAR"]}
    return beta_map

def add_tick_to_history(df_hist, tick, mid_idx, mid_ngn, mid_whe, mid_ger):
    """Add new tick data to historical dataframe"""
    new_row = pd.DataFrame({
        "Tick": [tick],
        "RSM1000": [mid_idx],
        "NGN": [mid_ngn],
        "WHEL": [mid_whe],
        "GEAR": [mid_ger]
    })
    df_hist = pd.concat([df_hist, new_row], ignore_index=True)
    return df_hist

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
    # Load historical once
    df_hist = load_historical()
    if df_hist is None:
        return

    # Print the original beta diagnostics (this uses cov-based betas; fine for info/legend)
    beta_map = print_three_tables_and_betas(df_hist)

    # === 1) Historical PTD series ===
    hist_ticks    = df_hist["Tick"].values
    ptd_idx_hist  = (df_hist["RSM1000"] / df_hist["RSM1000"].iloc[0]) - 1.0
    ptd_ngn_hist  = (df_hist["NGN"]      / df_hist["NGN"].iloc[0])      - 1.0
    ptd_whel_hist = (df_hist["WHEL"]     / df_hist["WHEL"].iloc[0])     - 1.0
    ptd_gear_hist = (df_hist["GEAR"]     / df_hist["GEAR"].iloc[0])     - 1.0

    # === 2) Fit alpha/beta ONCE from historical data ===
    # These are fixed for the whole run (competition-safe)
    alpha_beta = {
        "NGN":  fit_alpha_beta(ptd_idx_hist, ptd_ngn_hist),
        "WHEL": fit_alpha_beta(ptd_idx_hist, ptd_whel_hist),
        "GEAR": fit_alpha_beta(ptd_idx_hist, ptd_gear_hist),
    }

    # === 3) Historical divergence series (for plotting) ===
    a_ngn,  b_ngn  = alpha_beta["NGN"]
    a_whel, b_whel = alpha_beta["WHEL"]
    a_ger,  b_ger  = alpha_beta["GEAR"]

    hist_ngn  = compute_hist_divergence(ptd_idx_hist, ptd_ngn_hist,  a_ngn,  b_ngn)
    hist_whel = compute_hist_divergence(ptd_idx_hist, ptd_whel_hist, a_whel, b_whel)
    hist_gear = compute_hist_divergence(ptd_idx_hist, ptd_gear_hist, a_ger,  b_ger)

    # === 4) Historical price arrays for the price plot (unchanged) ===
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

    # Data buffers for live divergence plot
    ticks = []
    div_ngn_list, div_whe_list, div_ger_list = [], [], []

    # Buffers for live price plot
    ticks_p = []
    ngn_p, whel_p, gear_p, idx_p = [], [], [], []

    # Init dynamic divergence plot with historical data
    fig, ax, line_ngn, line_whel, line_gear = init_live_plot(
        hist_ticks, hist_ngn, hist_whel, hist_gear, beta_map
    )

    # Init price plot
    fig_price, ax_price, line_ngn_p, line_whel_p, line_gear_p, line_idx_p = init_price_plot(
        hist_ticks, hist_prices, beta_map
    )

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
        if None not in (
            base_idx, base_ngn, base_whe, base_ger,
            mid_idx,  mid_ngn,  mid_whe,  mid_ger
        ):

            # === 5) Live PTD series ===
            ptd_idx = (mid_idx / base_idx) - 1.0
            ptd_ngn = (mid_ngn / base_ngn) - 1.0
            ptd_whel = (mid_whe / base_whe) - 1.0
            ptd_ger = (mid_ger / base_ger) - 1.0

            # === 6) Live divergences using alpha+beta fair value ===
            div_ngn = (ptd_ngn  - (a_ngn  + b_ngn  * ptd_idx)) * 100.0
            div_whel = (ptd_whel - (a_whel + b_whel * ptd_idx)) * 100.0
            div_ger = (ptd_ger  - (a_ger  + b_ger  * ptd_idx)) * 100.0

            # === 7) Update divergence plot ===
            ticks.append(tick)
            div_ngn_list.append(div_ngn)
            div_whe_list.append(div_whel)
            div_ger_list.append(div_ger)

            update_live_plot(
                ax, line_ngn, line_whel, line_gear,
                ticks, div_ngn_list, div_whe_list, div_ger_list
            )

            # === 8) Update price plot (unchanged) ===
            ticks_p.append(tick)
            ngn_p.append(mid_ngn)
            whel_p.append(mid_whe)
            gear_p.append(mid_ger)
            idx_p.append(mid_idx)

            update_price_plot(
                ax_price, line_ngn_p, line_whel_p, line_gear_p, line_idx_p,
                ticks_p, ngn_p, whel_p, gear_p, idx_p
            )

            # === 9) Trade on raw divergence (no z-scores) ===
            trade_on_div(NGN,  div_ngn)
            trade_on_div(WHEL, div_whel)
            trade_on_div(GEAR, div_ger)

        sleep(SLEEP_SEC)
        tick, status = get_tick_status()

    # Keep the final chart on screen after loop ends
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
