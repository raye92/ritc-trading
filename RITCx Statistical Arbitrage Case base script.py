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
from dataclasses import dataclass
from typing import Dict, Optional

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
STOP_Z          = 4.0
BETA_NOTIONAL   = 50_000
DEFAULT_MAX_HOLD = 120
SLEEP_SEC       = 0.25
PRINT_HEARTBEAT = True
ENABLE_COINTEGRATION = False

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

# ========= BETA-NEUTRAL DIVERGENCE HELPERS =========
@dataclass
class BetaNeutralTrade:
    long_ticker: str
    short_ticker: str
    long_qty: int
    short_qty: int
    long_entry: float
    short_entry: float
    entry_tick: int
    entry_spread: float = 0.0  # Spread when trade was entered
    exit_tick: Optional[int] = None
    status: str = "OPEN"
    reason: str = ""
    realized_pnl: float = 0.0

    def unrealized_pnl(self, price_map: Dict[str, float]) -> float:
        long_price = price_map.get(self.long_ticker)
        short_price = price_map.get(self.short_ticker)
        if long_price is None or short_price is None:
            return 0.0
        return (long_price - self.long_entry) * self.long_qty + (self.short_entry - short_price) * self.short_qty

    def exposure_beta(self, price_map: Dict[str, float], beta_map: Dict[str, float]) -> float:
        long_price = price_map.get(self.long_ticker)
        short_price = price_map.get(self.short_ticker)
        if long_price is None or short_price is None:
            return 0.0
        exposures = [
            (self.long_qty * long_price, beta_map[self.long_ticker]),
            (-self.short_qty * short_price, beta_map[self.short_ticker])
        ]
        gross = sum(abs(v) for v, _ in exposures)
        if gross == 0:
            return 0.0
        return sum((v / gross) * b for v, b in exposures)

    def to_log_dict(self) -> Dict[str, float]:
        return {
            "long": self.long_ticker,
            "short": self.short_ticker,
            "long_qty": self.long_qty,
            "short_qty": self.short_qty,
            "entry_tick": self.entry_tick,
            "exit_tick": self.exit_tick,
            "reason": self.reason,
            "realized_pnl": self.realized_pnl,
        }


def compute_ptd_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    base = price_df.iloc[0]
    return price_df.divide(base).subtract(1.0)


def expected_ptd_returns(ptd_df: pd.DataFrame, beta_map: Dict[str, float], index_col: str = RSM1000) -> pd.DataFrame:
    idx_ptd = ptd_df[index_col]
    exp = {}
    for ticker in TRADABLE_TICKERS:
        exp[ticker] = beta_map.get(ticker, 1.0) * idx_ptd
    return pd.DataFrame(exp, index=ptd_df.index)


def compute_divergence_series(ptd_df: pd.DataFrame, expected_df: pd.DataFrame) -> pd.DataFrame:
    div = (ptd_df[list(TRADABLE_TICKERS)] - expected_df) * 100.0
    div.index = ptd_df.index
    return div


def build_divergence_history(df_hist: pd.DataFrame, beta_map: Dict[str, float]) -> pd.DataFrame:
    prices = df_hist[["RSM1000", "NGN", "WHEL", "GEAR"]].copy()
    ptd = compute_ptd_returns(prices)
    expected = expected_ptd_returns(ptd, beta_map)
    divergence = compute_divergence_series(ptd, expected)
    divergence["Tick"] = df_hist["Tick"].values
    return divergence.set_index("Tick")


def estimate_half_life(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    if len(clean) < 30:
        return None
    y = clean.diff().dropna()
    x = clean.shift(1).dropna().loc[y.index]
    if len(x) != len(y):
        return None
    beta = np.polyfit(x, y, 1)[0]
    if beta >= 0:
        return None
    return np.log(2) / -beta


def pick_divergence_trade(z_map: Dict[str, float], entry_z: float) -> Optional[Dict[str, float]]:
    """Pick a trade based on the spread between highest and lowest z-scores.
   
    Args:
        z_map: Dictionary of ticker -> z-score
        entry_z: Minimum spread (difference) between highest and lowest z-scores to enter trade
       
    Returns:
        Dict with 'long', 'short', 'long_z', 'short_z' if spread >= entry_z, else None
    """
    if not z_map or len(z_map) < 2:
        return None
   
    # Find ticker with highest z-score (most overvalued - SHORT)
    # Find ticker with lowest z-score (most undervalued - LONG)
    over = max(z_map.items(), key=lambda x: x[1])
    under = min(z_map.items(), key=lambda x: x[1])
   
    # Calculate spread between highest and lowest
    spread = over[1] - under[1]
   
    # Only trade if spread is large enough
    if spread < entry_z:
        return None
   
    return {
        "short": over[0],      # Short the most overvalued
        "short_z": over[1],
        "long": under[0],      # Long the most undervalued
        "long_z": under[1],
    }


def calc_beta_neutral_shares(long_ticker: str, short_ticker: str, price_map: Dict[str, float],
                             beta_map: Dict[str, float], base_notional: float = BETA_NOTIONAL):
    long_price = price_map.get(long_ticker)
    short_price = price_map.get(short_ticker)
    if long_price is None or short_price is None:
        return None
    beta_long = abs(beta_map.get(long_ticker, 1.0))
    beta_short = abs(beta_map.get(short_ticker, 1.0))
    if beta_long < 1e-6 or beta_short < 1e-6:
        return None
    long_value = base_notional
    short_value = base_notional * (beta_long / beta_short)
    long_qty = int(max(1, min(long_value / long_price, MAX_TRADE_SIZE)))
    short_qty = int(max(1, min(short_value / short_price, MAX_TRADE_SIZE)))
    return long_qty, short_qty


def enter_beta_neutral_trade(tick: int, long_tkr: str, short_tkr: str,
                             price_map: Dict[str, float], beta_map: Dict[str, float],
                             entry_spread: float = 0.0) -> Optional[BetaNeutralTrade]:
    if not within_limits():
        print("[LIMIT] Cannot enter beta-neutral trade; limits reached.")
        return None
    qtys = calc_beta_neutral_shares(long_tkr, short_tkr, price_map, beta_map)
    if qtys is None:
        print("[SIZING] Unable to size beta-neutral trade.")
        return None
    long_qty, short_qty = qtys
    ok_long = place_mkt(long_tkr, "BUY", long_qty)
    ok_short = place_mkt(short_tkr, "SELL", short_qty)
    if ok_long and ok_short:
        trade = BetaNeutralTrade(
            long_ticker=long_tkr,
            short_ticker=short_tkr,
            long_qty=long_qty,
            short_qty=short_qty,
            long_entry=price_map[long_tkr],
            short_entry=price_map[short_tkr],
            entry_tick=tick,
            entry_spread=entry_spread,
        )
        long_value = trade.long_qty * trade.long_entry
        short_value = trade.short_qty * trade.short_entry
        print(f"[BETA ENTER] Long {long_tkr} ({long_qty}sh ${long_value:,.0f}) / "
              f"Short {short_tkr} ({short_qty}sh ${short_value:,.0f}) | Spread: {entry_spread:.2f}")
        return trade
    print("[ORDER FAIL] Unable to enter beta-neutral trade.")
    if ok_long:
        place_mkt(long_tkr, "SELL", long_qty)
    if ok_short:
        place_mkt(short_tkr, "BUY", short_qty)
    return None


def scale_in_beta_neutral_trade(trade: BetaNeutralTrade, tick: int,
                                 price_map: Dict[str, float], beta_map: Dict[str, float],
                                 scale_notional: float = BETA_NOTIONAL) -> bool:
    """Add more positions to existing trade when spread widens."""
    if not within_limits():
        return False
    qtys = calc_beta_neutral_shares(trade.long_ticker, trade.short_ticker, price_map, beta_map, scale_notional)
    if qtys is None:
        return False
    add_long_qty, add_short_qty = qtys
    ok_long = place_mkt(trade.long_ticker, "BUY", add_long_qty)
    ok_short = place_mkt(trade.short_ticker, "SELL", add_short_qty)
    if ok_long and ok_short:
        # Update trade with new quantities (weighted average entry price)
        total_long_qty = trade.long_qty + add_long_qty
        total_short_qty = trade.short_qty + add_short_qty
        trade.long_entry = (trade.long_entry * trade.long_qty + price_map[trade.long_ticker] * add_long_qty) / total_long_qty
        trade.short_entry = (trade.short_entry * trade.short_qty + price_map[trade.short_ticker] * add_short_qty) / total_short_qty
        trade.long_qty = total_long_qty
        trade.short_qty = total_short_qty
        long_value = add_long_qty * price_map[trade.long_ticker]
        short_value = add_short_qty * price_map[trade.short_ticker]
        print(f"[SCALE IN] Added Long {trade.long_ticker} ({add_long_qty}sh ${long_value:,.0f}) / "
              f"Short {trade.short_ticker} ({add_short_qty}sh ${short_value:,.0f}) | "
              f"Total: {trade.long_qty}sh/{trade.short_qty}sh")
        return True
    if ok_long:
        place_mkt(trade.long_ticker, "SELL", add_long_qty)
    if ok_short:
        place_mkt(trade.short_ticker, "BUY", add_short_qty)
    return False


def exit_beta_neutral_trade(trade: BetaNeutralTrade, tick: int, price_map: Dict[str, float], reason: str) -> float:
    ok_long = place_mkt(trade.long_ticker, "SELL", trade.long_qty)
    ok_short = place_mkt(trade.short_ticker, "BUY", trade.short_qty)
    pnl = trade.unrealized_pnl(price_map)
    trade.exit_tick = tick
    trade.status = "CLOSED"
    trade.reason = reason
    trade.realized_pnl = pnl
    if ok_long and ok_short:
        print(f"[BETA EXIT] {reason} | {trade.long_ticker}/{trade.short_ticker} pnl={pnl:0.2f}")
    else:
        print("[ORDER FAIL] exit order issue encountered.")
    return pnl


# ========= DYNAMIC PLOT (single figure, 3 lines) =========
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

def init_spread_plot(hist_ticks=None, hist_spreads=None):
    plt.ion()
    fig, ax = plt.subplots()
    # Plot historical spreads (every 5th point for spacing)
    if hist_ticks is not None and hist_spreads is not None:
        if 'NGN-WHEL' in hist_spreads:
            ax.plot(hist_ticks[::5], hist_spreads['NGN-WHEL'][::5], '--', color='purple', alpha=0.3, linewidth=1.5,
                    label="NGN-WHEL (Hist)")
        if 'NGN-GEAR' in hist_spreads:
            ax.plot(hist_ticks[::5], hist_spreads['NGN-GEAR'][::5], '--', color='orange', alpha=0.3, linewidth=1.5,
                    label="NGN-GEAR (Hist)")
        if 'WHEL-GEAR' in hist_spreads:
            ax.plot(hist_ticks[::5], hist_spreads['WHEL-GEAR'][::5], '--', color='cyan', alpha=0.3, linewidth=1.5,
                    label="WHEL-GEAR (Hist)")
    # Create empty lines for live spreads
    line_nw, = ax.plot([], [], color='purple', label="NGN-WHEL (Live)")
    line_ng, = ax.plot([], [], color='orange', label="NGN-GEAR (Live)")
    line_wg, = ax.plot([], [], color='cyan', label="WHEL-GEAR (Live)")
    # Add horizontal lines for entry/exit thresholds
    ax.axhline(y=2.0, color='green', linestyle=':', alpha=0.5, label='Entry Threshold')
    ax.axhline(y=-2.0, color='green', linestyle=':', alpha=0.5)
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Exit Threshold')
    ax.axhline(y=-0.5, color='red', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.set_title("Z-Score Spreads Between Ticker Pairs")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Z-Score Spread")
    ax.grid(True)
    ax.legend()
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, line_nw, line_ng, line_wg

def update_spread_plot(ax, line_nw, line_ng, line_wg, ticks, spread_nw, spread_ng, spread_wg):
    line_nw.set_data(ticks, spread_nw)
    line_ng.set_data(ticks, spread_ng)
    line_wg.set_data(ticks, spread_wg)
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
    pair_results = []
    tradable_pairs = []
    pair_states = {}
    if ENABLE_COINTEGRATION:
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

    divergence_hist_df = build_divergence_history(df_hist, beta_map)
    div_vol = divergence_hist_df[list(TRADABLE_TICKERS)].std().replace(0.0, np.nan)
    div_vol_map = {t: float(div_vol.get(t, 1.0)) if not pd.isna(div_vol.get(t, np.nan)) else 1.0
                   for t in TRADABLE_TICKERS}
    avg_div_series = divergence_hist_df[list(TRADABLE_TICKERS)].mean(axis=1)
    est_half_life = estimate_half_life(avg_div_series)
    max_hold_ticks = DEFAULT_MAX_HOLD if est_half_life is None else int(max(DEFAULT_MAX_HOLD, est_half_life * 3))
    print(f"\nEstimated mean-reversion half-life: "
          f"{'n/a' if est_half_life is None else f'{est_half_life:0.1f}'} ticks "
          f"(timeout={max_hold_ticks} ticks)")
    print("\nHistorical divergence snapshot (last 5 rows):\n")
    print(divergence_hist_df.tail().to_string())

    # Calculate historical z-score spreads
    hist_z_scores = {}
    for ticker in TRADABLE_TICKERS:
        hist_div = divergence_hist_df[ticker].values
        hist_vol = div_vol_map.get(ticker, 1.0)
        hist_z_scores[ticker] = hist_div / hist_vol if hist_vol > 1e-6 else hist_div
   
    # Calculate historical spreads between pairs
    hist_spreads = {
        'NGN-WHEL': hist_z_scores['NGN'] - hist_z_scores['WHEL'],
        'NGN-GEAR': hist_z_scores['NGN'] - hist_z_scores['GEAR'],
        'WHEL-GEAR': hist_z_scores['WHEL'] - hist_z_scores['GEAR']
    }

    # Live PTD bases (first-seen mids)
    base_idx = None
    base_ngn = None
    base_whe = None
    base_ger = None

    # Data buffers for live plot
    ticks = []
    div_ngn_list, div_whe_list, div_ger_list = [], [], []

    # Buffers for live spread plot
    ticks_spread = []
    spread_nw_list, spread_ng_list, spread_wg_list = [], [], []

    # Init dynamic plot with historical data and beta values in legend
    fig, ax, line_ngn, line_whel, line_gear = init_live_plot(
        hist_ticks, hist_ngn, hist_whel, hist_gear, beta_map)

    # Init spread plot
    fig_spread, ax_spread, line_nw, line_ng, line_wg = init_spread_plot(
        hist_ticks, hist_spreads)

    # Track active positions per pair (can have multiple trades per pair)
    active_pair_positions: Dict[str, list] = {
        'NGN-WHEL': [],
        'NGN-GEAR': [],
        'WHEL-GEAR': []
    }
    trade_log = []
    signal_history = []
    realized_pnl_total = 0.0

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

            price_map = {
                RSM1000: mid_idx,
                NGN: mid_ngn,
                WHEL: mid_whe,
                GEAR: mid_ger,
            }

            divergence_map = {
                NGN: div_ngn,
                WHEL: div_whe,
                GEAR: div_ger,
            }
            z_map = {}
            for ticker in TRADABLE_TICKERS:
                denom = div_vol_map.get(ticker, 1.0)
                if abs(denom) < 1e-6:
                    denom = 1.0
                z_map[ticker] = divergence_map[ticker] / denom

            # Calculate and store z-score spreads
            spread_nw = z_map.get(NGN, 0.0) - z_map.get(WHEL, 0.0)
            spread_ng = z_map.get(NGN, 0.0) - z_map.get(GEAR, 0.0)
            spread_wg = z_map.get(WHEL, 0.0) - z_map.get(GEAR, 0.0)
           
            ticks_spread.append(tick)
            spread_nw_list.append(spread_nw)
            spread_ng_list.append(spread_ng)
            spread_wg_list.append(spread_wg)
            update_spread_plot(ax_spread, line_nw, line_ng, line_wg,
                             ticks_spread, spread_nw_list, spread_ng_list, spread_wg_list)

            # Calculate spreads between all pairs
            pair_spreads = {
                'NGN-WHEL': z_map.get(NGN, 0.0) - z_map.get(WHEL, 0.0),
                'NGN-GEAR': z_map.get(NGN, 0.0) - z_map.get(GEAR, 0.0),
                'WHEL-GEAR': z_map.get(WHEL, 0.0) - z_map.get(GEAR, 0.0),
            }

            signal_comment = "FLAT"
            unrealized = 0.0
            portfolio_beta_now = 0.0

            if PRINT_HEARTBEAT and tick % 10 == 0:  # Print every 10 ticks
                z_str = ", ".join([f"{t}:{z:.2f}" for t, z in z_map.items()])
                spread_str = ", ".join([f"{pair}:{spread:.2f}" for pair, spread in pair_spreads.items()])
                print(f"[TICK {tick}] Z-scores: {z_str}")
                print(f"  Spreads: {spread_str}")

            # Process each pair independently
            for pair_name, spread in pair_spreads.items():
                # Parse pair to get tickers
                if pair_name == 'NGN-WHEL':
                    ticker1, ticker2 = NGN, WHEL
                elif pair_name == 'NGN-GEAR':
                    ticker1, ticker2 = NGN, GEAR
                elif pair_name == 'WHEL-GEAR':
                    ticker1, ticker2 = WHEL, GEAR
                else:
                    continue

                # Determine long/short based on spread direction
                if spread > 0:
                    long_ticker, short_ticker = ticker2, ticker1  # ticker1 is overvalued
                else:
                    long_ticker, short_ticker = ticker1, ticker2  # ticker2 is overvalued

                # EXIT: If spread < 0.5, exit all positions for this pair
                if abs(spread) < 0.5:
                    positions = active_pair_positions[pair_name]
                    if positions:
                        total_pnl = 0.0
                        for trade in positions:
                            pnl = exit_beta_neutral_trade(trade, tick, price_map, "MEAN_REVERSION")
                            total_pnl += pnl
                            trade_log.append(trade.to_log_dict())
                        realized_pnl_total += total_pnl
                        if PRINT_HEARTBEAT:
                            print(f"[EXIT ALL] {pair_name} spread={spread:.2f} | Closed {len(positions)} positions | PnL: {total_pnl:.2f}")
                        active_pair_positions[pair_name] = []
                        signal_comment = f"EXIT {pair_name}"

                # ENTER: If spread > 2.0, place orders every tick
                elif abs(spread) > 2.0:
                    if within_limits():
                        trade = enter_beta_neutral_trade(
                            tick, long_ticker, short_ticker, price_map, beta_map, entry_spread=abs(spread)
                        )
                        if trade:
                            active_pair_positions[pair_name].append(trade)
                            if PRINT_HEARTBEAT:
                                print(f"[ENTER] {pair_name} spread={spread:.2f} | Long {long_ticker} / Short {short_ticker} | Total positions: {len(active_pair_positions[pair_name])}")
                            signal_comment = f"ENTER {pair_name}"
                        elif PRINT_HEARTBEAT and tick % 5 == 0:  # Less frequent to avoid spam
                            print(f"[ENTER FAIL] {pair_name} spread={spread:.2f} | Entry failed (check limits)")

            # Calculate total unrealized PnL and portfolio beta across all positions
            for pair_name, positions in active_pair_positions.items():
                for trade in positions:
                    unrealized += trade.unrealized_pnl(price_map)
                    portfolio_beta_now += trade.exposure_beta(price_map, beta_map)

            signal_history.append({
                "tick": tick,
                "div_NGN": div_ngn,
                "div_WHEL": div_whe,
                "div_GEAR": div_ger,
                "z_NGN": z_map.get(NGN),
                "z_WHEL": z_map.get(WHEL),
                "z_GEAR": z_map.get(GEAR),
                "signal": signal_comment,
                "portfolio_beta": portfolio_beta_now,
                "unrealized_pnl": unrealized,
                "realized_pnl": realized_pnl_total,
            })

            if ENABLE_COINTEGRATION:
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

    if signal_history:
        signal_df = pd.DataFrame(signal_history)
        print("\nSignal summary (last 10 observations):\n")
        print(signal_df.tail(10).to_string(index=False))
    else:
        print("\nNo signal observations captured.\n")

    if trade_log:
        trade_df = pd.DataFrame(trade_log)
        total_pnl = trade_df["realized_pnl"].sum()
        print("\nBeta-neutral trade log:\n")
        print(trade_df.to_string(index=False))
        print(f"\nTotal realized P&L: {total_pnl:0.2f}")
    else:
        print("\nNo beta-neutral trades executed.\n")

    # Keep the final chart on screen after loop ends
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()

