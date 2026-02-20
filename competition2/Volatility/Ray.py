"""RITC 2026 Algorithmic Market Making Trading Case - REST API Basic Script"""
import re
import requests
import os
from dotenv import load_dotenv
import time
import math

load_dotenv()

s = requests.Session()
s.headers.update({'X-API-key': os.getenv('API_KEY')})

ORDER_LIMIT = 10000

def get_tick():
    resp = s.get('http://localhost:9999/v1/case')
    if resp.ok:
        case = resp.json()
        return case['tick'], case['status']

def get_news():
    resp = s.get('http://localhost:9999/v1/news')
    if resp.ok:
        return resp.json()
    return None

# Optimized to remove unnecessary API calls
def lrg_mkt_order(ticker, action, quantity):
    quantity = int(quantity)
    if quantity <= 0:
        return

    for i in range(quantity // ORDER_LIMIT):
        s.post('http://localhost:9999/v1/orders', params={'ticker': ticker, 'type': 'MARKET', 'quantity': ORDER_LIMIT, 'action': action})
        
    remainder = quantity % ORDER_LIMIT
    if remainder > 0:
        s.post('http://localhost:9999/v1/orders', params={'ticker': ticker, 'type': 'MARKET', 'quantity': remainder, 'action': action})

def flatten_positions(ticker_list, prices):
    for ticker_symbol in ticker_list:     
        position = int(prices.get(ticker_symbol, {}).get('position', 0))
        if position > 0:
            lrg_mkt_order(ticker_symbol, 'SELL', position)
        elif position < 0:
            lrg_mkt_order(ticker_symbol, 'BUY', -position)

def cumulative_normal_distribution(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def black_scholes_price(S, K, T, r, vol, option_type):
    if T <= 0.00001 or vol <= 0:
        return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    
    d1 = (math.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    
    if option_type == 'call':
        price = S * cumulative_normal_distribution(d1) - K * math.exp(-r * T) * cumulative_normal_distribution(d2)
    else:
        price = K * math.exp(-r * T) * cumulative_normal_distribution(-d2) - S * cumulative_normal_distribution(-d1)
    
    return price

def calculate_delta(S, K, T, r, vol, option_type):
    if T <= 0.00001 or vol <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    d1 = (math.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * math.sqrt(T))
    
    if option_type == 'call':
        return cumulative_normal_distribution(d1)
    else:
        return cumulative_normal_distribution(d1) - 1.0

def main():
    tick, status = get_tick()
    # Assuming 's' is your requests.Session() object
    ticker_list = [i['ticker'] for i in s.get('http://localhost:9999/v1/securities').json()]
    previous_tick = -1
    news_set = set()
    current_volatility = 0
    option_count = 0
    runs_per_tick = 0
    
    # Wait for the case to actively start
    while status != 'ACTIVE':
        time.sleep(1)
        print(f"Game status is {status}. Waiting for the game to start...")
        tick, status = get_tick()
        
    print("Starting Volatility Arb Engine...")
    
    while status == 'ACTIVE':
        if previous_tick != tick:
            previous_tick = tick
            runs_per_tick = 0
            
        if runs_per_tick < 4:
            # 1. Fetch all securities in one single API call
            all_securities = s.get('http://localhost:9999/v1/securities').json()
            prices = {sec['ticker']: sec for sec in all_securities}

            # Recalculate option_count from local memory (Gross Limit tracking)
            option_count = 0
            for ticker in ticker_list:
                if ticker != "RTM":
                    option_count += abs(prices.get(ticker, {}).get('position', 0))

            rtm = prices.get("RTM", {})
            market_bid = rtm.get('bid', 0)
            market_ask = rtm.get('ask', 0)
            if market_bid == 0 or market_ask == 0:
                time.sleep(0.2)
                continue # Skip loop if market data is invalid
                
            S = (market_bid + market_ask) / 2.0
            
            # 2. Parse News for Realized Volatility
            news = get_news()
            if news:
                for item in news:
                    if item['news_id'] not in news_set:
                        news_set.add(item['news_id'])
                        body = item.get('body', '')
                        
                        # Look for the exact weekly volatility announcement
                        if 'realized volatility' in body and 'between' not in body:
                            m = re.search(r'volatility (?:of RTM (?:this week )?)?(?:is|will be) (\d+)%', body)
                            if m:
                                new_vol = int(m.group(1))
                                if current_volatility != new_vol:
                                    current_volatility = new_vol
                                    print(f"--- NEW VOLATILITY SHOCK: {current_volatility}% ---")
                                    flatten_positions(ticker_list, prices)
            
            # 3. Time calculation for Black-Scholes
            seconds_remaining = max(0.0001, 300 - tick) 
            T = (seconds_remaining / 300.0) * (20.0 / 240.0) 
            
            vol = current_volatility / 100.0
            r = 0.0
            
            should_print = (tick % 10 == 0 and runs_per_tick == 0)
            if should_print:
                print(f"\n===== TICK {tick} | Vol: {current_volatility}% | Options Traded: {option_count} =====")

            total_portfolio_delta = 0.0

            if vol > 0:
                for ticker in ticker_list:
                    if ticker == "RTM":
                        continue
                        
                    option_type = 'call' if 'C' in ticker else 'put'
                    strike_price = int(ticker[-2:])
                    
                    sec = prices.get(ticker, {})
                    opt_bid = sec.get('bid')
                    opt_ask = sec.get('ask')
                    if opt_bid is None or opt_ask is None:
                        continue
                    
                    ev = black_scholes_price(S, strike_price, T, r, vol, option_type)
                    
                    transaction_fee = 0.01 
                    required_edge = 0.02 
                    threshold = transaction_fee + required_edge

                    raw_delta = calculate_delta(S, strike_price, T, r, vol, option_type)
                    delta_per_contract = raw_delta * 100.0 

                    current_position = int(prices.get(ticker, {}).get('position', 0))
                    
                    # Track our running Delta based on what we CURRENTLY hold
                    total_portfolio_delta += delta_per_contract * current_position 

                    action = "HOLD"
                    profit = 0.0

                    # --- EXIT LOGIC ---
                    if current_position > 0 and ev <= (opt_bid + transaction_fee):
                        action = "CLOSE LONG"
                        lrg_mkt_order(ticker, 'SELL', current_position)
                        total_portfolio_delta -= delta_per_contract * current_position 
                        if should_print: print(f"  {ticker:10s} EV:{ev:6.2f} | Mid:{(opt_bid+opt_ask)/2:6.2f} | {action}")
                        continue 

                    elif current_position < 0 and ev >= (opt_ask - transaction_fee):
                        action = "CLOSE SHORT"
                        lrg_mkt_order(ticker, 'BUY', abs(current_position))
                        total_portfolio_delta -= delta_per_contract * current_position 
                        if should_print: print(f"  {ticker:10s} EV:{ev:6.2f} | Mid:{(opt_bid+opt_ask)/2:6.2f} | {action}")
                        continue 

                    # --- ENTRY LOGIC ---
                    if ev > (opt_ask + threshold):
                        if option_count >= 1000 and current_position >= 0:
                            if should_print: print(f"  {ticker:10s} EV:{ev:6.2f} | LIMIT REACHED (Cannot Buy)")
                            continue
                            
                        action = "BUY"
                        profit = ev - opt_ask - transaction_fee
                        qty = max(1, int(profit * 10))
                        lrg_mkt_order(ticker, 'BUY', qty)
                        total_portfolio_delta += delta_per_contract * qty 
                            
                    elif ev < (opt_bid - threshold):
                        if option_count >= 1000 and current_position <= 0:
                            if should_print: print(f"  {ticker:10s} EV:{ev:6.2f} | LIMIT REACHED (Cannot Sell)")
                            continue
                            
                        action = "SELL"
                        profit = opt_bid - ev - transaction_fee
                        qty = max(1, int(profit * 10))
                        lrg_mkt_order(ticker, 'SELL', qty)
                        total_portfolio_delta -= delta_per_contract * qty 

                    if should_print and action != "HOLD":
                        diff = ev - (opt_bid + opt_ask) / 2.0
                        print(f"  {ticker:10s} EV:{ev:6.2f} | Mid:{(opt_bid+opt_ask)/2:6.2f} | Diff:{diff:+.2f} | Profit:{profit:+.2f} | {action}")
                
                # --- MASTER PORTFOLIO HEDGE ---
                rtm_position = int(prices.get("RTM", {}).get('position', 0))
                ideal_rtm_position = -total_portfolio_delta
                shares_to_trade = int(ideal_rtm_position - rtm_position)

                # Execute hedge only if off by > 50 shares to save on 1-cent transaction fees
                if shares_to_trade > 50:
                    lrg_mkt_order("RTM", 'BUY', shares_to_trade)
                elif shares_to_trade < -50:
                    lrg_mkt_order("RTM", 'SELL', abs(shares_to_trade))
                    
            runs_per_tick += 1
            time.sleep(0.2)
            
        tick, status = get_tick()

if __name__ == "__main__":
    main()