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


def get_bid_ask(ticker):
    payload = {'ticker': ticker}
    resp = s.get ('http://localhost:9999/v1/securities/book', params = payload)
    if resp.ok:
        book = resp.json()
        bid_side_book = book['bids']
        ask_side_book = book['asks']

        # Check if book is empty
        if not bid_side_book or not ask_side_book:
            return None, None
        
        bid_prices_book = [item["price"] for item in bid_side_book]
        ask_prices_book = [item['price'] for item in ask_side_book]
        
        best_bid_price = bid_prices_book[0]
        best_ask_price = ask_prices_book[0]
  
        return best_bid_price, best_ask_price
    return None, None

def get_time_sales(ticker):
    payload = {'ticker': ticker}
    resp = s.get ('http://localhost:9999/v1/securities/tas', params = payload)
    if resp.ok:
        book = resp.json()
        time_sales_book = [item["quantity"] for item in book]
        return time_sales_book

def lrg_mkt_order(ticker, action, quantity):
    quantity = int(quantity)
    market_bid, market_ask = get_bid_ask(ticker)
    # print("LARGE MARKET ORDER", ticker, action, quantity, market_bid, market_ask)
    for i in range(quantity // ORDER_LIMIT):
        resp = s.post('http://localhost:9999/v1/orders', params={'ticker': ticker, 'type': 'MARKET', 'quantity': ORDER_LIMIT, 'action': action})
        # print(resp.json())
        quantity -= ORDER_LIMIT
    resp = s.post('http://localhost:9999/v1/orders', params={'ticker': ticker, 'type': 'MARKET', 'quantity': quantity, 'action': action})
    # print(resp.json())

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

def main():
    tick, status = get_tick()
    # Assuming 's' is your requests.Session() object
    ticker_list = [i['ticker'] for i in s.get('http://localhost:9999/v1/securities').json()]
    previous_tick = -1
    news_set = set()
    current_volatility = 0
    option_count = 0
    runs_per_tick = 0

    while status == 'ACTIVE':
        if previous_tick != tick:
            previous_tick = tick
            runs_per_tick = 0
            
        if runs_per_tick < 4:
            # 1. Fetch all securities in one single API call
            all_securities = s.get('http://localhost:9999/v1/securities').json()
            prices = {sec['ticker']: sec for sec in all_securities}

            # Recalculate option_count from local memory
            option_count = 0
            for ticker in ticker_list:
                if ticker != "RTM":
                    option_count += abs(prices.get(ticker, {}).get('position', 0))

            rtm = prices.get("RTM", {})
            market_bid = rtm.get('bid', 0)
            market_ask = rtm.get('ask', 0)
            S = (market_bid + market_ask) / 2.0
            
            # 2. Parse News for the true Realized Volatility
            news = get_news()
            if news:
                for item in news:
                    if item['news_id'] not in news_set:
                        news_set.add(item['news_id'])
                        body = item.get('body', '')
                        if 'realized volatility' in body:
                            m = re.search(r'volatility (?:of RTM (?:this week )?)?(?:is|will be) (\d+)%', body)
                            if m:
                                new_vol = int(m.group(1))
                                if current_volatility != new_vol:
                                    current_volatility = new_vol
                                    print(f"--- NEW VOLATILITY SHOCK: {current_volatility}% ---")
                                    flatten_positions(ticker_list, prices)
            
            # 3. Calculate Black-Scholes variables BEFORE the ticker loop
            seconds_remaining = max(0.0001, 300 - tick) 
            T = (seconds_remaining / 300.0) * (20.0 / 240.0) 
            
            vol = current_volatility / 100.0
            r = 0.0
            
            should_print = (tick % 10 == 0 and runs_per_tick == 0)
            if should_print:
                print(f"\n===== TICK {tick} | Vol: {current_volatility}% | Options Traded: {option_count} =====")

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

                    blunt_hedge_shares = 100
                    action = "HOLD"
                    profit = 0.0

                    if option_count >= 1000:
                        print(f"  {ticker:10s} EV:{ev:6.2f} | LIMIT REACHED")
                        continue

                    if ev > (opt_ask + threshold):
                        action = "BUY"
                        profit = ev - opt_ask - transaction_fee
                        lrg_mkt_order(ticker, 'BUY', 1)
                        option_count += 1
                        
                        if option_type == 'call':
                            lrg_mkt_order("RTM", 'SELL', blunt_hedge_shares)
                        else: # put
                            lrg_mkt_order("RTM", 'BUY', blunt_hedge_shares)
                            
                    elif ev < (opt_bid - threshold):
                        action = "SELL"
                        profit = opt_bid - ev - transaction_fee
                        lrg_mkt_order(ticker, 'SELL', 1)
                        option_count += 1
                        
                        if option_type == 'call':
                            lrg_mkt_order("RTM", 'BUY', blunt_hedge_shares)
                        else: # put
                            lrg_mkt_order("RTM", 'SELL', blunt_hedge_shares)

                    if should_print:
                        diff = ev - (opt_bid + opt_ask) / 2.0
                        print(f"  {ticker:10s} EV:{ev:6.2f} | Mid:{(opt_bid+opt_ask)/2:6.2f} | Diff:{diff:+.2f} | Profit:{profit:+.2f} | {action}")
                    
            runs_per_tick += 1
            time.sleep(0.2)
            
        tick, status = get_tick()

if __name__ == "__main__":
    print("Starting Ray...")
    main()
