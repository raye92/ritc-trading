"""RITC 2026 Algorithmic Market Making Trading Case - REST API Basic Script"""
import re
import requests
import os
from dotenv import load_dotenv
import time
import math
import asyncio


load_dotenv()

s = requests.Session()
s.headers.update({'X-API-key': os.getenv('API_KEY')})

ORDER_LIMIT = 10000
GROSS_LIMIT = 50000
NET_LIMIT = 30000
WEEKLY_LIMIT = None

def getWeeklyLimit():
    """Returns the weekly position limit from news (used as WEEKLY_LIMIT)."""
    resp = s.get('http://localhost:9999/v1/news')
    if resp.ok:
        news = resp.json()
        if news:
            news_string = news[0]['body'].split()
            for i in news_string:
                if i.isdigit():
                    return int(i)
    return None

def getTradingLimits():
    resp = s.get('http://localhost:9999/v1/limits')
    if resp.ok:
        info = resp.json()
        if info:
            return info[0]['gross_limit'], info[0]['net_limit']
    return None, None

def get_tick():
    resp = s.get('http://localhost:9999/v1/case')
    if resp.ok:
        case = resp.json()
        return case['tick'], case['status']

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

def get_ind_position(ticker):
    payload = {'ticker': ticker}
    resp = s.get('http://localhost:9999/v1/securities', params=payload)
    if resp.ok:
        securities = resp.json()
        for security in securities:
            if security['ticker'] == ticker:
                return security['position']
    return 0  # Return 0 if ticker not found or request fails

def get_position():
    resp = s.get ('http://localhost:9999/v1/securities')
    if resp.ok:
        book = resp.json()
        return abs(book[0]['position']) + abs(book[1]['position']) + abs(book[2]['position']) + abs(book[3]['position'])

def get_all_positions():
    """Fetches positions for all tickers in one API call."""
    resp = s.get('http://localhost:9999/v1/securities')
    if resp.ok:
        securities = resp.json()
        positions = {sec['ticker']: sec['position'] for sec in securities}
        total_options_exposure = sum(abs(pos) for ticker, pos in positions.items() if ticker != "RTM")
        return positions, total_options_exposure
    return {}, 0

def get_open_orders(ticker):
    payload = {'ticker': ticker}
    resp = s.get ('http://localhost:9999/v1/orders', params = payload)
    if resp.ok:
        orders = resp.json()
        buy_orders = [item for item in orders if item["action"] == "BUY"]
        sell_orders = [item for item in orders if item["action"] == "SELL"]
        return buy_orders, sell_orders

def get_order_status(order_id):
    resp = s.get ('http://localhost:9999/v1/orders' + '/' + str(order_id))
    if resp.ok:
        order = resp.json()
        return order['status']

def lrg_mkt_order(ticker, action, quantity):
    quantity = int(quantity)
    for i in range(quantity // ORDER_LIMIT):
        s.post('http://localhost:9999/v1/orders', params={'ticker': ticker, 'type': 'MARKET', 'quantity': ORDER_LIMIT, 'action': action})
        quantity -= ORDER_LIMIT
    s.post('http://localhost:9999/v1/orders', params={'ticker': ticker, 'type': 'MARKET', 'quantity': quantity, 'action': action})

async def async_spray(ticker, prices, ewma_vol, tick, current_individual_limit):
    sec = prices.get(ticker, {})
    best_bid = sec.get('bid')
    best_ask = sec.get('ask')
    pos = sec.get('position', 0)

    agg_abs_pos = sum(abs(v['position']) for k, v in prices.items() if k != 'RTM')
    net_pos = sum(v['position'] for k, v in prices.items() if k != 'RTM')

    if best_bid is None or best_ask is None:
        return

    mid = 0.5 * (best_bid + best_ask)
    TICK_SIZE = 0.01
    spread = max(TICK_SIZE, best_ask - best_bid)

    base_size = int(current_individual_limit * 0.1)
    bid_qty = base_size
    ask_qty = base_size
    bid_price = best_bid - TICK_SIZE
    ask_price = best_ask + TICK_SIZE

    if bid_qty > 0:
        await asyncio.to_thread(
            s.post, 'http://localhost:9999/v1/orders',
            params={'ticker': ticker, 'type': 'LIMIT', 'quantity': bid_qty, 'action': 'BUY', 'price': bid_price}
        )

    if ask_qty > 0:
        await asyncio.to_thread(
            s.post, 'http://localhost:9999/v1/orders',
            params={'ticker': ticker, 'type': 'LIMIT', 'quantity': ask_qty, 'action': 'SELL', 'price': ask_price}
        )

def flatten_positions(ticker_list, prices):
    for ticker_symbol in ticker_list:
        position = int(prices.get(ticker_symbol, {}).get('position', 0))
        limit = WEEKLY_LIMIT // 4
        if position > limit:
            lrg_mkt_order(ticker_symbol, 'SELL', position - limit)
        elif position < -limit:
            lrg_mkt_order(ticker_symbol, 'BUY', -(position + limit))

async def main_loop():
    global GROSS_LIMIT, NET_LIMIT, WEEKLY_LIMIT
    tick, status = get_tick()
    ticker_list = [i['ticker'] for i in s.get('http://localhost:9999/v1/securities').json() if i['ticker'] != 'RTM']
    previous_tick = -1
    runs_per_tick = 0

    limits = getTradingLimits()
    if limits[0] is not None:
        GROSS_LIMIT = limits[0]
    if limits[1] is not None:
        NET_LIMIT = limits[1]
    wl = getWeeklyLimit()
    if wl is not None:
        WEEKLY_LIMIT = wl

    CURRENT_INDIVIDUAL_LIMIT = NET_LIMIT // 4 if NET_LIMIT else 5000
    ewma_vol_dict = {ticker: 0.30 for ticker in ticker_list}

    while status != 'ACTIVE':
        time.sleep(1)
        print(f"Game status is {status}. Waiting for the game to start...")
        tick, status = get_tick()

    print("Starting Algo Market Making...")

    while status == 'ACTIVE':
        if previous_tick != tick:
            previous_tick = tick
            runs_per_tick = 0

        if runs_per_tick < 20:
            all_securities = s.get('http://localhost:9999/v1/securities').json()
            prices = {sec['ticker']: sec for sec in all_securities}

            if tick % 60 >= 55:
                flatten_positions(ticker_list, prices)
            else:
                tasks = [
                    async_spray(tkr, prices, ewma_vol_dict[tkr], tick, CURRENT_INDIVIDUAL_LIMIT)
                    for tkr in ticker_list
                ]
                await asyncio.gather(*tasks)

            runs_per_tick += 1
            await asyncio.sleep(0.05)

        tick, status = get_tick()

def main():
    asyncio.run(main_loop())

if __name__ == '__main__':
    main()
