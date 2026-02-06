"""RITC 2026 Algorithmic Market Making Trading Case - REST API Basic Script"""

import requests
from time import sleep
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

s = requests.Session()
s.headers.update({'X-API-key': os.getenv('API_KEY')})

#Sample setup
MAX_EXPOSURE = 15000
ORDER_LIMIT = 10000

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
        return abs(book[0]['position']) + abs(book[1]['position'])

def accept_tender(tender):
    tender_id = tender['tender_id']
    return s.post(f'http://localhost:9999/v1/tenders/{tender_id}')

def lrg_mkt_order(ticker, action, quantity):
    quantity = int(quantity)
    print("LARGE MARKET ORDER", ticker, action, quantity)
    for i in range(quantity // ORDER_LIMIT):
        s.post('http://localhost:9999/v1/orders', params={'ticker': ticker, 'type': 'MARKET', 'quantity': ORDER_LIMIT, 'action': action})
        quantity -= ORDER_LIMIT
    s.post('http://localhost:9999/v1/orders', params={'ticker': ticker, 'type': 'MARKET', 'quantity': quantity, 'action': action})

def flatten_positions(ticker_list):
    for ticker_symbol in ticker_list:
        position = int(get_ind_position(ticker_symbol))
        if position > 0:
            lrg_mkt_order(ticker_symbol, 'SELL', position)
        elif position < 0:
            lrg_mkt_order(ticker_symbol, 'BUY', abs(position))

async def place_bid(tender, tick):
    market_bid, market_ask = get_bid_ask(tender['ticker'])
    previous_tick = -1
    current_tick = tick
    print("PLACED BID CALLED, tick:", market_bid, market_ask, "tender action:", tender['action'], "tender price:", tender['price'])
    while current_tick <= tender['expires']:
        if previous_tick == current_tick:
            await asyncio.sleep(0)
            continue
        print(previous_tick, current_tick)
        previous_tick = current_tick
        current_tick, _ = get_tick()
        print("PLACED BID LOOP", current_tick, tender['expires'])
        print(previous_tick, current_tick)
        market_bid, market_ask = get_bid_ask(tender['ticker'])
        if market_bid is None or market_ask is None:
            await asyncio.sleep(0)
            continue
        if tender['action'] == 'SELL' and tender['price'] > market_bid:
            print("WE WANT TO SELL")
            accept_tender(tender)
        elif tender['action'] == 'BUY' and tender['price'] < market_ask:
            print("WE WANT TO BUY")
            accept_tender(tender)
        await asyncio.sleep(0)

async def main():
    tick, status = get_tick()
    ticker_list = [i['ticker'] for i in s.get('http://localhost:9999/v1/securities').json()]
    previous_tick = -1
    q = set()
    
    while status == 'ACTIVE':
        if previous_tick != tick:
            print("WHILE LOOP TICK:", previous_tick, tick)
            previous_tick = tick
            resp = s.get('http://localhost:9999/v1/tenders')
            if resp.ok:
                tenders = resp.json()
                for tender in tenders:
                    if (tender['tender_id'], tender['expires']) not in q and tender['is_fixed_bid'] == True:
                        asyncio.create_task(place_bid(tender, tick))
                        q.add((tender['tender_id'], tender['expires']))

            flatten_positions(ticker_list)
        await asyncio.sleep(0)
        # Flattens positions at every tick
        

        # for i in range(len(ticker_list)):
            
        #     ticker_symbol = ticker_list[i]
        #     position = get_position()
        #     best_bid_price, best_ask_price = get_bid_ask(ticker_symbol)

        #     # Skip if book is empty
        #     if best_bid_price is None or best_ask_price is None:
        #         continue

        #     if position < MAX_EXPOSURE:
        #         resp = s.post('http://localhost:9999/v1/orders', params = {'ticker': ticker_symbol, 'type': 'LIMIT', 'quantity': ORDER_LIMIT, 'price': best_bid_price, 'action': 'BUY'})
        #         resp = s.post('http://localhost:9999/v1/orders', params = {'ticker': ticker_symbol, 'type': 'LIMIT', 'quantity': ORDER_LIMIT, 'price': best_ask_price, 'action': 'SELL'})

        #     sleep(0.5) 

        #     s.post('http://localhost:9999/v1/commands/cancel', params = {'ticker': ticker_symbol})

        tick, status = get_tick()

if __name__ == '__main__':
    print("Ray's script is running")
    asyncio.run(main())
