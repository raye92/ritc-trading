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
commissions = {
    "RITC": 0.02, 
    "COMP": 0.02, 
    "TRNT": 0.01, 
    "MTRL": 0.01, 
    "BLU": 0.04, 
    "RED": 0.03, 
    "GRN": 0.02, 
    "WDY": 0.02, 
    "BZZ": 0.02, 
    "BNN": 0.03, 
    "VNS": 0.02, 
    "MRS": 0.02, 
    "JPTR": 0.02, 
    "STRN": 0.02
}

volatility = {
"RITC": "Low",
"COMP": "Medium",
"TRNT": "High",
"MTRL": "Low",
"BLU":  "High",
"RED":  "Low",
"GRN":  "Medium",
"WDY":  "Medium",
"BZZ":  "High",
"BNN":  "Medium",
"VNS":  "High",
"MRS":  "Medium",
"JPTR": "Low",
"STRN": "High"
}

liquidity = {
    "RITC": "Medium",
    "COMP": "High",
    "TRNT": "Medium",
    "MTRL": "Low",
    "BLU":  "High",
    "RED":  "Medium",
    "GRN":  "Medium",
    "WDY":  "High",
    "BZZ":  "Medium",
    "BNN":  "Medium",
    "VNS":  "Medium",
    "MRS":  "High",
    "JPTR": "Medium",
    "STRN": "Medium"
}

volatility_evs = {
    "Low": 0.00,
    "Medium": 0.08,
    "High": 0.25
}

liquidity_evs = {
    "Low": 0.05,
    "Medium": 0.03,
    "High": 0.00
}


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
    market_bid, market_ask = get_bid_ask(ticker)
    print("LARGE MARKET ORDER", ticker, action, quantity, market_bid, market_ask)
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

async def place_bid(tender, tick, cost):
    market_bid, market_ask = get_bid_ask(tender['ticker'])
    previous_tick = -1
    current_tick = tick
    while current_tick <= tender['expires']:
        if previous_tick == current_tick:
            await asyncio.sleep(0)
            continue
        previous_tick = current_tick
        current_tick, _ = get_tick()
        market_bid, market_ask = get_bid_ask(tender['ticker'])
        print(tender['action'], tender['price'], market_bid, market_ask, cost)
        diff = 0
        if tender['action'] == 'SELL':
            diff = tender['price'] - market_ask
        elif tender['action'] == 'BUY':
            diff = market_bid - tender['price']
        print("DIFF:", diff, "COSTS:", cost)
        if diff >  cost:
            print("WE WANT TO", tender['action'], "AT", tender['price'])
            accept_tender(tender)
        else:
            print("WE DON'T WANT TO DO ANYTHING")
        await asyncio.sleep(0)

async def place_tender(tender, cost):
    current_tick, _ = get_tick()
    while current_tick < tender['expires'] - 1:
        current_tick, _ = get_tick()
        await asyncio.sleep(0)

    market_bid, market_ask = get_bid_ask(tender['ticker'])

    if tender['action'] == 'SELL':
        price = market_ask + cost
    elif tender['action'] == 'BUY':
        price = market_bid - cost
    print("PRICE:", price, "Tick:", current_tick)
    s.post(f'http://localhost:9999/v1/tenders/{tender["tender_id"]}', params={'price': price})

async def main():
    tick, status = get_tick()
    ticker_list = [i['ticker'] for i in s.get('http://localhost:9999/v1/securities').json()]
    previous_tick = -1
    q = set()
    
    while status == 'ACTIVE':
        if previous_tick != tick:
            previous_tick = tick
            resp = s.get('http://localhost:9999/v1/tenders')
            if resp.ok:
                tenders = resp.json()
                for tender in tenders:
                    cost = commissions[tender['ticker']] + liquidity_evs[liquidity[tender['ticker']]] + volatility_evs[volatility[tender['ticker']]]
                    if (tender['tender_id'], tender['expires']) not in q and tender['is_fixed_bid'] == True:
                        print("CASE 1")
                        print(tender)
                        asyncio.create_task(place_bid(tender, tick, cost))
                        q.add((tender['tender_id'], tender['expires']))
                    elif (tender['tender_id'], tender['expires']) not in q and tender['is_fixed_bid'] == False:
                        print("CASE 2/3")
                        print(tender)
                        asyncio.create_task(place_tender(tender, cost))
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
