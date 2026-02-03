"""RITC 2026 Algorithmic Market Making Trading Case - REST API Basic Script"""

import requests
from time import sleep
import os
from dotenv import load_dotenv

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

def main():
    tick, status = get_tick()
    ticker_list = [i['ticker'] for i in s.get('http://localhost:9999/v1/securities').json()]
    previous_ticker = -1
    while status == 'ACTIVE':     
        if previous_ticker != tick:
            previous_ticker = tick
            print(f"TICK NUMBER: {tick}")
            for ticker_symbol in ticker_list:
                position = int(get_ind_position(ticker_symbol))
                print(f"Ticker symbol and position: {ticker_symbol}: {position}")
                if position > 0:
                    for i in range(position // ORDER_LIMIT):
                        s.post('http://localhost:9999/v1/orders', params={'ticker': ticker_symbol, 'type': 'MARKET', 'quantity': ORDER_LIMIT, 'action': 'SELL'})
                        print(f"Sold {ORDER_LIMIT} shares of {ticker_symbol}")
                        print(position)
                        position -= ORDER_LIMIT
                        print(position)
                    s.post('http://localhost:9999/v1/orders', params={'ticker': ticker_symbol, 'type': 'MARKET', 'quantity': position, 'action': 'SELL'})
                elif position < 0:
                    for i in range(abs(position) // ORDER_LIMIT):
                        s.post('http://localhost:9999/v1/orders', params={'ticker': ticker_symbol, 'type': 'MARKET', 'quantity': ORDER_LIMIT, 'action': 'BUY'})
                        print(f"Bought {ORDER_LIMIT} shares of {ticker_symbol}")
                        print(position)
                        position += ORDER_LIMIT
                        print(position)
                    s.post('http://localhost:9999/v1/orders', params={'ticker': ticker_symbol, 'type': 'MARKET', 'quantity': abs(position), 'action': 'BUY'})

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
    main()
