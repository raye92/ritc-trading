"""RITC 2026 Algorithmic Market Making Trading Case - REST API Basic Script"""
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import requests
from time import sleep

s = requests.Session()
load_dotenv()
APIKEY = os.getenv("API_KEY")
s.headers.update({'X-API-key': APIKEY})


#Sample setup
MAX_EXPOSURE = 15000
ORDER_LIMIT = 10_000
MAX_TRADE_SIZE = 10_000
PRINT_HEART_BEAT= True

BASEURL ='http://localhost:9999/v1'


def get_tick():
    resp = s.get(BASEURL + '/case')
    if resp.ok:
        case = resp.json()
        return case['tick'], case['status']
    else:
        print(resp)
    

def get_bid_ask(ticker):
    payload = {'ticker': ticker}
    resp = s.get (BASEURL + '/securities/book', params = payload)
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
    resp = s.get (BASEURL + '/securities/tas', params = payload)
    if resp.ok:
        book = resp.json()
        time_sales_book = [item["quantity"] for item in book]
        return time_sales_book

def get_ind_position(ticker):
    payload = {'ticker': ticker}
    resp = s.get(BASEURL + '/securities', params=payload)
    if resp.ok:
        securities = resp.json()
        for security in securities:
            if security['ticker'] == ticker:
                return security['position']
    return 0  # Return 0 if ticker not found or request fails

def get_position():
    resp = s.get (BASEURL + '/securities')
    if resp.ok:
        book = resp.json()
        return abs(book[0]['position']) + abs(book[1]['position']) 

def get_book():
    resp = s.get(BASEURL + '/securities')
    if resp.ok:
        return resp.json()

def get_open_orders(ticker):
    payload = {'ticker': ticker}
    resp = s.get (BASEURL + '/orders', params = payload)
    if resp.ok:
        orders = resp.json()
        buy_orders = [item for item in orders if item["action"] == "BUY"]
        sell_orders = [item for item in orders if item["action"] == "SELL"]
        return buy_orders, sell_orders

def get_order_status(order_id):
    resp = s.get (BASEURL + '/orders' + '/' + str(order_id))
    if resp.ok:
        order = resp.json()
        return order['status']
    
def get_tenders():
    resp = s.get(BASEURL+ '/tenders')
    if resp.ok:
        return resp.json()
    
def place_mkt(ticker, action, qty):
    qty = int(max(1, min(qty, MAX_TRADE_SIZE)))
    r = s.post(f"{BASEURL}/orders",
               params={"ticker": ticker, "type": "MARKET",
                       "quantity": qty, "action": action})
    if PRINT_HEART_BEAT:
        print(f"ORDER {action} {qty} {ticker} -> {'OK' if r.ok else 'FAIL'}")
    return r.ok

def acceptTender(tenderID):
    r = s.post(f"{BASEURL}/tenders/{tenderID}", params= {"id": tenderID})
    return r.ok

def refuseTender(tenderID):
    r = s.delete(f"{BASEURL}/tenders/{tenderID}", params= {"id": tenderID})
    return r.ok

def placeBid(tenderID, bid):
    r = s.post(f"{BASEURL}/tenders/{tenderID}", params= {"id": tenderID, "price": bid})

def lrg_mkt_order(ticker, action, quantity, price):
    quantity = int(quantity)
    tmp = quantity
    #print("LARGE MARKET ORDER", ticker, action, quantity)
    if price is not None:
        attempts = 0
        while quantity > 0 and attempts < 100 * (quantity // ORDER_LIMIT):
            amt = min(quantity , ORDER_LIMIT)
            resp = s.post('http://localhost:9999/v1/orders', params={'ticker': ticker, 'type': 'MARKET', 'quantity': amt, 'action': action, 'price': price})
            if resp:
                quantity  -= resp.json()['quantity_filled']
            attempts += 1
        print("sold ", (tmp - quantity) / tmp ,"% at desired price ", quantity, int(get_ind_position(ticker)), " remaining")
    else:
        print(" no desried price")
    for i in range(int(quantity) // ORDER_LIMIT):
        s.post('http://localhost:9999/v1/orders', params={'ticker': ticker, 'type': 'MARKET', 'quantity': ORDER_LIMIT, 'action': action})
        quantity -= ORDER_LIMIT
    s.post('http://localhost:9999/v1/orders', params={'ticker': ticker, 'type': 'MARKET', 'quantity': quantity, 'action': action})
    print(int(get_ind_position(ticker)))

def flatten_positions(ticker_map):
    # while loop to wait to make sure we're selling at desired price when we want to
    for tkr, price in ticker_map.items():
        if price != None:
            while( int(get_ind_position(tkr)) == 0):
                wait = 1
        position = int(get_ind_position(tkr))
        if position > 0:
            lrg_mkt_order(tkr, 'SELL', position, price)
        elif position < 0:
            lrg_mkt_order(tkr, 'BUY', abs(position), price)

def main():
    tick, status = get_tick()
    ticker_list = [i['ticker'] for i in s.get(BASEURL+ '/securities').json()]
    securities = {i['ticker']: i for i in s.get(BASEURL+ '/securities').json()}
    desiredPrices = {i : None for i in ticker_list}
    usedTenders = set()

    # set up buffer amts
    securities['RITC']['buffer'] = 0.04 # -> low volititlity
    securities['COMP']['buffer'] = 0.14 # -> high volitiility
    
   
    previous_tick = -1
    last = -1
    while status == 'ACTIVE':
        if previous_tick != tick:
            # set up tender list 
            tenders = []
            tmp = get_tenders()
            for t in tmp:
                if t["tender_id"] in usedTenders:
                    continue
                else:
                    tenders.append(t)
            if len(tenders) != last:
                print(len(tenders))
                last = len(tenders)

            # check for potential value 
            for curr in tenders:
                caption = curr['caption'].split()
                marketBid, marketAsk = get_bid_ask(curr['ticker'])

                #case 1
                if curr['is_fixed_bid'] == True:
                    if curr['action'] == 'SELL':
                        if curr['price'] - securities[curr['ticker']]['trading_fee'] - securities[curr['ticker']]['buffer'] > marketBid:
                            print('placed case1, tick', tick)
                            acceptTender(curr['tender_id'])
                            usedTenders.add(curr['tender_id'])
                            desiredPrices[curr['ticker']] = curr['price']
                    elif curr['action'] == 'BUY':
                        if curr['price'] + securities[curr['ticker']]['trading_fee'] + securities[curr['ticker']]['buffer'] < marketAsk:
                            print('placed case1, tick', tick)
                            acceptTender(curr['tender_id'])
                            usedTenders.add(curr['tender_id'])
                            desiredPrices[curr['ticker']] = curr['price']
                #case 2
                elif caption[-1] == 'filled.':
                    if curr['action'] == 'SELL':
                        print('placed case 2, tick', tick)
                        b = marketBid + securities[curr['ticker']]['trading_fee'] + securities[curr['ticker']]['buffer']
                        placeBid(curr['tender_id'], b)
                        desiredPrices[curr['ticker']] = b
                        usedTenders.add(curr['tender_id'])
                        
                    elif curr['action'] == 'BUY':
                        print('placed case 2, tick', tick)
                        b = marketAsk - securities[curr['ticker']]['trading_fee'] - securities[curr['ticker']]['buffer']
                        placeBid(curr['tender_id'], b)
                        desiredPrices[curr['ticker']] = b
                        usedTenders.add(curr['tender_id'])
                #case 3
                # curr['expires'] means the tick number on which the tender expires
                else:
                    if curr['action'] == 'SELL' and  curr['expires']- tick  < 5:
                        print('placed case 3, tick', tick)
                        b = marketBid + securities[curr['ticker']]['trading_fee'] + securities[curr['ticker']]['buffer']
                        placeBid(curr['tender_id'], b)
                        desiredPrices[curr['ticker']] = b
                        usedTenders.add(curr['tender_id'])
                    elif curr['action'] == 'BUY' and curr['expires'] - tick  < 5  :
                        print('placed case 3, tick', tick)
                        b = marketAsk - securities[curr['ticker']]['trading_fee'] - securities[curr['ticker']]['buffer'] 
                        placeBid(curr['tender_id'], b)
                        desiredPrices[curr['ticker']] = b
                        usedTenders.add(curr['tender_id'])

            previous_tick = tick
            flatten_positions(desiredPrices)
            desiredPrices = {i : None for i in ticker_list}
                    
        tick, status = get_tick()

if __name__ == '__main__':
    main()
