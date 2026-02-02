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
APIKEY = os.getenv("API_KEY")
s.headers.update({'X-API-key': APIKEY})

#Sample setup
MAX_EXPOSURE = 15000
ORDER_LIMIT = 500
BASEURL ='http://localhost:9999/v1/case'

def get_tick():
    resp = s.get(BASEURL)
    if resp.ok:
        case = resp.json()
        return case['tick'], case['status']

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
    
def flattenPosiitons():
     resp = requests.post(BASEURL + '/tenders', params={"all": 1})


    
    

def main():
    tick, status = get_tick()
    ticker_list = [i['ticker'] for i in s.get(BASEURL+ '/securities').json()]

    while status == 'ACTIVE':     
        

        tick, status = get_tick()

if __name__ == '__main__':
    main()
