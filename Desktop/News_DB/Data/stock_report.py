#!/usr/bin/env python3
# -*- coding: big-5 -*-
"""
Created on Thu Jul 22 22:23:08 2021

@author: rubylintu
"""
import requests
import urllib
import csv
from bs4 import BeautifulSoup
import sys


def getPage(url):
    """
    Utilty function used to get a Beautiful Soup object from a given URL
    """

    session = requests.Session()
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}
    try:
        req = session.get(url, headers=headers)
    except requests.exceptions.RequestException:
        return None
    bs = BeautifulSoup(req.text, 'html.parser')
    return bs

def scrape_stock_goodinfo(url):
    bs = getPage(url)
    title = bs.find('h1').text
    lines = bs.select('div.StoryBodyCompanionColumn div p')
    body = '\n'.join([line.text for line in lines])
    return url, title, body


#num=str(sys.argv[1])
num='4915'
url='https://goodinfo.tw/StockInfo/BasicInfo.asp?STOCK_ID=' + num

bs=getPage(url)

#url, title, body = scrape_stock_goodinfo(url)
