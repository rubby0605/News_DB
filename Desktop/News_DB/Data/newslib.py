#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 19:32:54 2021

@author: rubylintu
"""
from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen
from urllib.parse import urlparse
import datetime
import random
import re
from gtts import gTTS
import time
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import math
import smtplib
import time 

def read_stock_price(num, datestr_list, stock):
    #df = stock.read_stock_price(2330, ['220323'], '台積電')
    stock_price = np.zeros([10000,1])
    stock_price_est = np.zeros([10000,1])
    array=np.zeros([1000,2])
    total = 0
    amount = 0
    pre_price = 0
    oldmat4 = 0
    init_i = datetime.timedelta(0)
    deltatt = 0
    dateid=0
    df = pd.DataFrame(array)
    for datestr in datestr_list:
        with open(str(num)+'_'+datestr + '.txt') as f:
            lines = f.readlines()
        i=0
        for line in lines:
            mat = re.split(r'\t|\d{4}-| |:|\n', line)
            if len(mat) < 13 or int(num) != int(mat[1]):
                continue
            [mm,dd]=re.split('-',mat[10])
            if '-' not in mat[3]:
                price = float(mat[3])
                df.iloc[i,1] = price
                df.iloc[i,0] = (float(mat[11])-9)*60+float(mat[12])
                i = i + 1

    return df
def get_stock_price(num, datestr_list, stock):
    #stock_price, stock_price_est = stock.get_stock_price(2330, ['220323'], '台積電')
    MUX = 5
    MUY = 1
    SGM_X = 5 # average deal earning
    SGM_Y = 2 # deal amount per person  
    stock_price = np.zeros([10000,1])
    stock_price_est = np.zeros([10000,1])
    columns = ['c','n','z','tv','v','o','h','l','y']
    dict_stock = read_stock_list('stock_list_less.txt') #input file 
    stock_list_str = dict_stock.keys()
    stock_list = [int(dict_stock[stock]) for stock in stock_list_str]
    
    plt.figure(figsize=((10,8)))
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    num = dict_stock[stock]
    total = 0
    amount = 0
    pre_price = 0
    oldmat4 = 0
    init_i = datetime.timedelta(0)
    deltatt = 0
    dateid=0
    for datestr in datestr_list:
        with open(str(num)+'_'+datestr + '.txt') as f:
            lines = f.readlines()
        print(total)
        i=0
        for line in lines:
            mat = re.split(r'\t|\d{4}-| |:|\n', line)
            if len(mat) < 13 or int(num) != int(mat[1]):
                continue
            [mm,dd]=re.split('-',mat[10])
            if i == 0:
                a = datetime.datetime(2022, int(mm), int(dd), int(mat[11]), int(mat[12]), int(float(mat[13])))
            if '-' not in mat[3]:
                price = float(mat[3])
            else:
                if pre_price == 0:
                    if '-' not in mat[6]:
                        price = float(mat[6])
                    else:
                        price = 0
                else:
                    price = pre_price
            if "-" in mat[4] or price==0:
                dn = 0
            else:
                if mat[4] == oldmat4:
                    dn = 0
                else:
                    dn = int(mat[4])
            if price != 0:
                if i!=0:
                    oldbuffer2 = digest_oldbuffer(oldbuffer, price, 0.2)
                    oldbuffer = np.zeros([10000,1])
                if i == 0:
                    buffer = np.zeros(dn)
                    oldbuffer = np.zeros([10000,1])
                amount = amount + dn
                total = total + price * float(dn)
            #total = total * decay
            #amount = amount * decay
            b= datetime.datetime(2022, int(mm), int(dd), int(mat[11]), int(mat[12]), int(float(mat[13])))
            time_delta = b - a
            tt = time_delta.total_seconds() / 5
            stock_price[i] = price
            if amount == 0:
                stock_price_est[i] = 0
            else:
                stock_price_est[i] = total/ amount
            StopIteration
            print(amount)
            plt.plot(tt+deltatt, price,"k.")
            if amount !=0:
                plt.plot(tt+deltatt, total/amount,"r.")
            if '-' in mat[4]:
                oldmat4 = 0
            else:
                oldmat4 = mat[4]
            pre_price = price
            i = i + 1
        deltatt = deltatt + 4.5*3600 / 5
    return stock_price[0:i], stock_price_est[0:i]
#          ['c',   'n',       'z',      'tv' ,         'v',      'o',    'h',    'l',    'y'] 
#分別代表 ['股票代號','公司簡稱','當盤成交價','當盤成交量','累積成交量','開盤價','最高價','最低價','昨收價']
def find_duration_delta(year,mm,dd, duration):
    a= datetime.datetime(year, mm, dd)
    delta = datetime.timedelta(days=duration)
    b = a + delta
    return b.year, b.month, b.day
def is_stock_abs_low(num):
    twst = twstock.Stock(str(num))
    td = datetime.date.today()
    d7_y, d7_m, d7_d = find_duration_delta(td.year,td.month,td.day, -7)
    d30_y, d30_m, d30_d  = find_duration_delta(td.year,td.month,td.day, -30)
    d80_y, d80_m, d80_d = find_duration_delta(td.year,td.month,td.day, -80)
    st7 = twst.fetch_from(d7_y, d7_m-1)
    st30 = twst.fetch_from(d30_y, d30_m-1)
    st80 = twst.fetch_from(d80_y, d80_m-1)
    n7day = int(np.size(st7) / np.size(st7[0]))
    price = st7[n7day-1][3]
    n7_max = np.amax([st7[i][3] for i in range(n7day)])
    n7_min = np.amin([st7[i][3] for i in range(n7day)])
    n30day = int(np.size(st30) / np.size(st30[0]))
    n30_max = np.amax([st30[i][3] for i in range(n30day)])
    n30_min = np.amin([st30[i][3] for i in range(n30day)])
    n80day = int(np.size(st80) / np.size(st80[0]))
    n80_max = np.amax([st80[i][3] for i in range(n80day)])
    n80_min = np.amin([st80[i][3] for i in range(n80day)])
    output = 0
    if (n7_max - n7_min)*0.15 + n7_min > price:
        if (n30_max - n30_min)*0.15 + n30_min > price:
            if (n80_max - n80_min)*0.15 + n80_min > price:
                output = 1
    return output

def digest_oldbuffer(buffer, price, delta):
    i=0
    if len(buffer) == 0:
        return buffer
    for ele in buffer:
        if abs(ele-price) <= delta:
            buffer[i]=0
        i = i + 1
    return buffer
def combine_buffers(buffer,oldbuffer):
    if len(buffer)== 0:
        return oldbuffer
    j=0
    for i in range(1000):
        if oldbuffer[i] == 0:
            oldbuffer[i] = buffer[j]
            j = j +1
            if j == len(buffer):
                break
    return oldbuffer

def scrapeBondInfo(website):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}
    website2 = re.sub("http:","https:",website)
    response=requests.get(website2, headers=headers)
    bs = BeautifulSoup(response.text, 'html.parser')
    title = bs.find('title')
    body = bs.text
    mat=re.search("最新轉\(交\)換價格.+[0-9]+", body)
    current_price = float(re.sub("最新轉\(交\)換價格：","",mat.group()))
    mat=re.search("轉換溢價率.+[0-9]+", body)
    rate = float(re.sub("轉換溢價率：","",mat.group()))
    return bs, body, current_price, rate
   
def read_stock_daily(num, datestr):
    #日期，成交股數，成交金額，開盤價，最高價，最低價，收盤價，漲跌價差，成交筆數
    url2 = 'http://www.twse.com.tw/exchangeReport/STOCK_DAY?date='+datestr+'&stockNo=' + str(num) + '&type=ALL'
    req = requests.get(url2)
    data=req.json()
    if '很抱歉' in data['stat']:
        return -999, -999, -999, -999
    else:
        table = data['data']
        df = pd.DataFrame(table)
        return df, table, df.iloc[19,6], df.iloc[19,8]

def read_stock_list(filename):
    #filename = "stock_list.txt"
    with open(filename, 'r') as f:
        lines = f.readlines()
    dict_stock = {}
    for line in lines:
        mat = re.split(' |\t|\n',line);
        if len(mat)<=2:
            continue
        else:
            dict_stock[mat[1]] = mat[0]
    return dict_stock
def read_stock_fulllist_id0(filename):
    #filename = "stock_list.txt"
    with open(filename, 'r') as f:
        lines = f.readlines()
    dict_stock = {}
    for line in lines:
        mat = re.split(' |\t|\n',line);
        if len(mat)<=2:
            continue
        else:
            dict_stock[mat[0]] = 0
    return dict_stock

def read_stock_fulllist(filename):
    #filename = "stock_list.txt"
    with open(filename, 'r') as f:
        lines = f.readlines()
    dict_stock = {}
    for line in lines:
        mat = re.split(' |\t|\n',line);
        if len(mat)<=2:
            continue
        else:
            dict_stock[mat[3]] = mat[0]
    return dict_stock
OTC_STOCKS = {'8299'}

def craw_realtime(stock_number):
    if len(stock_number) > 1:
        stock_list = '|'.join(
            'otc_{}.tw'.format(target) if str(target) in OTC_STOCKS
            else 'tse_{}.tw'.format(target)
            for target in stock_number
        )
    else:
        stock_list = str(stock_number)
    url = (
        "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch="+
        stock_list
    )
    # ['c','n','z','tv','v','o','h','l','y']
    #分別代表 ['股票代號','公司簡稱','當盤成交價','當盤成交量','累積成交量','開盤價','最高價','最低價','昨收價']
    data = json.loads(urlopen(url).read())
    return data

"""This cell defineds the plot_candles function"""

def plot_candles(pricing, title=None, volume_bars=False, color_function=None, technicals=None):
    """Plots a candlestick chart using quantopian pricing data.
    
    Author: Daniel Treiman
    
    Args:
      pricing: A pandas dataframe with columns ['open_price', 'close_price', 'high', 'low', 'volume']
      title: An optional title for the chart
      volume_bars: If True, plots volume bars
      color_function: A function which, given a row index and price series, returns a candle color.
      technicals: A list of additional data series to add to the chart.  Must be the same length as pricing.
    """
    def default_color(index, open_price, close_price, low, high):
        return 'r' if open_price[index] > close_price[index] else 'g'
    color_function = color_function or default_color
    technicals = technicals or []
    open_price = pricing['open_price']
    close_price = pricing['close_price']
    low = pricing['low']
    high = pricing['high']
    oc_min = pd.concat([open_price, close_price], axis=1).min(axis=1)
    oc_max = pd.concat([open_price, close_price], axis=1).max(axis=1)
    
    if volume_bars:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3,1]},figsize=(7,7))
    else:
        fig, ax1 = plt.subplots(1, 1)
    if title:
        ax1.set_title(title)
    fig.tight_layout()
    x = np.arange(len(pricing))
    candle_colors = [color_function(i, open_price, close_price, low, high) for i in x]
    candles = ax1.bar(x, oc_max-oc_min, bottom=oc_min, color=candle_colors, linewidth=0)
    lines = ax1.vlines(x , low, high, color=candle_colors, linewidth=1)
    ax1.xaxis.grid(True)
    ax1.yaxis.grid(True)
    ax1.xaxis.set_tick_params(which='major', length=3.0, direction='in', top='off')
    ax1.set_yticklabels([])
    # Assume minute frequency if first two bars are in the same day.
    frequency = 'minute' if (pricing.index[1] - pricing.index[0]) == 0 else 'day'
    time_format = '%d-%m-%Y'
    if frequency == 'minute':
        time_format = '%H:%M'
    # Set X axis tick labels.
    #plt.xticks(x, [date.strftime(time_format) for date in pricing.index], rotation='vertical')
    for indicator in technicals:
        ax1.plot(x, indicator)
    
    if volume_bars:
        volume = pricing['volume']
        volume_scale = None
        scaled_volume = volume
        if volume.max() > 1000000:
            volume_scale = 'M'
            scaled_volume = volume / 1000000
        elif volume.max() > 1000:
            volume_scale = 'K'
            scaled_volume = volume / 1000
        ax2.bar(x, scaled_volume, color=candle_colors)
        volume_title = 'Volume'
        if volume_scale:
            volume_title = 'Volume (%s)' % volume_scale
        #ax2.set_title(volume_title)
        ax2.xaxis.grid(True)
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.title(str(stocknum))
    return fig 


    
def craw_one_month(stock_number,date):
    stock_list = str(stock_number)
    url = (#"日期","成交股數","成交金額","開盤價","最高價","最低價","收盤價","漲跌價差","成交筆數"
        "http://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date="+
        date+"&stockNo="+
        stock_list
    )
    data = json.loads(urlopen(url).read())
    return data#pd.DataFrame(data['data'],columns=data['fields'])
def get_index_wd_fullmonth(stock_number):
    datemat = re.split('\-| |\:|\.',str(datetime.datetime.today()))
    backdate = datemat.copy()
    if datemat[2] == '31':
        backdate[2] = '30'
    if datemat[1] == '1':
        backdate[1] = '12'
        backdate[0] = str(int(backdate[0]) - 1)
    else:
        if int(backdate[1]) < 11:
            backdate[1] = '0' + str(int(backdate[1]) - 1)
        else:
            backdate[1] = str(int(backdate[1]) - 1)
            
    datestring0 = backdate[0]+backdate[1]+backdate[2]
    datestring1 = datemat[0]+datemat[1]+datemat[2]
    
    data0 = craw_one_month(stock_number,datestring0)
    data1 = craw_one_month(stock_number,datestring1)
    
    if '很抱歉' in data0['stat']:
        return 0
    df0 = pd.DataFrame(data0['data'][0:-1], columns=['date','volume','price','open_price', 'high','low', 'close_price','diff','number' ])
    df1 = pd.DataFrame(data1['data'][0:-1], columns=['date','volume','price','open_price', 'high','low', 'close_price','diff','number' ])
    df = pd.concat([df0,df1],ignore_index=True)
    df['open_price'] = df['open_price'].apply(lambda x: x.replace(',',''))
    df['volume'] = df['volume'].apply(lambda x: x.replace(',',''))
    df['price'] = df['price'].apply(lambda x: x.replace(',',''))
    df['high'] = df['high'].apply(lambda x: x.replace(',',''))
    df['low'] = df['low'].apply(lambda x: x.replace(',',''))
    df['close_price'] = df['close_price'].apply(lambda x: x.replace(',',''))
    df['diff'] = df['diff'].apply(lambda x: x.replace('X0.00','0'))
    df['diff'] = df['diff'].astype(float)
    df['number'] = df['number'].apply(lambda x: x.replace(',',''))
    df['number'] = df['number'].astype(float)
    #df[3][:] = df[3][:].astype(float)
    #df[4][:] = df[4][:].astype(float)
    #df[5][:] = df[5][:].astype(float)
    #df[6][:] = df[6][:].astype(float)
    return df

def get_K(df):
    
    return df

def get_index_wd_onemonth(stock_number):
    datemat = re.split('\-| |\:|\.',str(datetime.datetime.today()))
    datestring = datemat[0]+datemat[1]+datemat[2]
    data = craw_one_month(stock_number,datestring)
    
    if '很抱歉' in data['stat']:
        return 0
    df = pd.DataFrame(data['data'],columns=data['fields'])
    size_df = len(df)
    for i in range(size_df-1,0,-1):
        index_wd = get_index_wd(df,i)
        if(index_wd):
            print(df.iloc[i,0])
    return get_index_wd(df,size_df-1)
    
def get_index_wd(df,i):
    test = 0
    price = float(re.sub(',','',df.iloc[i,6]))
    upp = float(re.sub(',','',df.iloc[i,4]))
    lwp = float(re.sub(',','',df.iloc[i,5]))
    bgp = float(re.sub(',','',df.iloc[i,3]))
    if abs(price - bgp) <=price/300 and upp-lwp >=price/120:
        return math.exp(-1/price*abs(price - bgp) ) * math.exp(-1/(upp-lwp)*price)
    else:
        return 0
def get_stock_capital():
    url = 'https://stock.wespai.com/p/16429'
    bs = getPage(url)
    
    
def getGoodInfo2(num):
    url = 'https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID=' + str(num)
    bs = getPage(url)
    if bs.find('table',{"class":"b0 p4_0"}) is None:
        capital=-1000
        industry="-1000"
        return capital, industry
    cnt_capital = bs.find('table',{"class":"b0 p4_0"}).find_next('table',{"class":"b0 p4_0"}).find('nobr').find_next('nobr').text
    str0=re.search('[0-9]*,*[0-9]+',cnt_capital)
    capital = -999
    industry="-999"
    if str0 is not None:
        print(str0)
        if ',' in str0.group(0):
            re.sub(',','',str0.group(0))
            capital = re.sub(',','',str0.group(0))
            industry = bs.find('table',{"class":"b1 p4_4 r10"}).find('td',{"bgcolor":"white"}).find_next('td',{"bgcolor":"white"}).text
        else:            
            capital = re.sub(',','',str0.group(0))
            industry = bs.find('table',{"class":"b1 p4_4 r10"}).find('td',{"bgcolor":"white"}).find_next('td',{"bgcolor":"white"}).text
    else:
        capital=-1000
        industry="-1000"
    return capital, industry
    
def getGoodInfo(num):
    url = 'https://goodinfo.tw/StockInfo/StockDetail.asp?STOCK_ID=' + str(num)
    bs = getPage(url)
    i=0
    price=-999
    dealnum=-999
    EPS=-999
    delEPS=-999
    netrate=-999
    #收盤價，交易量，EPS，ＥＰＳ差直，毛利率
    fullt= bs.find('table',{"class":"b1 p4_0 r0_10 row_bg_2n row_mouse_over","style":"width:100%;font-size:10pt;line-height:17px;"})
    if fullt==None:
        return price, dealnum, EPS, delEPS, netrate
    for line in fullt.find_all('tr'):
        i = i + 1
        if(i==3):
            j = 0
            for element in line.find_all('td'):
                j = j + 1
                if j == 8:
                    if '-' in element.string:
                        EPS = -999
                    else:
                        EPS=float(element.string)
                elif j == 4:
                    if '-' in element.string:
                        netrate = -999
                    else:
                        netrate=float(element.string)
    fullt = bs.find('table',{"class":"b1 p4_2 r10", "style":"width:100%;font-size:11pt;"})       
    line = fullt.find("tr",{"align":"center"})
    price = float(line.td.string)
    i=0
    for line in fullt.find_all('tr'):
        i=i+1
        if(i==6):
            mat=re.sub(',','',line.td.string)
            dealnum = mat
            print(dealnum)
        i = i + 1
                
    return price, dealnum, EPS, delEPS, netrate

def getPage(url):
    """
    Utilty function used to get a Beautiful Soup object from a given URL
    """
    parsed = urlparse(url)
    headers = {
    'host': parsed.netloc,
    'method': 'GET',
    'referer': 'https://www.google.com/',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}
    cookies = dict(uuid='b18f0e70-8705-470d-bc4b-09a8da617e15',UM_distinctid='15d188be71d50-013c49b12ec14a-3f73035d-100200-15d188be71ffd')

    try:
        response=requests.get(url, headers=headers, cookies=cookies, timeout=15)
        response.encoding='utf-8'
        bs = BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException:
        return None
    return bs
def getMeet(url):
    """
    Utilty function used to get a Beautiful Soup object from a given URL
    """
    headers = {
    'host': 'www.google.co.kr',
    'method': 'GET',
    'referer': 'https://www.google.com/',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/27.0.1453.93 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}
    proxies = {'http':'http://10.10.10.10:8765','https':'https://10.10.10.10:8765'}
    cookies = dict(uuid='__RequestVerificationToken	an3nsWQ04rvCAz2uhI_G0Bz2BDc_veKb6x-',UM_distinctid='15d188be71d50-013c49b12ec14a-3f73035d-100200-15d188be71ffd')
    #res = requests.get(url, headers=headers, cookies=cookies)
    
    try:
        response=requests.get(url, headers=headers, cookies=cookies)
        response.encoding='utf-8'
        bs = BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException:
        return None
    return bs
def scrapeHiStock(num):
    url='https://histock.tw/stock/'+str(num)+'/營業利益成長率'
    bs = getPage(url)
    return bs

def scrapeHiStock(num):
    url='https://pchome.megatime.com.tw/stock/sto3/sid'+str(num)+'.html'
    bs = getPage(url)
    
    return bs

def scrapeNYTimes(url):
    bs = getPage(url)
    title = bs.find('h1').text
    lines = bs.select('div.StoryBodyCompanionColumn div p')
    body = '\n'.join([line.text for line in lines])
    return Content(url, title, body)

def scrapeBrookings(url):
    bs = getPage(url)
    title = bs.find('h1').text
    body = bs.find('div', {'class', 'post-body'}).text
    return Content(url, title, body)

def scrapeNews(url):
    bs = getPage(url)
    title = bs.find('div').title
    body = bs.find('div', {'class', 'title'})
    return Content(url, title, body)

def scrapBingNews(keyword):
    url = r"https://www.bing.com/news/search?q=%22" + keyword + "%22&go=搜尋&qs=n&form=QBNT&sp=-1&pq=%22" + keyword
    bs = getPage(url)
    if bs is None:
        return url, None, '', None
    title = bs.find('title')
    body = bs.text
    return url, title, body, bs

def scrapGoogleNews(keyword):
    url="https://www.google.com/search?q="+keyword+"&tbm=nws&start=%d"
    bs = getPage(url)
    if bs is None:
        return url, None, '', None
    title = bs.find('title')
    body = bs.text
    return url, title, body, bs

def getGoogleBaInfo(str0):
    url="https://www.google.com/search?q="+str0+"+發行價&start=%d"
    response=requests.get(url)
    bs = getPage(url)
    title = bs.find('title')
    body = bs.text
    return url, title, body, bs


def gettpexCBInfo(num):
    url="https://www.tpex.org.tw/web/bond/publish/convertible_bond_search/memo.php?l=zh-tw"
    response=requests.get(url)
    bs = getPage(url)
    title = bs.find('title')
    body = bs.text
    return url, title, body, bs

def str_td(str0):
    return "<td>"+str0+"</td>"

def str_tr(str0):
    return "<tr>"+str0+"</tr>"

def getGoodStockInfo(num):
    url="https://goodinfo.tw/StockInfo/StockDetail.asp?STOCK_ID="+str(num)
    response=requests.get(url)
    response.encoding = "utf-8"
    bs = getPage(url)
    title = bs.find('title')
    body = bs.text
    return url, title, body


def scrapeGoogleStockInfo(num):
    url="https://www.google.com/search?q="+str(num)+"股價&start=%d"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}
    response=requests.get(url, headers=headers)
    bs = getPage(url)
    title = bs.find('title')
    body = bs.text
    str_stockprice=re.search("股價[0-9]+",body)
    mat = re.split("股價",str_stockprice.group(0))
    stockprice = str(mat[1])
    url="https://www.google.com/search?q="+str(num)+"成交量&start=%d"
    response=requests.get(url)
    bs = getPage(url)
    title = bs.find('title')
    body = bs.text
    str_deal=re.search("[0-9]成交量",body)
    mat = re.split("成交量",str_deal.group(0))
    deal = str(mat[0])
    return deal, stockprice




    
def scrape_stockclub(num):
    if(num == -999):
        num = 3896602#5769561 #5794329
    url="https://www.cmoney.tw/follow/channel/articles-" + str(num) + "#/"
    response=requests.get(url)
    bs = getPage(url)
    title = bs.find('title')
    body = re.sub("\n","",bs.text)
    #data = json.loads(body)
    return url, title, bs


def get_stock_info(num):
    headers = {
    'host': 'mis.twse.com.tw',
    'method': 'GET',
    'referer': 'https://mis.twse.com.tw/',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    cookies = dict(uuid='b18f0e70-8705-470d-bc4b-09a8da617e15',UM_distinctid='15d188be71d50-013c49b12ec14a-3f73035d-100200-15d188be71ffd')

    # 先試上市 (tse_)，若 msgArray 為空則試上櫃 (otc_)
    for prefix in ['tse_', 'otc_']:
        url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={prefix}{num}.tw"
        res = requests.get(url, headers=headers, cookies=cookies, timeout=15)
        data = res.json()
        data2 = data.get('msgArray', [])
        if data2 and data2[0].get('c') and 'n' in data2[0]:
            break

    if not data2 or not data2[0].get('c') or 'n' not in data2[0]:
        return url, []

    dict2=data2[0]
    num=dict2['c']
    name = dict2['n']
    deal = dict2['z']
    dealamount = dict2['tv']
    wholedealamount = dict2['v']
    startprice = dict2['o']
    highestprice = dict2['h']
    lowestprice = dict2['l']
    yestprice = dict2['y']
    data3 = [num, name, deal, dealamount, wholedealamount, startprice, highestprice, lowestprice, yestprice]
    return url, data3


class Website:
    """ 
    Contains information about website structure
    """

    def __init__(self, name, url, titleTag, bodyTag):
        self.name = name
        self.url = url
        self.titleTag = titleTag
        self.bodyTag = bodyTag

#data clean
def getNewsDataClean(full_text, content_dict, str_datetime, bs):
    contents = re.sub(r'([a-zA-Z0-9]*新聞[a-zA-Z0-9]*)','AAAAA',full_text)
    #data split
    delimiters = "a", "...", "(C)", "(R)"
    regexPattern = '|'.join(map(re.escape, delimiters)) # 'a|\\.\\.\\.|\\(C\\)'
    content = re.split(regexPattern, contents) # 
    #data record into dictionary
    regexp = re.compile(r'(.+AAAAA.+)')
    regexc = re.compile(r'(.+AAAAA.+)')
    for line in content: 
        fields0 = re.sub(r'(([a-zA-Z]+\.)+|\.){2,4}','', line)
        fields1 = re.sub(r'.+(新闻|新聞|News|\n+).+',' ',fields0);
        if(len(fields1)>=20 and not regexp.search(fields1)):
            if fields1 in content_dict.keys():
              print("skip")
            else:
                content_dict[fields1]=str(datetime.datetime.now());
    str_full_text = str(content_dict)
    
    return str_full_text, content_dict
def getNewsDate(full_text, content_dict, str_datetime, bs):
    contents = re.sub(r'([a-zA-Z0-9]*新聞[a-zA-Z0-9]*)','AAAAA',full_text)
    #data split
    delimiters = "a", "...", "(C)", "(R)"
    regexPattern = '|'.join(map(re.escape, delimiters)) # 'a|\\.\\.\\.|\\(C\\)'
    content = re.split(regexPattern, contents) # 
    #data record into dictionary
    regexp = re.compile(r'(.+AAAAA.+)')
    regexc = re.compile(r'(.+AAAAA.+)')
    for line in content: 
        fields0 = re.sub(r'(([a-zA-Z]+\.)+|\.){2,4}','', line)
        fields1 = re.sub(r'.+(新闻|新聞|News|\n+).+',' ',fields0);
        if(len(fields1)>=20 and not regexp.search(fields1)):
            if fields1 in content_dict.keys():
              print("skip")
            else:
                aa = bs.find(text = re.compile(fields1))
                if(aa==None):
                    content_dict[fields1]=str(datetime.datetime.now());
                    continue
                columns = aa.parent.parent.parent
                url2 = re.sub('\/url\?q=','', columns.get('href'))
                bs_text = getPage(url2)
                aa=re.search(r"[0-9]{4}[-\.][0-9]{1,2}[-\/.][0-9]{1,2}",bs_text.text)
                datestring0 = aa.group()
                datestring1 = re.split('[-\.]',datestring0)
                if(len(datestring1) == 3):
                    yy=datestring1[0]
                    if(yy==3):
                        yy = str(int(yy)+1911)
                    mm = datestring1[1]
                    if(len(mm)==1):
                        mm = "0" + mm
                    dd = datestring1[2]
                    if(len(dd)==1):
                        dd= "0" + dd
                    datestring = yy+mm+dd
                else:
                    datestring = datestring1 
                content_dict[fields1]=datestring#str(datetime.datetime.now());
    str_full_text = str(content_dict)
    
    return str_full_text, content_dict

def get_datetime():
    str_now = str(datetime.datetime.now())
    return str_now

def save_files(str_full_text, content_dict, basepath, str_now, keyword):
    str_now2 = re.sub(r'[-|: ]','_',str_now)
    str_now3 = re.sub(r'_[0-9]+_[0-9]+.[0-9]+$','00',str_now2)
    str_now = str_now3
    #save csv file as $date.csv
    csv_file_name = basepath + str_now + '_' + keyword + '.csv'
    
    print(csv_file_name)
    print(content_dict)
    with open(csv_file_name, 'w') as f:
        for key in content_dict.keys():
            f.write("%s,%s\n"%(key,content_dict[key]))
    tts=gTTS(text=str_full_text, lang='zh', slow=False)
    radio_filename = basepath + str_now + '_' + keyword + '.mp3'
    print(radio_filename)
    tts.save(radio_filename)
    return 1
def save_mp3(str_full_text, content_dict, keyword):
    tts=gTTS(text=str_full_text, lang='zh', slow=False)
    radio_filename = 'html/' + keyword + '.mp3'
    print(radio_filename)
    tts.save(radio_filename)
    return 1


def save_csv_file(str_full_text, content_dict, basepath, str_now, keyword):
    str_now2 = re.sub(r'[-|: ]','_',str_now)
    str_now3 = re.sub(r'_[0-9]+_[0-9]+.[0-9]+$','00',str_now2)
    str_now = str_now3
    #save csv file as $date.csv
    csv_file_name = basepath + str_now + '_' + keyword + '.csv'
    
    print(csv_file_name)
    print(content_dict)
    with open(csv_file_name, 'w') as f:
        for key in content_dict.keys():
            f.write("%s,%s\n"%(key,content_dict[key]))
    f.close()
    return 1
def create_filename(keyword):
    str_now = get_datetime()
    str_now2 = re.sub(r'[-|: ]','_',str_now)
    str_now3 = re.sub(r'_[0-9]+_[0-9]+.[0-9]+$','00',str_now2)
    str_now = str_now3
    filename =str_now + '_' + keyword
    return filename

def highlight_word(fulltext, word):
    fulltext2 = re.sub(word, "<mark>"+word+"</mark>", fulltext)
    return fulltext2

def lowlight_word(fulltext, word):
    fulltext2 = re.sub(word, "<mark style=\"color: white; background-color:green\">"+word+"</mark>", fulltext)
    return fulltext2

def add_keyword(keyword):
    dict1 = read_keyword('bull')
    dict2 = read_keyword(keyword)
    dict3 = read_keyword('bear')
    for word in dict2.keys():
        if(type(dict2[word])==str):
            continue
        if (float(dict2[word]) > 1) and (word not in dict1):
            dict1[word] = dict2[word]
            print(dict2[word], word)
        elif (float(dict2[word]) < -1) and (word not in dict3):
            dict3[word] = dict2[word]
            print(dict2[word], word)
    #save file
    filename = 'Data/dict_bull.csv'
    f1 = open(filename,'w')
    for word in dict1.keys():
        print(word)
        f1.write(word+','+str(dict1[word])+',')
    f1.close()
    filename = 'Data/dict_bear.csv'
    f2 = open(filename,'w')
    for word in dict3.keys():
        f2.write(word+','+str(dict3[word])+',')
    f2.close()
    return 1
def get_full_year_data(num,a1,a2):
    mat0=datetime.datetime.today()
    mat = re.split('\-|\:| ',str(mat0))
    mm0 = int(mat[1])
    yy0 = int(mat[0])
    mm1 = [ i-12 if i>12 else i for i in range(mm0,mm0+12,1)]
    mm = [str(mm1[i]) if mm1[i] >= 10 else "0"+str(mm1[i]) for i in range(11)]
    yy= [str(yy0) if i>12 else str(yy0-1) for i in range(mm0,mm0+12,1)]
    mat=np.chararray([240,9],itemsize=100)
    mat[:,:]='0'
    for i in range(11):
        data = craw_one_month(num, str(yy[i])+str(mm[i])+"01")
        if('很抱歉' in data['stat']):
            plt.figure(figsize=((10,8)))
            plt.savefig('Data/'+str(num)+'.png')
            return 1
        if i == 0:
            mat0=np.array(data['data'])
            for k in range(int(np.size(mat0)/9)- 1):
                if(len(mat0[k,0]) ==0 or len(mat0[k,0]) >=20):
                    continue;
                mat[k,:] = mat0[k,:]
            j = k+1
        else:
            mat0=np.array(data['data'])
            for k2 in range(int(np.size(mat0)/9)- 1):
                if(len(mat0[k2,:]) ==0 or len(mat0[k2,:]) >=20):
                    continue;
                mat[j,:] = mat0[k2,:]
                j = j+1
        time.sleep(random.random()*1)
    df = pd.DataFrame(mat[0:j-2,:])
    mat_plt = np.zeros([j-2,1])
    for i in range(j-2):
        mat_plt[i] = float(re.sub(',','',df.iloc[i,6].decode("utf-8")))
    plt.figure(figsize=((10,8)))
    plt.plot(range(j-2),mat_plt,'k-')
    plt.plot(range(j-2),a1*np.ones([j-2,1]),'r-')
    plt.plot(range(j-2),a2*np.ones([j-2,1]),'b-')
    plt.show()
    plt.savefig(str(num)+'.png')
    df.dict_date = {}
    for i in range(j-2):
        df.dict_date[re.sub('/','',df.iloc[i,0].decode("utf-8"))] = i
    return df
        
def make_full_year_plot(num):
    plt.plot(df["日期"], df["收盤價"])
    
def build_website(keyword, ekeyword, num):#!/usr/bin/env python3
    # parameter setting
    
    content_dict={}
    #initialize
    # Main
    str_now= get_datetime()
    url, title, fulltext = scrapBingNews(keyword)
    url, title, fulltext2 = scrapGoogleNews(keyword)
    fulltext = fulltext + fulltext2
    str_full_text, content_dict = getNewsDataClean(fulltext, content_dict, str_now)
    filename = 'html/News_'+ekeyword+'.html'
    fo = open(filename,'w')  

    f = open('html/Sample.html')
    ii = 0
    lines = f.readlines()
    lines2 = ''
    for line in lines:
        if 'SampleTitleXXX' in line:
            lines2=lines2 + keyword
            print('1')
        elif 'FullText' in line:
            for sentence in content_dict.keys():
                lines2 = lines2 + sentence
                print('!!!')
        else:
            lines2=lines2 + line
        ii = ii + 1
    lines2 = re.sub('SampleHeaderXXXXX',keyword,lines2);
    lines2 = re.sub('SampleSubHeaderXXXXX',str_now,lines2);
    lines2 = re.sub('SampleClassXXXXX',keyword+str_now,lines2);
    lines2 = re.sub('SampleMp3XXXXX',ekeyword,lines2);
    dict_bull = read_keyword('bull')
    dict_bear = read_keyword('bear')
    for word in dict_bull.keys():
        print(word)
        lines2 = highlight_word(lines2, word)    
    for word in dict_bear.keys():
        lines2 = lowlight_word(lines2, word)

    for line in lines2:
        fo.write(line)
    fo.close()
    return 0
def read_keyword(keyword):
    dict_keyword = {}
    f=open('Data/dict_'+keyword+'.csv','r')
    line = f.readline()
    lines = f.readlines()
    for line in lines:
        mat=re.split(',|\n',line)
        if len(mat) >= 1:
            i=0
            while i <len(mat)-2:
                try: 
                    if int(mat[i+1]) and (len(mat[i+1]) > 0):
                        dict_keyword[mat[i]] = int(mat[i+1])
                        i=i+2
                        continue
                except:
                    i = i + 1
    f.close()
    return dict_keyword

def monthly_report(year, month):
    
    # 假如是西元，轉成民國
    if year > 1990:
        year -= 1911
    
    url = 'https://mops.twse.com.tw/nas/t21/sii/t21sc03_'+str(year)+'_'+str(month)+'_0.html'
    if year <= 98:
        url = 'https://mops.twse.com.tw/nas/t21/sii/t21sc03_'+str(year)+'_'+str(month)+'.html'
    
    # 偽瀏覽器
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    
    # 下載該年月的網站，並用pandas轉換成 dataframe
    r = requests.get(url, headers=headers)
    r.encoding = 'big5'

    dfs = pd.read_html(StringIO(r.text), encoding='big-5')

    df = pd.concat([df for df in dfs if df.shape[1] <= 11 and df.shape[1] > 5])
    
    if 'levels' in dir(df.columns):
        df.columns = df.columns.get_level_values(1)
    else:
        df = df[list(range(0,10))]
        column_index = df.index[(df[0] == '公司代號')][0]
        df.columns = df.iloc[column_index]
    
    df['當月營收'] = pd.to_numeric(df['當月營收'], 'coerce')
    df = df[~df['當月營收'].isnull()]
    df = df[df['公司代號'] != '合計']
    
    # 偽停頓
    time.sleep(5)

    return df


def financial_statement(year, season):

    if year >= 1000:
        year -= 1911
        
    url = 'http://mops.twse.com.tw/mops/web/ajax_t163sb06'
    form_data = {
        'encodeURIComponent':1,
        'step':1,
        'firstin':1,
        'off':1,
        'TYPEK':'sii',
        'year': year,
        'season': season,
    }

    response = requests.post(url,form_data)
    response.encoding = 'utf8'
    return response.text
def remove_td(column):
    remove_one = column.split('<')
    remove_two = remove_one[0].split('>')
    return remove_two[1].replace(",", "")

def translate_dataFrame(response):
     # 拆解內容
    table_array = response.split('<table')
    tr_array = table_array[1].split('<tr')
    
    # 拆解td
    data = []
    index = []
    column = []
    for i in range(len(tr_array)):
        td_array = tr_array[i].split('<td')
        if(len(td_array)>1):
            code = remove_td(td_array[1])
            name = remove_td(td_array[2])
            revenue  = remove_td(td_array[3])
            profitRatio = remove_td(td_array[4])
            profitMargin = remove_td(td_array[5])
            preTaxIncomeMargin = remove_td(td_array[6])
            afterTaxIncomeMargin = remove_td(td_array[7])
            if(type(code) == float):
                data.append([code, revenue, profitRatio, profitMargin, preTaxIncomeMargin, afterTaxIncomeMargin])
                index.append(name)
            if( i == 1 ):
                column.append(code)
                column.append(revenue)
                column.append(profitRatio)
                column.append(profitMargin)
                column.append(preTaxIncomeMargin)
                column.append(afterTaxIncomeMargin)
                
    return pd.DataFrame(data=data, index=index, columns=column)


