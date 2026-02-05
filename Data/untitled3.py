#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:59:40 2024

@author: rubylintu
"""
import requests
from bs4 import BeautifulSoup as bs

# Get login page
response = requests.get('https://meet.eslite.com/')


# Get csrf token
token = "__RequestVerificationToken"

bs0 = bs(response.text,"html.parser")

# Get cookies
cookies = response.cookies


# Authentication
response = requests.post(
    'https://meet.eslite.com/tw/tc/member/membernewsfeed#',
    data = {
        'access_token'   : [token, token],
        'account_id'  : '0908008812',
        'password' : 'r780605',
        'consumption ' : '1000000',
        'User-Agent' : cookies
    },
    cookies = cookies
)

print(response.text)

