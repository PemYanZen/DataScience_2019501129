#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 09:35:57 2021
@author: pemayangdon
"""


import requests
from bs4 import BeautifulSoup
import pandas as pd
from pandas import DataFrame
import seaborn as sn
import matplotlib.pyplot as plt


from selenium import webdriver
import time
import datetime


import glob

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
    
from selenium.webdriver.chrome.options import Options


from joblib import Parallel, delayed
import multiprocessing

PATH = "/Users/pemayangdon/Desktop/chromedriver"
f = datetime.datetime(2017, 2, 1) 
t = datetime.datetime(2021, 1, 2)

#By using scraping tools download the last 1000 trading days historical data (daily, weekly and monthly) for all S&P500 companies in to your system. Use parallelization to make download faster. (Note: Saturdays and Sundays and some festival days etc. are not trading days. The NYSE and NASDAQ average about 253 trading days a year)
sp500 = []

df_sp500 = DataFrame()
def getSP500():
    page = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find("table", id = "constituents")
    
    rows = table.findAll('tr')
    c = 0
    for tr in rows:
        c+=1
        if (c == 1):
            continue
        cols = tr.findAll('td')
        sp500.append([cols[0].getText(), cols[3].getText()])
        
    df_sp500 = DataFrame(sp500,columns=['Companies', 'sector'])

    df_sp500.to_csv("sp500.csv")

# getSP500()


def getHistoricalDataWeekly(ticker):
   
    
    chromeOptions = Options()
    chromeOptions.add_experimental_option("prefs", {"download.default_directory": "/Users/pemayangdon/Desktop/finalExam/Weekly"})
    
    driver = webdriver.Chrome(executable_path = PATH, options = chromeOptions)
    
    url = "https://finance.yahoo.com/quote/"+ticker+"/history?period1="+str(int(time.mktime(f.timetuple())))+"&period2="+str(int(time.mktime(t.timetuple())))+"&interval=1wk&filter=history&frequency=1wk"
    
    driver.get(url)
    
    time.sleep(3)
    
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")

    time.sleep(5)    
    WebDriverWait(driver, 40).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#Col1-1-HistoricalDataTable-Proxy > section > div.Pt\(15px\) > div.C\(\$tertiaryColor\).Mt\(20px\).Mb\(15px\) > span.Fl\(end\).Pos\(r\).T\(-6px\) > a > span'))).click()
    time.sleep(10)
    
    driver.quit()

    
# getHistoricalData("NEE")

def getHistoricalDataMonthly(ticker):    
   
    chromeOptions = Options()
    chromeOptions.add_experimental_option("prefs", {"download.default_directory": "/Users/pemayangdon/Desktop/finalExam/Monthly"})
    
    driver = webdriver.Chrome(executable_path = PATH, options = chromeOptions)
    
    url = "https://finance.yahoo.com/quote/"+ticker+"/history?period1="+str(int(time.mktime(f.timetuple())))+"&period2="+str(int(time.mktime(t.timetuple())))+"&interval=1mo&filter=history&frequency=1mo"
    
    driver.get(url)
    
    time.sleep(3)
    
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")

    time.sleep(5)  
    
    WebDriverWait(driver, 40).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#Col1-1-HistoricalDataTable-Proxy > section > div.Pt\(15px\) > div.C\(\$tertiaryColor\).Mt\(20px\).Mb\(15px\) > span.Fl\(end\).Pos\(r\).T\(-6px\) > a > span'))).click()
    time.sleep(10)
    
    driver.quit()
    
    
    
def getHistoricalDataDaily(ticker):
    
    chromeOptions = Options()
    chromeOptions.add_experimental_option("prefs", {"download.default_directory": "/Users/pemayangdon/Desktop/finalExam/Daily"})
    
    driver = webdriver.Chrome(executable_path = PATH, options = chromeOptions)
    
    url = "https://finance.yahoo.com/quote/"+ticker+"/history?period1="+str(int(time.mktime(f.timetuple())))+"&period2="+str(int(time.mktime(t.timetuple())))+"&interval=1d&filter=history&frequency=1d"
    
    driver.get(url)
    
    time.sleep(3)
    
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")

    time.sleep(5)  
    WebDriverWait(driver, 40).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#Col1-1-HistoricalDataTable-Proxy > section > div.Pt\(15px\) > div.C\(\$tertiaryColor\).Mt\(20px\).Mb\(15px\) > span.Fl\(end\).Pos\(r\).T\(-6px\) > a > span'))).click()
    time.sleep(10)
    
    driver.quit()



df_sp500ticker = pd.read_csv("sp500.csv")
df_sp5 = df_sp500ticker['Companies']

def parallelDownload(term):
    
    df_sp5 = df_sp500ticker['Companies']
    
    num_cores = multiprocessing.cpu_count()
    
    if (term == 'w'):
        Parallel(n_jobs=num_cores)(delayed(getHistoricalDataWeekly)(ticker=i) for i in df_sp5[:10])
    elif (term == 'm'):
        Parallel(n_jobs=num_cores)(delayed(getHistoricalDataMonthly)(ticker=i) for i in df_sp5[:10])
    else:
        Parallel(n_jobs=num_cores)(delayed(getHistoricalDataDaily)(ticker=i) for i in df_sp5[:10])

# parallelDownload('w')
# parallelDownload('m')
# parallelDownload('d')

# for i in df_sp5[:10]:
#     getHistoricalDataWeekly(i)
      # getHistoricalDataMonthly(i)
        # getHistoricalDataDaily(i)

    
    

#QUESTION 2
    
def compute_gain(df):
    df_sorted = df.sort_values(by="Date",ascending=True).set_index("Date")
    df = df_sorted['Close']
    return df.pct_change()
    


def dailyGain():
    gainers = {}

    for filename in glob.glob('/Users/pemayangdon/Desktop/finalExam/Daily/*.csv'):
        data_df = pd.read_csv(filename)
        
        data_df['Date']= pd.to_datetime(data_df['Date'])
        df4 = round(compute_gain(data_df), 2).tolist()
        
        gainers[filename[filename.rfind('/')+1:filename.find('.')]] = df4
    return gainers
   
    

def weeklyGain():
    gainers = {}

    for filename in glob.glob('/Users/pemayangdon/Desktop/finalExam/Weekly/*.csv'):
        data_df = pd.read_csv(filename)
        data_df['Date']= pd.to_datetime(data_df['Date'])
        df4 = round(compute_gain(data_df), 2).tolist()
                
        gainers[filename[filename.rfind('/')+1:filename.find('.')]] = df4
    return gainers

def monthlyGain():
    gainers = {}

    for filename in glob.glob('/Users/pemayangdon/Desktop/finalExam/Monthly/*.csv'):
        data_df = pd.read_csv(filename)
        data_df['Date']= pd.to_datetime(data_df['Date'])
        df4 = round(compute_gain(data_df), 2).tolist()
                
        gainers[filename[filename.rfind('/')+1:filename.find('.')]] = df4
    return gainers

daily_gainers  = dailyGain()

weekly_gainers = weeklyGain()
# print(weekly_gainers)

monthly_gainers = monthlyGain()
# print(monthly_gainers)

 


#QUESTION 3


df_gainers_dict = {}# 
def corrMatrix():
    daily_gain = pd.DataFrame.from_dict(daily_gainers)
    
    corrMatrix = daily_gain.corr()    
    sn.heatmap(corrMatrix, annot=True, cmap=['red','blue'])
    plt.show()
    
    weekly_gain = pd.DataFrame.from_dict(weekly_gainers)
    corrMatrix = weekly_gain.corr()    
    sn.heatmap(corrMatrix, annot=True, cmap=['red','blue'])
    plt.show()
    
    monthly_gain = pd.DataFrame.from_dict(monthly_gainers)
    corrMatrix = monthly_gain.corr()
    
    sn.heatmap(corrMatrix, annot=True, cmap=['red','blue'])
    plt.show()
    
# corrMatrix()



# QUESTION 4
def plot():
    
    daily_gains = pd.DataFrame.from_dict(daily_gainers, orient='index')

    daily_tickers = daily_gains.index

    daily_avg = pd.DataFrame(daily_gains.mean(axis=1), index=daily_tickers)
    daily_avg.sort_values(by=0,ascending=False, inplace=True)
    
    top_= daily_avg.head(5) #top 5
    top_idx = top_.index
    
    bottom_ = daily_avg.tail(5) #bottom 5
    bottom_idx = bottom_.index
    
    # print(top_idx)
    sectors = df_sp500ticker['sector']
    sectors.drop_duplicates(inplace=True)
    
    # print(df_sp500ticker.head(3))
    
   
    df3 = DataFrame()
    sec_list = []
    
    dict1_top = {}
    dict1_bottom = {}
    for c in top_idx:
        df3 = df_sp500ticker[(df_sp500ticker['Companies'] == str(c)+"\n")]
        for s in sectors:
            val = df3['sector'] == str(s)
           
            if (val.bool()):
                if s not in sec_list:
                    sec_list.append(s)
                if s not in dict1_top:
                    dict1_top[s] = 1
                else:
                    dict1_top[s]+=1
        
    # print(dict1_top)
    
    for c in bottom_idx:
        df3 = df_sp500ticker[(df_sp500ticker['Companies'] == str(c)+"\n")]
        
        for s in sectors:
            val = df3['sector'] == str(s)
           
            if (val.bool()):
                if s not in sec_list:
                    sec_list.append(s)
                if s not in dict1_bottom:
                    dict1_bottom[s] = 1
                else:
                    dict1_bottom[s]+=1
        
    # print(dict1_bottom)
    
    
    data = []
    for sc in sec_list:
        tmp = []
        tmp.append(sc)
        if sc in dict1_top:
            tmp.append(dict1_top[sc])
        else:
            tmp.append(0)
        if sc in dict1_bottom:
            tmp.append(dict1_bottom[sc])
        else:
            tmp.append(0)
        data.append(tmp)
    df5 = pd.DataFrame(data,columns=['sectors','top5', 'bottom5'])
    idx = df5['sectors']
    
    df = pd.DataFrame({'top5': df5['top5'],'bottom5': df5['bottom5']}, index = idx)
    # ax = df5.plot.bar(rot=0)
    
# plot()
    
    
    