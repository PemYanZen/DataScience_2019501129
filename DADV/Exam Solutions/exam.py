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
    
    time.sleep(20)
    
    
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
    
    time.sleep(20)
    
    
    WebDriverWait(driver, 40).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#Col1-1-HistoricalDataTable-Proxy > section > div.Pt\(15px\) > div.C\(\$tertiaryColor\).Mt\(20px\).Mb\(15px\) > span.Fl\(end\).Pos\(r\).T\(-6px\) > a > span'))).click()
    time.sleep(10)
    
    driver.quit()
    
    
    
def getHistoricalDataDaily(ticker):
    
    chromeOptions = Options()
    chromeOptions.add_experimental_option("prefs", {"download.default_directory": "/Users/pemayangdon/Desktop/finalExam/Daily"})
    
    driver = webdriver.Chrome(executable_path = PATH, options = chromeOptions)
    
    url = "https://finance.yahoo.com/quote/"+ticker+"/history?period1="+str(int(time.mktime(f.timetuple())))+"&period2="+str(int(time.mktime(t.timetuple())))+"&interval=1d&filter=history&frequency=1d"
    
    driver.get(url)
    
    time.sleep(20)
    
    
    WebDriverWait(driver, 40).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#Col1-1-HistoricalDataTable-Proxy > section > div.Pt\(15px\) > div.C\(\$tertiaryColor\).Mt\(20px\).Mb\(15px\) > span.Fl\(end\).Pos\(r\).T\(-6px\) > a > span'))).click()
    time.sleep(10)
    
    driver.quit()



df_sp500ticker = pd.read_csv("sp500.csv")
df_sp5 = df_sp500ticker['Companies']

def parallelDownload(term):
    
    df_sp5 = df_sp500ticker['Companies']
    
    num_cores = multiprocessing.cpu_count()
    
    Parallel(n_jobs=num_cores)(delayed(getHistoricalDataWeekly)(ticker=i) for i in df_sp5[:6])
        

    
# for i in df_sp5[:5]:
#     getHistoricalDataDaily(i)
    # getHistoricalDataMonthly(i)
#     getHistoricalDataWeekly(i)

    

   

#QUESTION 2
    
def compute_gain(df):
    df_sorted = df.sort_values(by="Date",ascending=True).set_index("Date")
    # .last(period)
    gain = (df_sorted['Close'].iloc[-1] / df_sorted['Close'].iloc[0])  - 1
    return gain


def dailyGain():
    gainers = []

    for filename in glob.glob('/Users/pemayangdon/Desktop/finalExam/Daily/*.csv'):
        data_df = pd.read_csv(filename)
        
        data_df['Date']= pd.to_datetime(data_df['Date'])
        gainers.append([filename[filename.rfind('/')+1:filename.find('.')],round(compute_gain(data_df), 2)*100])
        
    daily_gainers = pd.DataFrame(gainers,columns=['Companies','DailyGain'])
    
    
    daily_gainers.sort_values(['DailyGain'], ascending=[False], ignore_index=True, inplace=True)
    return daily_gainers
    

def weeklyGain():
    gainers = []

    for filename in glob.glob('/Users/pemayangdon/Desktop/finalExam/Weekly/*.csv'):
        data_df = pd.read_csv(filename)
        data_df['Date']= pd.to_datetime(data_df['Date'])
        gainers.append([filename[filename.rfind('/')+1:filename.find('.')],round(compute_gain(data_df), 2)*100])
    
    weekly_gainers = pd.DataFrame(gainers,columns=['Companies','WeeklyGain'])
    
    weekly_gainers.sort_values(['WeeklyGain'], ascending=[False], ignore_index=True, inplace=True)
    return weekly_gainers

def monthlyGain():
    gainers = []

    for filename in glob.glob('/Users/pemayangdon/Desktop/finalExam/Monthly/*.csv'):
        data_df = pd.read_csv(filename)
        data_df['Date']= pd.to_datetime(data_df['Date'])
        gainers.append([filename[filename.rfind('/')+1:filename.find('.')],round(compute_gain(data_df), 2)*100])
    
    monthly_gainers = pd.DataFrame(gainers,columns=['Companies','MonthlyGain'])
    
    monthly_gainers.sort_values(['MonthlyGain'], ascending=[False], ignore_index=True, inplace=True)
    return monthly_gainers

daily_gainers  = dailyGain()
# print(daily_gainers)


weekly_gainers = weeklyGain()
# print(weekly_gainers)

monthly_gainers = monthlyGain()
# print(monthly_gainers)




#QUESTION 3


df_gainers_dict = {}
merg01 = pd.merge(daily_gainers, weekly_gainers, on='Companies')
df_merge_gains = pd.merge(merg01, monthly_gainers, on='Companies')

def corrMatrix():
    
    # merg01 = pd.merge(daily_gainers, weekly_gainers, on='Companies')
    # df_merge_gains = pd.merge(merg01, monthly_gainers, on='Companies')
    # print(df_merge_gains)
    
    df_gainers_dict = df_merge_gains.set_index('Companies').T.to_dict('list')
    cols = []
    
    for c in df_merge_gains['Companies']:
        cols.append(c)
        
    # print(df_merge_gains['Companies'])
    df = pd.DataFrame(df_gainers_dict,columns=cols)    
    corrMatrix = df.corr()
    # print(corrMatrix)
    
    sn.heatmap(corrMatrix, annot=True)
    plt.show()

# corrMatrix()



# #QUESTION 4
def plot():
    top_ = df_merge_gains.head(3)
    bottom_ = df_merge_gains.tail(3)
    
    sectors = df_sp500ticker['sector']
    sectors.drop_duplicates(inplace=True)
    
    
    top1 = top_['Companies']
    bot1 = bottom_['Companies']
   
    
    df3 = DataFrame()
    temp = []
    
    dict1_top25 = {}
    dict1_bottom = {}
    for c in top1:
        # print(c)
        df3 = df_sp500ticker[(df_sp500ticker['Companies'] == str(c)+"\n")]
        
        for s in sectors:
            val = df3['sector'] == str(s)
           
            if (val.bool()):
                if s not in dict1_top25:
                    dict1_top25[s] = 1
                else:
                    dict1_top25[s]+=1
        
    # print(dict1_top25)
    
    for c in top1:
        # print(c)
        df3 = df_sp500ticker[(df_sp500ticker['Companies'] == str(c)+"\n")]
        
        for s in sectors:
            val = df3['sector'] == str(s)
           
            if (val.bool()):
                if s not in dict1_bottom:
                    dict1_bottom[s] = 1
                else:
                    dict1_bottom[s]+=1
        
    # print(dict1_bottom)
    
    data = []
    for i in sectors:
        t = []
        if i in dict1_top25:
            t.append(i)
            t.append(dict1_top25[i])
        if i in dict1_bottom:
            t.append(dict1_bottom[i])
        if len(t) != 0:
            data.append(t)
    # print(data)   
    df5 = pd.DataFrame(data,columns=['sectors','top', 'bottom'])
        
    
    df = pd.DataFrame({'top': df5['top'],'bottom': df5['bottom']}, index = df5['sectors'])
    ax = df5.plot.bar(rot=0)
    
# plot()
