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


import glob

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
    


from joblib import Parallel, delayed
import multiprocessing

PATH = "/Users/pemayangdon/Desktop/chromedriver"

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
    
    url = "https://finance.yahoo.com/quote/"+ticker+"/history?period1=1519842600&period2=1544639400&interval=1d&filter=history&frequency=1d"
    driver.get(url)
    
    time.sleep(5)
    
    
    element = driver.find_element_by_xpath('//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[1]/div[3]/span/div/span/span')
    driver.execute_script("arguments[0].innerText = 'Weekly'", element)
    time.sleep(10)
    
    WebDriverWait(driver, 40).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#Col1-1-HistoricalDataTable-Proxy > section > div.Pt\(15px\) > div.C\(\$tertiaryColor\).Mt\(20px\).Mb\(15px\) > span.Fl\(end\).Pos\(r\).T\(-6px\) > a > span'))).click()
    time.sleep(10)
    
    driver.quit()

    
# getHistoricalData("NEE")

def getHistoricalDataMonthly(ticker):    
    chromeOptions = Options()
    chromeOptions.add_experimental_option("prefs", {"download.default_directory": "/Users/pemayangdon/Desktop/finalExam/Monthly"})
    
    driver = webdriver.Chrome(executable_path = PATH, options = chromeOptions)
    

    url = "https://finance.yahoo.com/quote/"+ticker+"/history?period1=1519842600&period2=1544639400&interval=1d&filter=history&frequency=1d"
    driver.get(url)
    
    time.sleep(5)
       
    
    element = driver.find_element_by_xpath('//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[1]/div[3]/span/div/span/span')
    driver.execute_script("arguments[0].innerText = 'Monthly'", element)
    time.sleep(5)
    
    WebDriverWait(driver, 40).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#Col1-1-HistoricalDataTable-Proxy > section > div.Pt\(15px\) > div.C\(\$tertiaryColor\).Mt\(20px\).Mb\(15px\) > span.Fl\(end\).Pos\(r\).T\(-6px\) > a > span'))).click()
    time.sleep(20)
    
    driver.quit()
    
    
    
def getHistoricalDataDaily(ticker):
    
    chromeOptions = Options()
    chromeOptions.add_experimental_option("prefs", {"download.default_directory": "/Users/pemayangdon/Desktop/finalExam/Daily"})
    
    driver = webdriver.Chrome(executable_path = PATH, options = chromeOptions)
    

    url = "https://finance.yahoo.com/quote/"+ticker+"/history?period1=1519842600&period2=1544639400&interval=1d&filter=history&frequency=1d"
    driver.get(url)
    
    time.sleep(5)
       
     
    element = driver.find_element_by_xpath('//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[1]/div[3]/span/div/span/span')
    driver.execute_script("arguments[0].innerText = 'Daily'", element)
    time.sleep(5)
    
    WebDriverWait(driver, 40).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#Col1-1-HistoricalDataTable-Proxy > section > div.Pt\(15px\) > div.C\(\$tertiaryColor\).Mt\(20px\).Mb\(15px\) > span.Fl\(end\).Pos\(r\).T\(-6px\) > a > span'))).click()
    time.sleep(20)

    driver.quit()




df_sp500ticker = pd.read_csv("sp500.csv")
df_sp5 = df_sp500ticker['Companies']

def parallelDownload():
    df_sp5 = df_sp500ticker['Companies']
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(getHistoricalDataWeekly)(ticker=i) for i in df_sp5[:6])
        

#for i in df_sp5[:6]:
    #getHistoricalDataWeekly(i)
    #getHistoricalDataDaily(i)
    #getHistoricalDataMonthly(i)
    


#QUESTION 2
    
def compute_gain(df, period):
    df_sorted = df.sort_values(by="Date",ascending=True).set_index("Date").last(period)
    gain = (df_sorted['Close'].iloc[-1] / df_sorted['Close'].iloc[0])  - 1
    return gain


gainers = []
df_gainers = DataFrame()

def getGainDf():
    for filename in glob.glob('data/*.csv'):
        data_df = pd.read_csv(filename)
        data_df['Date']= pd.to_datetime(data_df['Date'])
        gainers.append([filename[filename.find('/')+1:filename.find('.')],round(compute_gain(data_df, "1M"), 2)*100])
    
    df_gainers = pd.DataFrame(gainers,columns=['Companies','Gain'])
    
    df_gainers.sort_values(['Gain'], ascending=[False], ignore_index=True, inplace=True)
    # .iloc[0:10,:]
    
    # print(df_gainers.head(3))
# getGainDf()


# #QUESTION 3
df_gainers_dict = {}

def corrMatrix():
    print(len(df_gainers['Companies']))

    for i in range(0,len(df_gainers['Companies'])):
        df_gainers_dict[df_gainers['Companies'][i]] = df_gainers['Gain'][i]
        
    # print(df_gainers_dict.keys())
    # df = pd.DataFrame(df_gainers_dict, columns=[df_gainers_dict.keys()])
    
    corrMatrix = df_gainers.corr()
    # print(corrMatrix)
    
    sn.heatmap(corrMatrix, annot=True)
    plt.show()

# corrMatrix()

# #QUESTION 4
def plot():
    top_ = df_gainers.head(3)
    bottom_ = df_gainers.tail(3)
    
    sectors = df_sp500ticker['sector']
    sectors.drop_duplicates(inplace=True)
    
    
    top1 = top_['Companies']
    bot1 = bottom_['Companies']
    # print(df_gainers)
    # print(top1)
    # print(bot1)
    
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
    
    print(df5)
    
    
    df = pd.DataFrame({'top': df5['top'],'bottom': df5['bottom']}, index = df5['sectors'])
    ax = df5.plot.bar(rot=0)
    
    
