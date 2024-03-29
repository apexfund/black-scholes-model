# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
import QuantLib
import yfinance as yf
import datetime as dt
import pandas_datareader as pdr
from scipy.stats import norm

from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho

# Trying random data
ticker = "INMD"
tickers = [ticker, '^GSPC']
start = dt.datetime(2016, 12, 1)
end = dt.datetime(2022, 1, 1)
 
data = yf.download(tickers=tickers, start=start, end=end, interval="1mo")
data = data['Adj Close']

# Getting options chain from YFinance

def options_chain(symbol):

  ticker = yf.Ticker(symbol)

  # Options expiration dates
  exps = ticker.options
  options = pd.DataFrame()

  for e in exps:
    opt = ticker.option_chain(e)
    opt = pd.DataFrame().append(opt.calls).append(opt.puts)
    opt['expirationDate'] = datetime.datetime.strptime(e, '%Y-%m-%d')
    options = options.append(opt, ignore_index = True)

    # Need to add 1 day to get correct expiration day due to YFinance error
    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days = 1)
    time = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365
    options['DTE'] = time
    
    # Making boolean column to denote if option is a call option
    options['Call'] = options['contractSymbol'].str[4:].apply(lambda x: "C" in x)

    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2

    # To calculate the delta
    K_para = options['strike']
    if (options['Call'] == "True").any():
      optionType = 'c'
    else:
      optionType = 'p'

    options['Delta'] = delta_calc(r, S, K_para, time, sigma, optionType)


    # Dropping unnecessary data
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])

  return options

def blackScholes(r, S, K, T, sigma, type = "c"):
  "Calculating the Black-Scholes price of a call/put"

  d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  try:
    if type == "c":
      price = S * norm.cdf(d1, 0, 1) - K * np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif type == "p":
      price = K * np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
    return price
  except:
    print("Please confirm whether this is a call ('c') or a put ('p') option.")



# Greeks - Alpha, Beta, Delta, Gamma, Theta, Vega, Rho, Sigma

# Beta = covariance / variance
def beta_calc():
  log_returns = np.log(data / data.shift())
  covariance = log_returns.cov()
  variance = log_returns['^GSPC'].var()

  beta = covariance.loc[ticker, '^GSPC'] / variance

  return beta

# Delta - rate of change of the theoretical option
def delta_calc(r, S, K, T, sigma, type = "c"):
  "Calculating Delta of an option"

  d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))

  try:
    if type == "c":
      delta_calc = norm.cdf(d1, 0, 1)
    elif type == "p":
      delta_calc = norm.cdf(-d1, 0, 1)
    return delta_calc
  except:
    print("Please confirm whether this is a call ('c') or a put ('p') option.")

# Gamma - rate of change in Delta with respect to changes in the underlying price

def gamma_calc(r, S, K, T, sigma, type = "c"):
  "Calculating Gamma of an option"

  d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  try:
      gamma_calc = norm.pdf(d1, 0, 1) / (S * sigma * np.sqrt(T))
      return gamma_calc
  except:
    print("Please confirm whether this is a call ('c') or a put ('p') option.")

# Vega - measures sensitivity to volatility
def vega_calc(r, S, K, T, sigma, type = "c"):
  "Calculating Vega of an option"

  d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  try:
    vega_calc = S * norm.pdf(d1, 0, 1) * np.sqrt(T)
    return vega_calc * 0.01
  except:
    print("Please confirm whether this is a call ('c') or a put ('p') option.")

# Theta - measures the sensitivity of the value of the derivative to the passage of time
def theta_calc(r, S, K, T, sigma, type = "c"):
  "Calculating Theta of an option"

  d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  try:
    if type == "c":
      theta_calc = -S * norm.pdf(d1, 0, 1) * sigma/(2 * np.sqrt(T)) - r * K * np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif type == "p":
      theta_calc = -S * norm.pdf(d1, 0, 1) * sigma/(2 * np.sqrt(T)) + r * K * np.exp(-r*T)*norm.cdf(-d2, 0, 1)
    return theta_calc/365
  except:
    print("Please confirm whether this is a call ('c') or a put ('p') option.")

# Rho - measures the sensitivity to the interest rate
def rho_calc(r, S, K, T, sigma, type = "c"):
  "Calculating Rho of an option"

  d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  try:
    if type == "c":
      rho_calc = K * T * np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif type == "p":
      rho_calc = -K * T * np.exp(-r*T)*norm.cdf(-d2, 0, 1)
    return rho_calc * 0.01
  except:
    print("Please confirm whether this is a call ('c') or a put ('p') option.")


r = 0.0391
S = yf.Ticker(ticker).info['regularMarketPrice']
K = 120
T = 365/365
sigma = 0.30
option_type = 'c'

option_price = round(blackScholes(r, S, K, T, sigma, option_type), 2)
delta = round(delta_calc(r, S, K, T, sigma, option_type), 3)
gamma = round(gamma_calc(r, S, K, T, sigma, option_type), 3)
vega = round(vega_calc(r, S, K, T, sigma, option_type), 3)
theta = round(theta_calc(r, S, K, T, sigma, option_type), 3)
rho = round(rho_calc(r, S, K, T, sigma, option_type), 3)
beta = round(beta_calc(), 2)


print("Current Price:", str(S), "- Strike Price:", round(K, 2))
print("Option Price: ", option_price)
print("       Delta: ", delta)
print("       Gamma: ", gamma)
print("        Vega: ", vega)
print("       Theta: ", theta)
print("         Rho: ", rho)
print("        Beta: ", beta)



################# Running Fig Leaf Strat ################# 

# Want a relatively volatile stock - high beta

def getLeapsCallFrame():
  options = options_chain(ticker)
  oneYearDate = datetime.date.today() + datetime.timedelta(days = 365)
  oneYearDate = pd.Timestamp(oneYearDate.year, oneYearDate.month, oneYearDate.day)

  leapsCallOptions = options.loc[options['expirationDate'] >= oneYearDate]

  return leapsCallOptions

def purchaseLeapsCall():
  options = getLeapsCallFrame()
  optionExpDate = ""
  optionStrike = -1
  optionBid = -1
  optionDelta = 0 

  for idx in options.index:
    if (options['Delta'][idx] > 0.75 and isLeapsCallItm(S, options['strike'][idx])):
      optionExpDate = str(options['expirationDate'][idx])[:10]
      optionStrike = options['strike'][idx]
      optionBid = options['bid'][idx]
      optionDelta = options['Delta'][idx]
      break

  return (optionExpDate, round(optionStrike, 2), round(optionBid, 2), round(optionDelta, 2))

def isLeapsCallItm(stockCurrPrice, leapsStrikePrice):
  if (0.8 * stockCurrPrice > leapsStrikePrice):
    return True
  else:
    return False

data = purchaseLeapsCall()
leapsDate = data[0]
leapsStrikePrice = data[1]
leapsBidPrice = data[2]
leapsDelta = data[3]

print(leapsDate, leapsStrikePrice, leapsBidPrice, leapsDelta)

# Find intrinsic value/premium - goal is to be high
intrinsicValue = round(S - leapsStrikePrice, 2)
if (intrinsicValue < 0):
  intrinsicValue = 0
premium = leapsBidPrice - intrinsicValue

print(intrinsicValue, premium)

def getShortTermCallFrame():
  options = options_chain(ticker)
  oneMonthDate = datetime.date.today() + datetime.timedelta(days = 30)
  oneMonthDate = pd.Timestamp(oneMonthDate.year, oneMonthDate.month, oneMonthDate.day)

  upperTimeBoundDate = datetime.date.today() + datetime.timedelta(days = 45)
  upperTimeBoundDate = pd.Timestamp(upperTimeBoundDate.year, upperTimeBoundDate.month, upperTimeBoundDate.day)
  
  shortTermCalls = options.loc[options['expirationDate'] >= oneMonthDate and options['expirationDate'] <= upperTimeBoundDate]

  return shortTermCalls

def purchaseShortTermCall(leapsDelta):
  options = getShortTermCallFrame()
  optionExpDate = ""
  optionStrike = -1
  optionBid = -1
  optionDelta = 0 

  for idx in options.index:
    if (options['Delta'][idx] + 0.1 <= leapsDelta and isShortTermCallOtm(S, options['strike'][idx])):
      optionExpDate = str(options['expirationDate'][idx])[:10]
      optionStrike = options['strike'][idx]
      optionBid = options['bid'][idx]
      optionDelta = options['Delta'][idx]
      break

  return (optionExpDate, round(optionStrike, 2), round(optionBid, 2), round(optionDelta, 2))

def isShortTermCallOtm(stockCurrPrice, callStrikePrice):
  if (stockCurrPrice < callStrikePrice):
    return True
  else:
    return False


bsOptionPrice = blackScholes(r, S, leapsStrikePrice, T, sigma, option_type)
if (leapsBidPrice <= bsOptionPrice):
  print("Fair price for contract")

  # Find a call with a Delta of 0.1 lower than LEAPS delta
  shortTermCallData = purchaseShortTermCall(leapsDelta)

else:
  print("Not a fair price for contract. Black Scholes fair price: " + str(bsOptionPrice))
# Check the option price and compare to the Black Scholes model




