# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import QuantLib
import yfinance as yf
import datetime as dt
import pandas_datareader as pdr
from scipy.stats import norm

from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho

# Trying random data
ticker = "TWTR"
tickers = [ticker, '^GSPC']
start = dt.datetime(2016, 12, 1)
end = dt.datetime(2022, 1, 1)
 
data = yf.download(tickers=tickers, start=start, end=end, interval="1mo")
data = data['Adj Close']

def blackScholes(r, S, K, T, sigma, type = "c"):
  "Calculating the Black-Scholes price of a call/put"

  d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  try:
    if type == "c":
      price = S * norm.cdf(d1, 0, 1) - K * np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif type == "p":
      price = K * np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
    return price, bs(type, S, K, T, r, sigma)
  except:
    print("Please confirm whether this is a call ('c') or a put ('p') option.")



# Greeks - Alpha, Beta, Delta, Gamma, Theta, Vega, Rho, Sigma

# Beta = covariance / variance
log_returns = np.log(data / data.shift())

covariance = log_returns.cov()
variance = log_returns['^GSPC'].var()

beta = covariance.loc[ticker, '^GSPC'] / variance

print("Beta: " + str(beta))

# Delta - rate of change of the theoretical option
def delta_calc(r, S, K, T, sigma, type = "c"):
  "Calculating Delta of an option"

  d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))

  try:
    if type == "c":
      delta_calc = norm.cdf(d1, 0, 1)
    elif type == "p":
      delta_calc = norm.cdf(-d1, 0, 1)
    return delta_calc, delta(type, S, K, T, r, sigma)
  except:
    print("Please confirm whether this is a call ('c') or a put ('p') option.")

# Gamma - rate of change in Delta with respect to changes in the underlying price

def gamma_calc(r, S, K, T, sigma, type = "c"):
  "Calculating Gamma of an option"

  d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  try:
      gamma_calc = norm.pdf(d1, 0, 1) / (S * sigma * np.sqrt(T))
      return gamma_calc, gamma(type, S, K, T, r, sigma)
  except:
    print("Please confirm whether this is a call ('c') or a put ('p') option.")

# Vega - measures sensitivity to volatility
def vega_calc(r, S, K, T, sigma, type = "c"):
  "Calculating Vega of an option"

  d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  try:
    vega_calc = S * norm.pdf(d1, 0, 1) * np.sqrt(T)
    return vega_calc * 0.01, vega(type, S, K, T, r, sigma)
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
    return theta_calc/365, theta(type, S, K, T, r, sigma)
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
    return rho_calc * 0.01, rho(type, S, K, T, r, sigma)
  except:
    print("Please confirm whether this is a call ('c') or a put ('p') option.")

r = 0.0391
S = 30
K = 40
T = 240/365
sigma = 0.30
option_type = 'p'
print("Option Price: ", [round(x, 3) for x in blackScholes(r, S, K, T, sigma, option_type)])
print("       Delta: ", [round(x, 3) for x in delta_calc(r, S, K, T, sigma, option_type)])
print("       Gamma: ", [round(x, 3) for x in gamma_calc(r, S, K, T, sigma, option_type)])
print("        Vega: ", [round(x, 3) for x in vega_calc(r, S, K, T, sigma, option_type)])
print("       Theta: ", [round(x, 3) for x in theta_calc(r, S, K, T, sigma, option_type)])
print("         Rho: ", [round(x, 3) for x in rho_calc(r, S, K, T, sigma, option_type)])

