import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import date
from dateutil.relativedelta import relativedelta

# Download data function
def download_data(ticker_symbols, start_date,  end_date):
        data = {}
        for symbol in ticker_symbols:
                symbol_data = yf.download(symbol, start=start_date, end=end_date)
                data[symbol] = symbol_data['Adj Close']
        symbol_prices = pd.DataFrame(data)
        return symbol_prices


# returns annualized return percentage
def calculate_annualized_returns(ticker_data):
    daily_returns = ticker_data.pct_change().dropna()
    annualized_returns = daily_returns.mean() * 252  # 252 trading days in a year
    return annualized_returns


# returns covariance matrix
def calculate_annualized_covariance(ticker_data):
    daily_returns = ticker_data.pct_change().dropna()
    annualized_covariance = daily_returns.cov() * 252
    return annualized_covariance


# returns a portfolio's returns and standard deviation given a specific weight
def portfolio_performance(weights, returns, covariance_matrix):
    # return = sum of weights*returns
    portfolio_return = np.sum(returns * weights)
    # standard deviations = sqrt(variance) = sqrt(weights.T * covariance * weights) 
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return portfolio_return, portfolio_std_dev

# function that optimizese a portfolio given a target return percent
def optimize_portfolio(returns, covariance_matrix, target_return):
    num_tickers = len(returns)
    args = (returns, covariance_matrix)
    constraints = [{# Weights sum to 1
                    'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   # Return matches target return
                   {'type': 'eq', 'fun': lambda x: portfolio_performance(x, returns, covariance_matrix)[0] - target_return}]  
    bounds = tuple((0, 1) for i in range(num_tickers))  # Weight bounds (0 to 1)

    # Initial guess for equal weights
    initial_weights = num_tickers * [1. / num_tickers]
    def vol_function(weights):
          return portfolio_performance(weights, returns, covariance_matrix)[1]

    result = minimize(vol_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


