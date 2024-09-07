from functions import funcs
import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

def main():
        #pre-defined list of etfs
        # *********************************************
        # CAN CHANGE TO DIFFERENT TICKER SYMBOLS
        # *********************************************
        etfs = ['SPY', 'VOO', 'SCHD', 'VTI', 'QQQ']

        # calculate the today's date and the date exactly 10 years in the past
        present_date = date.today()
        past_date = present_date - relativedelta(years=10)
        present_date = present_date.strftime('%Y-%m-%d')
        past_date = past_date.strftime('%Y-%m-%d')

        # download the etf data from dates above
        etf_data = funcs.download_data(etfs, past_date, present_date)
        prices = pd.DataFrame(etf_data)

        # Calculate annualized returns and covariance
        annualized_returns = funcs.calculate_annualized_returns(prices)
        annualized_covariance = funcs.calculate_annualized_covariance(prices)

        
        # Ask user for target return percentage (e.g., 5% annual return = 0.05)
        target_return = float(input("Enter your target annual return percentage (as a decimal, e.g., 0.05 for 5%): "))

        # Optimize the portfolio for minimum volatility at the target return
        optimized_portfolio = funcs.optimize_portfolio(annualized_returns, annualized_covariance, target_return)

        # Display the optimized portfolio weights
        optimized_weights = optimized_portfolio.x
        print("Optimized Portfolio Weights (Min Volatility):")
        for etf, weight in zip(etfs, optimized_weights):
            print(f"{etf}: {weight:.4f}")

        # Calculate and display the sum of the optimized weights
        weights_sum = np.sum(optimized_weights)
        print(f"Sum of Optimized Weights: {weights_sum:.4f}")

        # Equal weights for comparison
        equal_weights = np.array([1 / len(etfs)] * len(etfs))  # Equal weights for all ETFs

        # Performance of the optimized portfolio
        opt_return, opt_std_dev = funcs.portfolio_performance(optimized_weights, annualized_returns, annualized_covariance)

        # Performance of the equally weighted portfolio
        eq_return, eq_std_dev = funcs.portfolio_performance(equal_weights, annualized_returns, annualized_covariance)

        # Print the comparison between optimized and equal weights portfolios
        print("\nComparison of Optimized vs Equal Weights Portfolio:")
        print(f"Optimized Portfolio - Return: {opt_return:.4f}, Volatility: {opt_std_dev:.4f}")
        print(f"Equal Weights Portfolio - Return: {eq_return:.4f}, Volatility: {eq_std_dev:.4f}")


     
        
    
# risk = weights*covariance matrix * weights.T
# returns = returns vector * weights.T or by dot product

if __name__ == "__main__":
    main()
