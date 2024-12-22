## AUTHOR: David Yaich
## Date: 12/21/2024
##
## Purpose:
##
## Purpose of this code is to demonstrate Modern Portfolio Theory (MPT). Below is a list of stocks on the 
## S&P 500. The objective is to plot the effeicient frontier and find the optimal portfolio. This requires
## iterating through 50000 different weight distributions of the current holdings.Using these weights, 
## the code will calculate the expected return and standard deviation for the portfolio. This will then
##  be plotted with its CAL to find the tangent point on the efficiency frontier. This is the 
## optimal portfolio. The Sharpe Ratio will be computed in tandem as well. The second part of this code
## will calculate the future return of the portfolio using a Monte Carlo simulation. Should be noted: No short 
## selling is allowed and all weights must be positive.
##

import numpy as np 
import yfinance as yf
import pandas as pd 
import matplotlib.pyplot as plt
import random

# Stocks in portfolio
ticker = ['QUBT', 'QBTS', 'RGTI', 'SWPPX']

# Gathers all the stock data (close, open, etc)
get_data = yf.download(ticker, start="2022-01-01", end = '2024-12-21')['Adj Close']

# Obtains your daily returns
returns = get_data.pct_change().dropna()

# The expected returns can be the average of the historical returns
exp_returns = returns.mean()

# Calc the covariance
cov = returns.cov().to_numpy()

# Risk Free Rate - Todays rate is 4.52% annual. Converted it to daily rate 
rff = (4.52/100)/360

# Function to give random weights for n stocks in the portfolio
# all equaling to 1 
def gen_weights(n):

    numbers = [random.random() for i in range(n)]
    total = sum(numbers)
    normalized_numbers = [num / total for num in numbers]

    return normalized_numbers

# No. of Assets 
n = len(ticker)

# Fills a total list of 50000 sublists of weights for the stocks within the portfolio 
weight_list = [gen_weights(n) for i in range(50000)]

#Find the expected return of the portfolio. This is done by using the dot product. 
# This is done for every weight combo
weighted_returns = [np.dot(weights, exp_returns) for weights in weight_list]

# This will find the SD of the portfolio for every weight combo 
port_sd = []
for i in weight_list:
    weights = np.array(i)
    sd = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    port_sd.append(sd)

SR = []

for i,j in zip(weighted_returns, port_sd):
    temp = (i - rff)/(j)
    SR.append(temp)

# Prints out the greatest Sharpe Ratio and its corresponding weights
print("The largest Sharpe Ratio: " + str(max(SR)))
print("Corresponsing portfolio weights: ")
for i, j in zip(ticker, weight_list[SR.index(max(SR))]):
    print(i +': ' + str(j * 100) + '%')

# Calculates optimal return and risk
optimal_return = weighted_returns[SR.index(max(SR))]
optimal_risk = port_sd[SR.index(max(SR))]

# Capital Allocation Line (CAL), also has a slope that is the Share Ratio
slope = (optimal_return - rff)/optimal_risk


# Another form of optimization that can be done is conducting a 
# Monte Carlo sim of future returns. Using the historical mean and SD, 
# a normal distribution can be created. This code can take 252 daily 
# future returns (trading days) for each stock and do this 50000 times.

trad_days = 252
sim_no = 50000

sim_returns = np.random.multivariate_normal(
    mean=returns.mean(),
    cov=returns.cov(),
    size=(sim_no, trad_days)
)

# Since this is the future, the optimal weights are determined already
# by historical data
opt_weights = weight_list[SR.index(max(SR))]

# The future daily portfolio returns are calculated 
future_returns = np.dot(sim_returns, opt_weights)

# The future return after a year is calculated by compounding each day  
cumulative_returns = (1 + future_returns).cumprod(axis=1)
avg_fut_return = cumulative_returns.mean()

print('The average expected future return from this portfolio one year from now is ' + str(round((avg_fut_return - 1)*100, 2))+ '%')

# Plot the Efficient Frontier and CAL
cal_x = np.linspace(0, max(port_sd), 100) 
cal_y = rff + slope * cal_x 
plt.figure(figsize=(10, 6))
plt.scatter(port_sd, weighted_returns, s=20, alpha=0.5, label="Random Portfolios")
plt.plot(cal_x, cal_y, color='red', label="Capital Allocation Line (CAL)")
plt.scatter(optimal_risk, optimal_return, color='orange', label="Optimal Portfolio", s=100)
plt.xlabel('Standard Deviation (Risk)')
plt.ylabel('Expected Return (%)')
plt.title('Efficient Frontier with Capital Allocation Line')
plt.legend()
plt.grid(True)

# Plot future (simulated) return distribution. This can be done with a histogram
final_values = cumulative_returns[:, -1]  # Take the cumulative return of each simulation
plt.figure(figsize=(10, 6))
plt.hist(final_values, bins=50, alpha=0.7)
plt.title('Simulated Portfolio Returns (1 Year)')
plt.xlabel('Cumulative Return')
plt.ylabel('Frequency')
plt.grid(True)

plt.show()