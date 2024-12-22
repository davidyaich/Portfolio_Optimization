import numpy as np 
import yfinance as yf
import pandas as pd 
import matplotlib.pyplot as plt
import random

# Stocks in portfolio
ticker = ['AMZN', 'GOLD', 'TSLA', 'META', 'GOOG', 'CL=F', 'NFLX']

# Gathers all the stock data (close, open, etc)
get_data = yf.download(ticker, start="2022-01-01", end = '2024-12-21')['Close']

# Obtains your daily returns
returns = get_data.pct_change().dropna()

# The expected returns can be the average of the historical returns
exp_returns = returns.mean()

# Calc the covariance
cov = returns.cov().to_numpy()

# Risk Free Rate - Todays rate is 4.52% annual. Converted it to daily rate 
rff = (4.52/100)/365

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

print("The largest Sharpe Ratio: " + str(max(SR)))
print("Corresponsing portfolio weights: ")
for i, j in zip(ticker, weight_list[SR.index(max(SR))]):
    print(i +': ' + str(j))


plt.figure(figsize=(10, 6))
plt.scatter(port_sd, weighted_returns, s=20, alpha=0.5)
plt.xlabel('Standard Deviation (Risk)', fontsize=12)
plt.ylabel('Expected Return', fontsize=12)
plt.title('Efficient Frontier', fontsize=14)
plt.grid(True)
plt.show()
