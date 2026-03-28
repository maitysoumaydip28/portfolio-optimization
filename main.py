import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']

data = yf.download(stocks, start='2020-01-01', auto_adjust=True)['Close']

data = data.dropna(axis=1)

returns = data.pct_change().dropna()

mean_returns = returns.mean()
cov_matrix = returns.cov()

num_portfolios = 5000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    weights = np.random.random(len(data.columns))
    weights /= np.sum(weights)

    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    results[0, i] = portfolio_std
    results[1, i] = portfolio_return
    results[2, i] = results[1, i] / results[0, i]

plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.title('Efficient Frontier')
plt.colorbar(label='Sharpe Ratio')
plt.show()

max_sharpe_idx = np.argmax(results[2])
print("Best Return:", results[1, max_sharpe_idx])
print("Best Risk:", results[0, max_sharpe_idx])
