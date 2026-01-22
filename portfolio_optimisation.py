import yfinance as yf
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

print("\n\n\nPortfolio Optimisation QUBO\n")

tickers = ["AAPL", "TSLA", "JPM", "JNJ"]

data = pd.concat(
    [yf.Ticker(t).history(start="2022-01-01", end="2024-12-31")["Close"].rename(t)
     for t in tickers],
    axis=1
)

returns = np.log(data / data.shift(1)).dropna()

annual_returns = 252 * returns.mean()        # expected annual returns μ
cov_matrix = 252 * returns.cov()             # annualised covariance Σ

print("\n=== Annualised Expected Returns (μ) ===")
print(annual_returns.round(4))
print("\n=== Annualised Covariance Matrix (Σ) ===")
print(cov_matrix.round(6))


lam = 0.5     # risk aversion
k = 2         # number of assets to include
P = 5.0       # penalty for violating the "exactly k assets" constraint


n = len(tickers)
Q = np.zeros((n, n), dtype=np.float64)

for i in range(n):
    # diagonal term: return + risk + constraint penalty
    Q[i, i] = -annual_returns[i] + lam * cov_matrix.iloc[i, i] + P * (1 - 2 * k)
    for j in range(i + 1, n):
        # off-diagonal term: risk + constraint interaction
        Q[i, j] = lam * cov_matrix.iloc[i, j] + 2 * P
        Q[j, i] = Q[i, j]

print("\n=== QUBO Matrix (Q) ===")
print(pd.DataFrame(Q, index=tickers, columns=tickers).round(4))


model = gp.Model("Portfolio_QUBO")

x = model.addMVar(n, vtype=GRB.BINARY, name="x")

obj = x @ Q @ x

model.setObjective(obj, GRB.MINIMIZE)

model.addConstr(x.sum() == k, name="budget")

# Solve
model.optimize()


solution = x.X.round().astype(int)
selected = [tickers[i] for i in range(n) if solution[i] == 1]

print("\n=== Optimal Portfolio Solution ===")
print("Bitstring:", "".join(map(str, solution)))
print("Selected stocks:", selected)
print("Objective (energy):", model.objVal)
 