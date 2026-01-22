# Portfolio Optimisation (QUBO) — Mean-Variance Stock Selection

This project demonstrates **portfolio optimisation using a QUBO formulation** (Quadratic Unconstrained Binary Optimisation), solved using the **Gurobi Optimizer**.

The goal is to select **exactly `k` assets** from a small universe of stocks while balancing:

- **High expected return**
- **Low risk (variance / covariance)**
- **A strict selection constraint** (choose exactly `k` assets)

---

## Why QUBO Optimisation?

**QUBO optimisation** is a framework for solving problems where decisions are naturally **binary** (yes/no, include/exclude, pick/not pick). Many real-world problems can be expressed in this form, including:

- **Portfolio selection** (choose a subset of assets)
- **Scheduling** (assign tasks to workers or time slots)
- **Routing and logistics** (routing vehicles or deliveries efficiently)
- **Resource allocation** (distributing limited resources across competing choices)
- **Feature selection in machine learning**

QUBO is particularly useful because it is the standard model format used by:
- **Quantum annealing** approaches (QUBO/Ising models)
- **Many classical optimisation techniques**, including mixed-integer solvers such as **Gurobi**

In this project, QUBO is used to solve a **mean-variance portfolio selection** problem where each stock is represented by a binary variable.

---

## Assets Used

The portfolio universe contains 4 stocks:

- **AAPL** (Apple)
- **TSLA** (Tesla)
- **JPM** (JPMorgan Chase)
- **JNJ** (Johnson & Johnson)

**Data source:** Yahoo Finance via `yfinance`  
**Date range:** `2022-01-01` → `2024-12-31`

---

## What the Code Does

### 1) Download daily closing prices

The program downloads daily **Close** prices for each ticker and combines them into a single DataFrame:

```python
data = pd.concat(
    [yf.Ticker(t).history(start="2022-01-01", end="2024-12-31")["Close"].rename(t)
     for t in tickers],
    axis=1
)
```


### 2) Compute Log Returns

Daily log returns are calculated as:

$$
    r_t = \log\left(\frac{P_t}{P_{t-1}}\right)
$$

```python

returns = np.log(data / data.shift(1)).dropna()
```

---

## 3) Annualise expected returns and covariance

Expected annual return $\mu$ and annualised covariance matrix $\Sigma$ are computed using:

$$
\mu = 252 \cdot \mathrm{mean}(r)
$$

$$
\Sigma = 252 \cdot \mathrm{cov}(r)
$$

```python
annual_returns = 252 * returns.mean()
cov_matrix = 252 * returns.cov()
```

These values are used as inputs for the portfolio risk-return optimisation.

---

## 4) Formulate Mean–Variance Portfolio Selection as a QUBO

Each stock is represented by a binary decision variable:


$$
x_i \in \{0, 1\}
$$


Where:

- $x_i = 1$ means stock *i* is selected  
- $x_i = 0$ means stock *i* is excluded

### Objective: return vs risk

The objective combines return and risk:


$$
\text{minimise: } -\mu^T x + \lambda x^T \Sigma x
$$

### Constraint: select exactly $k$ assets

Selecting exactly $k$ assets is enforced using a quadratic penalty term:


$$
P\left(\sum_i x_i - k\right)^2
$$

So the full QUBO form becomes:


$$
\text{minimise: } x^T Q x
$$

Where **Q** contains contributions from:

- Expected return (encourages high return)
- Risk/covariance (penalises volatility and correlation)
- Constraint penalty (forces exactly $k$ assets)

---

## Parameters Used

| Parameter | Meaning | Value |
|---|---|---:|
| $\lambda$ | Risk aversion term | 0.5 |
| $k$ | Number of assets required | 2 |
| $P$ | Penalty weight for constraint | 5.0 |

---

## Solving the QUBO with Gurobi

The QUBO is solved as a quadratic binary optimisation problem:

```python
x = model.addMVar(n, vtype=GRB.BINARY, name="x")
obj = x @ Q @ x
model.setObjective(obj, GRB.MINIMIZE)
model.addConstr(x.sum() == k, name="budget")
model.optimize()
```

---

## Results

### Annualised Expected Returns ($\mu$)

```
AAPL    0.1149
TSLA    0.0144
JPM     0.1598
JNJ    -0.0311
```

- Highest return: **JPM** (0.1598)  
- Lowest return: **JNJ** (-0.0311) *(negative expected return)*

### Annualised Covariance Matrix ($\Sigma$)

```
          AAPL      TSLA       JPM       JNJ
AAPL  0.073231  0.083460  0.023743  0.008286
TSLA  0.083460  0.374224  0.045774  0.000494
JPM   0.023743  0.045774  0.062169  0.010786
JNJ   0.008286  0.000494  0.010786  0.026707
```

**Key observations:**

- **TSLA** has the highest variance (0.374224) → very risky  
- **JNJ** is low-risk, but has negative return  
- **AAPL + JPM** covariance is moderate, making them a reasonable balanced pair  

### QUBO Matrix ($Q$)

```
         AAPL     TSLA      JPM      JNJ
AAPL -15.0783  10.0417  10.0119  10.0041
TSLA  10.0417 -14.8272  10.0229  10.0002
JPM   10.0119  10.0229 -15.1287  10.0054
JNJ   10.0041  10.0002  10.0054 -14.9556
```

**Interpretation:**

- Diagonal values combine:
  - reward for return (negative term)
  - risk penalty (positive term)
  - constraint penalty contribution
- Off-diagonals represent interaction effects (risk correlation + penalty structure)

---

## Optimal Portfolio Solution

- **Bitstring:** `1010`  
- **Selected stocks:** `['AAPL', 'JPM']`  
- **Objective (energy):** `-10.183211587735265`

**Selected Portfolio: AAPL + JPM**

This is consistent with intuition:

- **JPM** provides the strongest expected return  
- **AAPL** provides solid return with manageable risk  

The portfolio avoids:

- **TSLA** (very high risk)  
- **JNJ** (negative expected return)  
