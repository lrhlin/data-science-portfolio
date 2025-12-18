# What Drives Supermarket Sales? Evidence from Favorita Stores

## Overview
This project analyzes large-scale retail transaction data to identify key drivers of supermarket sales
across stores, product categories, and time.

The goal is to understand which factors most consistently explain sales variation in a large retail chain,
rather than to optimize short-term prediction accuracy.


## Data
- Daily sales data by store and product family
- Transaction, promotion, and holiday indicators
- Store location and regional metadata
- External macroeconomic indicators
- Scale: hundreds of thousands of observations


## Methods
To triangulate robust sales drivers, I use a combination of:
- Fixed-effects panel regressions
- Regularized regression (LASSO / Elastic Net)
- Time-series models with exogenous variables
- Store-level clustering to capture regional demand patterns

The focus is on interpretability and robustness across methods.


## Key Findings
- Promotions and seasonality are the dominant drivers of sales
- Effects vary substantially by product category and region
- Macroeconomic conditions correlate with aggregate sales trends
- Simple, interpretable models provide actionable insights for planning


## Notes
- This repository contains cleaned analysis code and documentation
- Full project context is summarized on the Notion portfolio
