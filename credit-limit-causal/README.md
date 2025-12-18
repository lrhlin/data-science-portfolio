# Credit Limit Policy and Default Risk

## Overview
This project studies whether assigning borrowers to higher credit limits causally affects short-term default risk.
Credit limits are a core product and policy lever in consumer finance, but their effects are ambiguous:
higher limits may increase risky borrowing or provide liquidity buffers that reduce default.

Using observational administrative data, I estimate the causal effect of higher credit limits on next-month default risk.


## Data
- Administrative credit card data with borrower demographics, account characteristics, and repayment outcomes
- Outcome: next-month default
- Treatment: assignment to a high credit limit regime
- Key challenge: credit limits are endogenously assigned based on borrower risk


## Methods
To address selection bias in observational data, I use multiple causal inference approaches:
- Propensity score matching
- Inverse probability weighting (IPW)
- Doubly robust estimation

I explicitly define the target estimand and conduct balance checks and robustness analyses across specifications.


## Key Findings
- Higher credit limits reduce short-term default risk on average
- Results are robust across matching, IPW, and doubly robust estimators
- Evidence suggests the effect operates through liquidity buffers rather than increased spending risk


## Notes
- This repository contains analysis code and supporting materials
- Raw data are not shared due to confidentiality
- Full project narrative and visualizations are available on the Notion portfolio
