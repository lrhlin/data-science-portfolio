# Extracting Targeted Sentiment from Tweets

## Overview
This project focuses on a constrained NLP task: extracting the specific text span within a tweet
that best represents its sentiment, given the tweet and its sentiment label.

The task highlights challenges common in real-world NLP, including short and noisy text,
ambiguous sentiment boundaries, and limited labeled data.

## Data & Task
- Input: tweet text and sentiment label (positive, neutral, negative)
- Output: character-level sentiment-relevant text span
- Evaluation: Jaccard similarity between predicted and true spans


## Approach
I explored multiple modeling strategies and their tradeoffs:
- Rule-based baselines for neutral sentiment
- Token-level sequence labeling models
- Transformer-based span prediction models
- Post-processing heuristics informed by error analysis

The emphasis was on understanding task constraints rather than leaderboard optimization.


## Key Observations
- Annotation ambiguity often limits performance more than model capacity
- Simple heuristics can outperform complex models in specific edge cases
- Error analysis is essential for interpreting evaluation metrics on noisy text


## Notes
- This project is included as a technical supplement to the portfolio
- Kaggle competition context is provided for reference only
