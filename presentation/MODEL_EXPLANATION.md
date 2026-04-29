# Model Explanation

## Project Goal

The goal of this project is to predict upcoming NBA player statistics using machine learning. Instead of only relying on simple season averages, the project uses past player performance, recent trends, matchup history, and game context to estimate how a player may perform in future games.

The project predicts stat lines for upcoming games and also provides confidence scores, which helps users understand how reliable a prediction may be.

## Machine Learning Approach

This project uses supervised learning because the model is trained on past NBA games where the actual player statistics are already known. The model learns from previous games and then uses those patterns to make predictions for future games.

The project uses a Gradient Boosting regression model from scikit-learn. Regression is used because the goal is to predict numerical values, such as points, assists, rebounds, steals, blocks, field goal percentage, and three-point percentage.

Gradient Boosting is an ensemble method, meaning it combines multiple decision trees. Each tree helps correct mistakes from the previous trees. This works well for tabular sports data because the model can learn from different player, matchup, and game-context features.

A separate model is trained for each statistic. This means the points model is separate from the assists model, rebounds model, and so on. This is useful because different basketball stats can depend on different patterns.

## Model Configuration

The project uses scikit-learn's `GradientBoostingRegressor` with this setup:

- `n_estimators = 200`
- `max_depth = 3`
- `learning_rate = 0.05`
- `subsample = 0.8`
- `loss = squared_error`

The model uses 200 shallow decision trees with a max depth of 3. The lower learning rate helps the model learn gradually, while the subsample value lets each tree train on part of the data. This setup helps balance accuracy and overfitting.

## Features Used by the Model

The model uses several features to make predictions:

- Season averages up to the current point in the season
- Last 3-game averages to represent recent performance
- Previous performance against the same opponent
- Home or away game status
- Days of rest before the game
- Number of games already played

These features give the model more context than a basic season average. For example, a player may have a strong season average but could be in a recent slump, or a player may perform differently depending on the opponent.

## Why These Features Matter

Season averages help represent a player's overall performance level. Recent averages help capture whether a player has been performing above or below their usual level. Matchup history can show whether a player has done well or poorly against a specific opponent.

Home or away status and rest days can also affect performance. A player may perform differently at home compared to away games, and rest can impact energy, minutes played, and overall production.

The number of games played also acts as a sample-size feature. A player with more games played gives the model more reliable information than a player with very limited current-season data.

## Temporal Validation

The project sorts games chronologically before training and testing. This is important because the model should not train on future games when predicting earlier games.

This is called temporal validation. It better represents a real prediction setting because only past data is available when making a prediction about an upcoming game.

## Prediction Output

The project predicts a stat line for upcoming games. The prediction includes a rounded value, a more precise predicted value, a confidence estimate, and an expected range.

For example, a prediction may show a rounded line like 35 points, a more precise model output like 34.8 points, and a confidence percentage showing how likely the player is to reach that line.

This helps users understand both the model's prediction and the uncertainty around that prediction.

## Confidence Scoring

The confidence score is calculated using a normal distribution centered around the model's predicted value. The project uses the formula:

```text
P(stat >= line) = 1 - Φ((line - 0.5 - prediction) / σ)
