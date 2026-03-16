# Day 8 Summary

**Objective :** `Explain the fraud detection model using SHAP.`

**Key Observations :**

- V14 is the most important feature influencing fraud predictions.

- Low values of V14 strongly increase fraud probability.

- V4, V12, and V3 also contribute significantly.

- Transaction Amount and Time have smaller influence compared to PCA features.

**Key Insight:**

The model identifies fraud using hidden behavioral patterns captured by PCA components, rather than simple variables like transaction amount

**Observation about Transaction Amount:**

The SHAP analysis shows that the transaction amount is not among the most important fraud indicators. This is because fraud detection relies more on behavioral patterns rather than simple transaction size. Fraudsters often perform transactions with moderate amounts to avoid detection by rule-based systems. Additionally, the dataset uses PCA-transformed features (V1–V28) which capture complex relationships between multiple hidden transaction attributes such as spending behavior, merchant patterns, and transaction context. These behavioral signals provide stronger indicators of fraud than the raw transaction amount

## Multiple Threshold Strategy  

In real fraud detection systems, models typically use multiple thresholds instead of a single classification cutoff. Transactions with very high fraud probability are automatically blocked, while moderately suspicious transactions are sent for manual review by fraud analysts. Transactions with low fraud probability are allowed normally. This approach balances fraud prevention, customer experience, and operational workload

## Things learned till now

```
EDA
Class imbalance handling
Evaluation metrics
Threshold optimization
Ensemble models
Hyperparameter tuning
XGBoost modeling
Model explainability (SHAP)
Business interpretation
```