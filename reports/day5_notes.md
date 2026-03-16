# Day 5 Summary 


## Tree Models For Fraud Detection  

Objective : The goal of Day 5 was to evaluate whether tree-based ensemble models can detect fraud transactions better than a simple linear model (Logistic Regression). Specifically, we tested:

- Random Forest

- Gradient Boosting

and compared them with the earlier Logistic Regression baseline.

##

Fraud patterns are rarely linear. Fraudsters often create complex behavioural patterns that linear models cannot capture.

##

Logistic regression (baseline):

```
Recall ≈ 70%
Precision ≈ 88%
PR-AUC ≈ 0.77
```

Random Forest (default parameters) :
```
Recall ≈ 81.6%
Precision ≈ 91.9%
PR-AUC ≈ 0.88

Confusion Matrix : 
TN = 56857
FP = 7
FN = 18
TP = 80
```

- 80 frauds detected
- Only 7 customers blocked

Gradient Boosting (default parameters) :
```
Recall ≈ 76.5%
Precision ≈ 86.2%
PR-AUC ≈ 0.71

Confusion Matrix : 
TN = 56852
FP = 12
FN = 23
TP = 75
```

Performance was slightly worse than Random Forest because RF reduces variance and works well with default parameters too but GB requires parameter tuning to unlock its full potential.

## Key Learning from Day 5

From this experiment I learned:

- Tree-based ensemble models are very effective for fraud detection.

- Random Forest provides a strong baseline even with default parameters.

- Gradient Boosting can perform worse initially if hyperparameters are not tuned.

- Model evaluation should focus on both metrics and confusion matrix interpretation.
