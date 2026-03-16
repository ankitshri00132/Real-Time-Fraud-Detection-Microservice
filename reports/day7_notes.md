# Day 7 Summary

- XGBoost improves classical Gradient Boosting.

- Instead of modifying the dataset, XGBoost adjusts the learning objective to prioritize fraud samples using `scale_pos_weight`

- Hyperparameter tuning significantly improved performance.Important parameters include: learning_rate, max_depth, n_estimators,subsample, colsample_bylevel

- Optimal Threshold found : 0.22
This improved recall while maintaining strong precision

```
PR-AUC ≈ 0.885
Recall ≈ 0.878
Precision ≈ 0.851
```
