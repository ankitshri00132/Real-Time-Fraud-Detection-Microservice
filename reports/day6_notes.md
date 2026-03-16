# Day 6 Summary 

- Hyperparameter tuning improves ensemble models

- efault parameters rarely produce optimal fraud detection performance.

- hreshold tuning is critical even after tuning models

- Optimal thresholds were:

    - Random Forest → 0.16
    - Gradient Boosting → 0.14  

Far from the default 0.5.

- Precision–Recall tradeoff defines fraud detection systems

- Different thresholds create different operational behaviors.

- PR-AUC is the most reliable ranking metric as accuracy remained ~99.9% for all models but was meaningless. PR-AUC better captured fraud detection capability.

## Random Forest performed better initially because it is a bagging-based ensemble that reduces variance and is robust to noisy datasets. It also works very well with independent features like PCA components. Gradient Boosting, on the other hand, learns sequentially and is more sensitive to hyperparameters. With default parameters it may underperform until properly tuned.