
# Day 3 Summary

- After using LR + class_weights, LR + under_sampling , LR + over_sampling (SMOTE), all of these increased my recall aggresively but my precision fell hard too.  

- Almost all models increased Recall as it were all same and reduced precision same too , only LR+under_sampling reduced precision more than both of them & increase avg_precision more than them too.

- None of them looks stable.

## Conclusion  

- Resampling changes decision boundary.

- High recall often destroys precision.

- PR-AUC staying high means model still has signal.

- Fraud modeling is `threshold optimization` problem.  
