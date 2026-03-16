# Day 4 Summary  

- The goal was to understand how decision threshold affects fraud detection performance and to determine whether threshold tuning is more effective than resampling techniques like SMOTE or class weighing.  

- In fraud detection:

  - Dataset is extremely imbalanced (0.17% fraud)
  - Cost of False Negative >> Cost of False Positive
  - Therefore, default threshold is usually not optimal.  

##

- Used predicted probabilities from Logistic Regression.

- Evaluated multiple thresholds from 0.0 to 1.0.

- Calculated Precision, Recall, and F1 at each threshold.

- Created a Threshold vs Metrics table.

- Selected best threshold based on F1 score.

### Observation 1

        At threshold = 0.5:

        Recall ≈ 70%

        Precision ≈ 88%

        At threshold = 0.1:

        Recall ≈ 81%

        Precision ≈ 83%

        Lowering threshold improved recall significantly while only         slightly reducing precision.

        This provided a better balance.

### Observation 2 : Threshold Tuning vs SMOTE

        SMOTE / class_weight:

        Recall ≈ 95%

        Precision ≈ 5%

        Thousands of false positives

        Threshold tuning:

        Recall ≈ 81%

        Precision ≈ 83%

        Very few false positives (16)

Conclusion : Threshold tuning is more stable and controlled compared to aggressive resampling.  

## Core Learning from Day 4

- Fraud detection is a threshold optimization problem.

- High ROC-AUC means good ranking ability.

- PR-AUC is more informative than ROC-AUC in imbalance.

- Resampling is not always necessary.

- Default threshold (0.5) should never be blindly used.

- Professional fraud systems choose threshold based on business cost.
