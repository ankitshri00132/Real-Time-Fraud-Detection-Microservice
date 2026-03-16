## Day 1 Summary

- Dataset shape : (284807,31)  

- Fraud percentage : 0.17%

- Why accuracy is misleading ?  
If a model predicts all transactions as non-fraud,there will be high accuracy i.e. 99.83% but it is of no use & it is not acceptable. Because our dataset is highly imbalanced as it has legitmate transaction 99.83% and fraud transaction 0.17%.  
In this case, accuracy should not be used to measure our model because both FP and FN are costly.  

- What makes fraud detection difficult ?  

    1. Extreme Class Imbalance
    2. Adaptive Fraudsters
    3. Complex & Contextual Patterns
    4. High Cost of False Positives
    5. Label Noise & Delayed Detection
    6. Real-Time & Regulatory Constraints  
