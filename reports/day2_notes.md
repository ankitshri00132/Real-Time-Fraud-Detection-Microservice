## Day 2 summary

- If we increase recall aggresively then we'll decrease the precision and it will increase the number of False Positives , which means precision reduces and customer gets blocked wrongly. 

- ROC-AUC can be misleading in extreme imbalance because it evaluates model performance using the False Positive Rate (FPR), which is calculated using the vast number of True Negatives (TN).  
A massive number of negative examples can keep the FPR artificially low even when the model makes many false positive errors, leading to an overly optimistic performance score that fails to reflect poor minority class detection 

- So, we should use PR-AUC metric because it will provide view of our model's performance across all possible decision thresholds, rather than just a single point and it completely focuses on Precision and Recall and it doesn't focus on True Negatives (Non-Fraud).

- Is logistic regression good enough for fraud detection ?  
Logistic regression works well with linear data but Fraudsters are rarely 'Linear'. They uses complex patterns that LR might miss & it is sensitive to outliers.