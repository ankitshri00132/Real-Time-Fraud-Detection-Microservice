# Day 9 Summary

## Machine Learning Pipeline

- A machine learning pipeline combines preprocessing and model training steps into a single reproducible workflow. Pipelines prevent data leakage by ensuring transformations are fitted only on training data and applied consistently to new data. They also simplify deployment by packaging preprocessing and the trained model into one object that can be saved and reused.

- In this project, the pipeline includes scaling of the `Amount` and `Time` features followed by the XGBoost fraud detection model
