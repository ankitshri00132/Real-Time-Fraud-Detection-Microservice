import pandas as pd
import joblib
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score,confusion_matrix

from xgboost import XGBClassifier


# load data function
def load_data(path):
    df = pd.read_csv(path)
    return df


# train test split
def split_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=2,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


# calculate imbalance ratio
def compute_scale_pos_weight(y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    return np.sqrt(neg/pos)

# building preprocessor
def _preprocessor():
    numeric_features = ['Amount', 'Time']

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features)
        ],
        remainder="passthrough"
    )

    return preprocessor


# building model
def _model(scale_pos_weight):
    model = XGBClassifier(n_estimators=200,
                          colsample_bylevel=0.8,
                          learning_rate=0.3,
                          max_depth=4,
                          subsample=0.9,
                          scale_pos_weight=scale_pos_weight,
                          random_state=42,
                          n_jobs=-1
                          )
    return model


# building full pipeline
def _pipeline(preprocessor,model):

    pipeline = Pipeline(
        steps=[
            ("preprocessing",preprocessor),
            ("model",model)
        ]
    )

    return pipeline


# evaluate model 
def evaluate_model(model,X_test,y_test):
    y_prob = model.predict_proba(X_test)[:,1]

    threshold = 0.11

    y_pred = (y_prob>=threshold).astype(int)

    print("Classification Report : \n",classification_report(y_test,y_pred))

    roc = roc_auc_score(y_test,y_prob)
    pr = average_precision_score(y_test,y_prob)
    cm = confusion_matrix(y_test,y_pred)

    print("ROC-AUC : ",roc)
    print("PR-AUC",pr)
    print("Confusion Matrix :\n",cm)


# saving pipeline
def save_model(pipeline,name):

    joblib.dump(pipeline,f"models/{name}")
    print("!! Pipeline Saved Successfully !! ")



# main training function which orchestrate everything

def main():
    data_path = "data/creditcard.csv"

    df = load_data(data_path)

    X_train,X_test,y_train,y_test = split_data(df)

    scale_pos_weight = compute_scale_pos_weight(y_train)

    preprocessor = _preprocessor()

    model = _model(scale_pos_weight)

    pipeline = _pipeline(preprocessor,model)
    print('Training Model \n')
    pipeline.fit(X_train,y_train)
    print("Model Trained\n")

    evaluate_model(pipeline,X_test,y_test)

    save_model(pipeline,'fraud_pipeline_v01.pkl')


if __name__ == "__main__":
    main()