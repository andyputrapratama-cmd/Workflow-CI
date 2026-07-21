import os
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri("https://dagshub.com/AndyPutraPratama/Eksperimen_SML_Andy-Putra-Pratama.mlflow")
mlflow.set_experiment("Customer_Churn_Tuning")

def load_data():
    df = pd.read_csv('namadataset_preprocessing/customer_churn_preprocessing.csv')
    X = df.drop(columns=['churn'])
    y = df['churn']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model():
    X_train, X_test, y_train, y_test = load_data()
    
    with mlflow.start_run():
        params = {"n_estimators": 200, "max_depth": 10, "random_state": 42}
        mlflow.log_params(params)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)
        
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        
        print("Model tuning and manual logging complete.")

if __name__ == "__main__":
    train_model()
