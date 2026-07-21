import os
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Customer_Churn_Basic")
mlflow.autolog()

def load_data():
    df = pd.read_csv('namadataset_preprocessing/customer_churn_preprocessing.csv')
    X = df.drop(columns=['churn'])
    y = df['churn']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model():
    X_train, X_test, y_train, y_test = load_data()
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Metrics: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

if __name__ == "__main__":
    train_model()
