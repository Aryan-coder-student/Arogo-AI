import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


config_path = "config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

model_save_dir = config["models"]["loc"]


param_grids = {
    "Logistic Regression": (LogisticRegression(), {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}),
    "Random Forest": (RandomForestClassifier(), {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30]}),
    "Gradient Boosting": (GradientBoostingClassifier(), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5]}),
    "SVM": (SVC(probability=True), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf", "poly"]}),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}),
    "Decision Tree": (DecisionTreeClassifier(), {"max_depth": [None, 10, 20, 30]}),
    "Naive Bayes": (GaussianNB(), {})
}

os.makedirs(model_save_dir, exist_ok=True)

def train_and_evaluate_models(X_train, y_train, X_test, y_test, feature_names):
    for name, (model, params) in param_grids.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")

            grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1, scoring="accuracy")
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_probs = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None

            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")
            roc_auc = roc_auc_score(y_test, y_probs[:, 1]) if y_probs is not None and len(set(y_test)) == 2 else None

            
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", roc_auc)

            
            if hasattr(best_model, "feature_importances_"):
                feature_importance = best_model.feature_importances_
            elif hasattr(best_model, "coef_"):
                feature_importance = np.abs(best_model.coef_).flatten()
            else:
                feature_importance = None

            if feature_importance is not None:
                feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f"Feature {i}" for i in range(len(feature_importance))]
                importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
                importance_df = importance_df.sort_values(by="Importance", ascending=False)
        
                os.makedirs("results", exist_ok=True)
                importance_df.to_csv(os.path.join(f"./feature_importance/{name}_feature_importance.csv"), index=False)


            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(best_model, name, signature=signature)

            joblib.dump(best_model, os.path.join(model_save_dir, f"{name}_model.pkl"))
            print(f"Model {name} trained and logged to MLflow")
