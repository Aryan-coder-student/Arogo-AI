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
    
    model_save_dir = config["models"]["loc"]
    roc_auc_dir = config["metrics"]["roc_auc"]
    classification_report_dir = config["metrics"]["classification_report"]
    confusion_matrix_dir = config["metrics"]["confusion_matrix"]
    feature_importance_dir = config["metrics"]["feature_importance"]

 
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(roc_auc_dir, exist_ok=True)
    os.makedirs(classification_report_dir, exist_ok=True)
    os.makedirs(confusion_matrix_dir, exist_ok=True)
    os.makedirs(feature_importance_dir, exist_ok=True)

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

         
            report = classification_report(y_test, y_pred, output_dict=True)
            report_path = os.path.join(classification_report_dir, f"{name}_classification_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
            mlflow.log_artifact(report_path)

            
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test))
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(f"Confusion Matrix - {name}")
            cm_path = os.path.join(confusion_matrix_dir, f"{name}_confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)

             
            if roc_auc is not None:
                fpr, tpr, _ = roc_curve(y_test, y_probs[:, 1])
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve - {name}")
                plt.legend()
                roc_path = os.path.join(roc_auc_dir, f"{name}_roc_auc.png")
                plt.savefig(roc_path)
                plt.close()
                mlflow.log_artifact(roc_path)

           
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
        
                feature_path = os.path.join(feature_importance_dir, f"{name}_feature_importance.csv")
                importance_df.to_csv(feature_path, index=False)
                mlflow.log_artifact(feature_path)

            
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(best_model, name, signature=signature)

            model_path = os.path.join(model_save_dir, f"{name}_model.pkl")
            joblib.dump(best_model, model_path)
            print(f"Model {name} trained and logged to MLflow")
