import os
import joblib
import json
import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Load Configurations
config_path = "config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

model_save_dir = config["models"]["loc"]

# Ensure results directory exists
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs("results", exist_ok=True)

# Define Parameter Grid
param_grids = {
    "Logistic Regression": (LogisticRegression(), {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}),
    "Random Forest": (RandomForestClassifier(), {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30]}),
    "Gradient Boosting": (GradientBoostingClassifier(), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5]}),
    "SVM": (SVC(probability=True), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf", "poly"]}),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}),
    "Decision Tree": (DecisionTreeClassifier(), {"max_depth": [None, 10, 20, 30]}),
    "Naive Bayes": (GaussianNB(), {})
}

 
model_results = {}

import os
import yaml
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
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

# Load Configurations using yaml module
config_path = "config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Extract Paths from the YAML file
model_save_dir = config["models"]["loc"]
roc_auc_path = config["metrics"]["roc_auc"]
conf_matrix_path = config["metrics"]["confusion_matrix"]
class_report_path = config["metrics"]["classification_report"]
feature_importance_path = config["metrics"]["feature_importance"]

# Ensure all necessary directories exist using os module
for path in [model_save_dir, roc_auc_path, conf_matrix_path, class_report_path, feature_importance_path]:
    os.makedirs(path, exist_ok=True)

# Define Parameter Grid for different models
param_grids = {
    "Logistic Regression": (LogisticRegression(), {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}),
    "Random Forest": (RandomForestClassifier(), {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30]}),
    "Gradient Boosting": (GradientBoostingClassifier(), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5]}),
    "SVM": (SVC(probability=True), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf", "poly"]}),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}),
    "Decision Tree": (DecisionTreeClassifier(), {"max_depth": [None, 10, 20, 30]}),
    "Naive Bayes": (GaussianNB(), {})
}

# Initialize Results Dictionary
model_results = {}

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

     
            model_results[name] = {
                "Best Params": grid_search.best_params_,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC AUC": roc_auc if roc_auc is not None else "N/A"
            }

     
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", roc_auc)

 
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix - {name}")
            plt.savefig(f"{conf_matrix_path}/{name}_confusion_matrix.png")
            plt.close()
 
            class_report = classification_report(y_test, y_pred, output_dict=True)
            class_report_df = pd.DataFrame(class_report).transpose()
            plt.figure(figsize=(8, 5))
            sns.heatmap(class_report_df.iloc[:-1, :-1], annot=True, cmap="coolwarm", fmt=".2f")
            plt.title(f"Classification Report - {name}")
            plt.savefig(f"{class_report_path}/{name}_classification_report.png")
            plt.close()

    
            if roc_auc is not None:
                fpr, tpr, _ = roc_curve(y_test, y_probs[:, 1])
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, label=f"AUC: {roc_auc:.2f}", color="darkorange")
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve - {name}")
                plt.legend()
                plt.savefig(f"{roc_auc_path}/{name}_roc_auc_curve.png")
                plt.close()

 
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

                importance_df.to_csv(f"{feature_importance_path}/{name}_feature_importance.csv", index=False)

 
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(best_model, name, signature=signature)

    
            joblib.dump(best_model, os.path.join(model_save_dir, f"{name}_model.pkl"))
            print(f"Model {name} trained and logged to MLflow")
 
    with open("results/model_evaluation.json", "w") as json_file:
        json.dump(model_results, json_file, indent=4)
    print("Model evaluation results saved to results/model_evaluation.json")
