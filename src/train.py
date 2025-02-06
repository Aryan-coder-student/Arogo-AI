import os
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from model import train_and_evaluate_models


config_path = "config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

data_config = config["data"]

train_path = data_config["train"]
test_path = data_config["test"]


df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values
X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values
feature_names = df_train.columns[:-1]


mlflow.set_experiment("Model_Training_Experiment")


train_and_evaluate_models(X_train, y_train, X_test, y_test, feature_names)

print("Model training and evaluation complete.")
