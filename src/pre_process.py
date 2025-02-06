import os
import pandas as pd
import yaml
import utils  
from sklearn.model_selection import train_test_split

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_data(path):
    return pd.read_csv(os.path.join(path, "survey.csv"))

def preprocess_data(df):
    df = utils.remove_duplicates(df)
    df.drop(columns=["Timestamp", "comments", "state", "Country", "no_employees"], inplace=True)
    df["Gender"] = df["Gender"].apply(utils.clean_gender)

    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df = utils.remove_outliers_iqr(df, numerical_columns)
    df = utils.fill_missing_values(df)
    df["treatment"] = df["treatment"].replace({"Yes": 1, "No": 0})
    df.to_csv("check.csv")
    return df

def split_and_save_data(X, y, config):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dir = os.path.dirname("./data/gold")

    os.makedirs(train_dir, exist_ok=True)
    train_data = pd.DataFrame(X_train)
    train_data['target'] = y_train
    train_data.to_csv(config["data"]["train"], index=False)
    
    test_data = pd.DataFrame(X_test)
    test_data['target'] = y_test                                                  
    test_data.to_csv(config["data"]["test"], index=False)

    print("Data saved successfully!")

def main():
    config = load_config()
    df = load_data(os.path.join(config["data"]["raw"]))
    df = preprocess_data(df)
    X, y = utils.transform_data(df,os.path.join("../pre_process/preprocessing_pipeline.pkl"))
    X_res, y_res = utils.balance_data(X, y)

    split_and_save_data(X_res, y_res, config)

if __name__ == "__main__":
    main()
