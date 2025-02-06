import pandas as pd
import numpy as np
import pickle
import mlflow
import os 
import mlflow.sklearn
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def remove_duplicates(df):
    
    mlflow.log_param("duplicates_removed", df.duplicated().sum())
    return df.drop_duplicates()

def clean_gender(gender):
     
    gender = str(gender).strip().lower()
    male_variants = {"m", "male", "man", "cis male", "cis man", "male (cis)", "guy (-ish)", "male-ish", "msle", "mail"}
    female_variants = {"f", "female", "woman", "cis female", "cis-female/femme", "female (cis)", "trans woman"}
    non_binary_variants = {
        "non-binary", "genderqueer", "fluid", "androgyne", "agender", "enby", "queer/she/they"
    }

    standardized_gender = "Male" if gender in male_variants else "Female" if gender in female_variants else "Non-binary" if gender in non_binary_variants else "Other"
    
    return standardized_gender

def remove_outliers_iqr(df, columns):
     
    initial_size = len(df)
    for col in columns:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    final_size = len(df)
    mlflow.log_param("outliers_removed", initial_size - final_size)
    return df

def fill_missing_values(df):
     
    missing_before = df.isnull().sum().sum()
    
    df["self_employed"] = df["self_employed"].fillna(df["self_employed"].mode()[0])
    df["work_interfere"] = df["work_interfere"].fillna(df["work_interfere"].mode()[0])

    missing_after = df.isnull().sum().sum()
    mlflow.log_param("missing_values_filled", missing_before - missing_after)
    return df

def transform_data(df, pipeline_path):
    
    df_new = df.drop(columns=["treatment"])
    categorical_columns = list(df_new.select_dtypes(include=['object']).columns)
    numerical_columns = list(df_new.select_dtypes(exclude=['object']).columns)

    column_transformer = ColumnTransformer(
        transformers=[
            ("cat_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_columns),
            ("num_scaler", MinMaxScaler(), numerical_columns)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[("transform", column_transformer)])
    pipeline.fit(df_new)

     
    mlflow.sklearn.log_model(pipeline, "preprocessing_pipeline")
    os.makedirs("../pre_process", exist_ok=True)
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)

    transformed_data = pipeline.transform(df_new)
    transformed_df = pd.DataFrame(transformed_data, columns=categorical_columns + numerical_columns)

    return transformed_df, df["treatment"].values

def balance_data(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    mlflow.log_param("data_balanced", len(X_res) - len(X))
    return X_res, y_res
