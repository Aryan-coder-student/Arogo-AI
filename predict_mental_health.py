import os 
import pickle
import pandas as pd 
import numpy as np
import yaml 
import glob 
from sklearn.metrics import precision_score
import joblib


config = yaml.safe_load(open("config.yaml"))["models"]
id2label = {
        'Gender': {0: 'Female', 1: 'Male', 2: 'Non-binary'},
        'self_employed': {0: 'No', 1: 'Yes'},
        'family_history': {0: 'No', 1: 'Yes'},
        'treatment': {0: 'No', 1: 'Yes'},
        'work_interfere': {0: 'Never', 1: 'Often', 2: 'Rarely', 3: 'Sometimes'},
        'remote_work': {0: 'No', 1: 'Yes'},
        'tech_company': {0: 'No', 1: 'Yes'},
        'benefits': {0: "Don't know", 1: 'No', 2: 'Yes'},
        'care_options': {0: 'No', 1: 'Not sure', 2: 'Yes'},
        'wellness_program': {0: "Don't know", 1: 'No', 2: 'Yes'},
        'seek_help': {0: "Don't know", 1: 'No', 2: 'Yes'},
        'anonymity': {0: "Don't know", 1: 'No', 2: 'Yes'},
        'leave': {0: "Don't know", 1: 'Somewhat difficult', 2: 'Somewhat easy', 3: 'Very difficult', 4: 'Very easy'},
        'mental_health_consequence': {0: 'Maybe', 1: 'No', 2: 'Yes'},
        'phys_health_consequence': {0: 'Maybe', 1: 'No', 2: 'Yes'},
        'coworkers': {0: 'No', 1: 'Some of them', 2: 'Yes'},
        'supervisor': {0: 'No', 1: 'Some of them', 2: 'Yes'},
        'mental_health_interview': {0: 'Maybe', 1: 'No', 2: 'Yes'},
        'phys_health_interview': {0: 'Maybe', 1: 'No', 2: 'Yes'},
        'mental_vs_physical': {0: "Don't know", 1: 'No', 2: 'Yes'},
        'obs_consequence': {0: 'No', 1: 'Yes'}
    }
label2id = {
    'Gender': {'Female': 0, 'Male': 1, 'Non-binary': 2},
    'self_employed': {'No': 0, 'Yes': 1},
    'family_history': {'No': 0, 'Yes': 1},
    'treatment': {'No': 0, 'Yes': 1},
    'work_interfere': {'Never': 0, 'Often': 1, 'Rarely': 2, 'Sometimes': 3},
    'remote_work': {'No': 0, 'Yes': 1},
    'tech_company': {'No': 0, 'Yes': 1},
    'benefits': {"Don't know": 0, 'No': 1, 'Yes': 2},
    'care_options': {'No': 0, 'Not sure': 1, 'Yes': 2},
    'wellness_program': {"Don't know": 0, 'No': 1, 'Yes': 2},
    'seek_help': {"Don't know": 0, 'No': 1, 'Yes': 2},
    'anonymity': {"Don't know": 0, 'No': 1, 'Yes': 2},
    'leave': {"Don't know": 0, 'Somewhat difficult': 1, 'Somewhat easy': 2, 'Very difficult': 3, 'Very easy': 4},
    'mental_health_consequence': {'Maybe': 0, 'No': 1, 'Yes': 2},
    'phys_health_consequence': {'Maybe': 0, 'No': 1, 'Yes': 2},
    'coworkers': {'No': 0, 'Some of them': 1, 'Yes': 2},
    'supervisor': {'No': 0, 'Some of them': 1, 'Yes': 2},
    'mental_health_interview': {'Maybe': 0, 'No': 1, 'Yes': 2},
    'phys_health_interview': {'Maybe': 0, 'No': 1, 'Yes': 2},
    'mental_vs_physical': {"Don't know": 0, 'No': 1, 'Yes': 2},
    'obs_consequence': {'No': 0, 'Yes': 1}
}


def predict_mental_health(data ,labels=None):
    result_dict = dict()
    model_dir = config["loc"]
    
    for model_file in os.listdir(model_dir):
        if model_file.endswith(".pkl"):  
            model_path = os.path.join(model_dir, model_file)
            model = joblib.load(model_path)
            
            
            predictions = model.predict(data)

            
            confidence_scores = None
            if hasattr(model, "predict_proba"):  
                confidence_scores = model.predict_proba(data)[:, 1]  
            
            
            precision = None
            if labels is not None:
                precision = precision_score(labels, predictions, pos_label=1)

            result_dict[model_file] = {
                "prediction": predictions.tolist(),
                "confidence": confidence_scores.tolist() if confidence_scores is not None else None,
                "precision": precision
            }
    
    return result_dict


def clean_gender(gender):
    gender = str(gender).strip().lower()
    male_variants = ["m", "male", "man", "cis male", "cis man", "maile", "mal", "malr", "male (cis)", "guy (-ish) ^_^", "male-ish", "maile" ,"msle", "mail", "malr"]
    female_variants = ["f", "female", "woman", "cis female", "cis-female/femme", "female (cis)", "femake", "trans woman", "female (trans)"]
    non_binary_variants = [
        "non-binary", "genderqueer", "fluid", "androgyne", "agender", "enby", "queer/she/they", "something kinda male?",
        "ostensibly male, unsure what that really means", "male leaning androgynous"
    ]
    
    if gender in male_variants:
        return "Male"
    elif gender in female_variants:
        return "Female"
    elif gender in non_binary_variants:
        return "Non-binary"
    elif gender in {"nah", "all", "a little about you", "p"}:
        return "Unknown"
    else:
        return "Other"


def preprocess_and_predict(input_data):
    processed_data = input_data
    for key, value in input_data.items():
        if key == "Gender":
            value = clean_gender(value)
            processed_data[key] = value  

    # print("Processed Data:", processed_data)

    df = pd.DataFrame([processed_data])
    # print(df.dtypes)
    pipeline = joblib.load(config["pre_process_pipeline"])
    processed_df = pipeline.transform(df)
    print("Processed DF:", processed_df)
    predictions = predict_mental_health(processed_df)

    return predictions

if __name__ == "__main__":
    sample_input = {
    "Age": 21.0,
    "Gender": "Male",
    "self_employed": "No",
    "family_history": "Yes",
    "work_interfere": "Sometimes",
    "remote_work": "Yes",
    "tech_company": "No",
    "benefits": "Yes",
    "care_options": "Yes",
    "wellness_program": "No",
    "seek_help": "Yes",
    "anonymity": "Don't know",
    "leave": "Somewhat easy",
    "mental_health_consequence": "Maybe",
    "phys_health_consequence": "No",
    "coworkers": "Some of them",
    "supervisor": "Yes",
    "mental_health_interview": "No",
    "phys_health_interview": "Maybe",
    "mental_vs_physical": "Yes",
    "obs_consequence": "No"
}
    results = preprocess_and_predict(sample_input)
    print(results)