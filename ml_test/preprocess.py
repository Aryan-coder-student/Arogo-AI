import pandas as pd
from ml_test.utils import clean_gender
import joblib
import yaml 
config = yaml.safe_load(open("config.yaml"))["models"]
def preprocess_data(input_data):
    processed_data = input_data
    for key, value in input_data.items():
        if key == "Gender":
            value = clean_gender(value)
            processed_data[key] = value

    df = pd.DataFrame([processed_data])
    pipeline = joblib.load(config["pre_process_pipeline"])
    # print(pipeline)
    processed_df = pipeline.transform(df)
    return processed_df