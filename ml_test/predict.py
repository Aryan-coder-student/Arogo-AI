import os
import joblib
from sklearn.metrics import precision_score
import yaml

config = yaml.safe_load(open("config.yaml"))["models"]

def predict_mental_health(data, labels=None):
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
