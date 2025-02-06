import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from flask import Flask, request, jsonify
# from flask_cors import CORS
from ml_test.preprocess import preprocess_data
from ml_test.predict import predict_mental_health


app = Flask(__name__)
# CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:

        input_data = request.get_json()
        processed_data = preprocess_data(input_data)
        predictions = predict_mental_health(processed_data)

        return jsonify(predictions), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
