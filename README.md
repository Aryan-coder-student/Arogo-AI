Got it! I’ve ensured that **each file description** is included and clearly highlighted in the updated `README.md`. Here's the revised version with all file descriptions intact and organized for better readability:

---

```markdown
# 🧠 Mental Health Treatment Prediction: ML Pipeline

![Project Banner](screenshots/banner.png) <!-- Add a banner image here -->

A **machine learning pipeline** for predicting mental health treatment needs, featuring **data preprocessing**, **model training**, **experiment tracking**, **Flask API**, and an **interactive Streamlit app**.

---

## 🚀 **Features**

- **Data Preprocessing**: Clean, encode, and balance datasets using **SMOTE**.
- **Model Training**: Train and evaluate multiple ML models (Logistic Regression, Random Forest, SVM, etc.).
- **Hyperparameter Tuning**: Optimize models using **GridSearchCV**.
- **Experiment Tracking**: Log experiments with **MLflow**.
- **Flask API**: Real-time predictions via REST API.
- **Streamlit App**: Interactive UI for predictions and model insights.
- **DVC Pipeline**: Automate workflows with **DVC** for reproducibility.

---

## 📂 **File Descriptions**

### **1. `pre_process.py`**
- Loads and preprocesses raw data.
- Cleans and encodes categorical values.
- Handles missing values and outliers.
- Balances the dataset using **SMOTE**.
- Saves the processed dataset in `./data/gold/`.

### **2. `train.py`**
- Loads the processed dataset.
- Splits data into `train` and `test` sets.
- Trains multiple machine learning models.
- Logs experiment tracking using **MLflow**.
- Saves trained models to `./models/`.

### **3. `model.py`**
- Defines various **ML models**:
  - Logistic Regression
  - Random Forest
  - SVM
  - KNN
  - Decision Tree
  - Gradient Boosting
  - Naive Bayes
- Uses **GridSearchCV** for hyperparameter tuning.
- Logs model performance (Accuracy, Precision, Recall, F1-score, ROC AUC).

### **4. `utils.py`**
- Contains helper functions for:
  - Data cleaning (`remove_duplicates`, `fill_missing_values`, `clean_gender`).
  - Feature transformation (`transform_data`).
  - Dataset balancing (`balance_data`).

### **5. `predict_mental_health.py`**
- Loads trained models from the `models/` directory.
- Preprocesses input data using the saved pipeline.
- Predicts **mental health treatment needs** for new data.
- Returns model confidence scores and precision metrics.

### **6. `app.py` (Flask API)**
- A **REST API** for real-time predictions.
- Accepts **POST** requests at `/predict`.
- Returns JSON responses with:
  - **Prediction** (Treatment Needed / Not Needed)
  - **Confidence Score**
  - **Model Performance Summary**

### **7. `streamlit_app.py` (Interactive Web App)**
- A **Streamlit-based UI** for real-time predictions and model evaluation.
- Features:
  - **User-friendly form** for entering mental health survey data.
  - **Live predictions** from multiple models.
  - **Visualization tabs** for:
    - **LIME explanations**
    - **Feature importance**
    - **Confusion matrices**
    - **ROC curves**
    - **Classification reports**
  - **Interactive sidebar** for model performance metrics.
  - **Dark-themed UI** with responsive styling.

---

## 🛠️ **How to Run**

### 1️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2️⃣ **Set Up DVC**
Initialize DVC and add remote storage (optional):
```bash
dvc init
```

### 3️⃣ **Run the DVC Pipeline**
Execute the full pipeline:
```bash
dvc repro
```

### 4️⃣ **Start the Flask API**
Run the API server:
```bash
python src/app.py
```
Access the API at:
```
http://localhost:5000/predict
```

### 5️⃣ **Launch the Streamlit App**
Start the interactive UI:
```bash
streamlit run src/streamlit_app.py
```
Access the app at:
```
http://localhost:8501
```

---

## 📡 **API Usage**

Send a **POST** request to `/predict` with the following JSON payload:

```json
{
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
```

#### Example with **cURL**:
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @input.json
```

---

## 🎥 **Demo**

### **Streamlit App UI**
![alt text](image.png)

### **Demo Video**
[Watch Demo](videos/demo.mp4)

---

## 📊 **Model Insights**

The **Streamlit app** provides:
- **Real-time predictions** from multiple models.
- **Visualizations**:
  - Feature importance
  - Confusion matrices
  - ROC curves
  - LIME explanations
- **Performance metrics**:
  - Accuracy, Precision, Recall, F1-score, ROC AUC

#### Example: Feature Importance
![Feature Importance](ec29070c1b1006fa89f69cdb5d9af2a2d4c2eef74c537c0ebd61fe21.png)
---

## 🔧 **Tech Stack**

- **Python** (>=3.8)
- **Pandas, NumPy, Scikit-learn, Imbalanced-learn**
- **MLflow** for experiment tracking
- **Flask** for API development
- **Streamlit** for interactive UI
- **DVC** for data versioning and pipeline automation
- **Matplotlib, Seaborn** for visualizations
- **LIME** for model explanations

---

## 📌 **Future Enhancements**

- **Cloud Deployment**: Deploy API & UI to **AWS/GCP/Azure**.
- **Docker Support**: Containerize the application for easier deployment.
- **UI Improvements**: Add animations and custom CSS for a polished look.
- **FastAPI Integration**: Replace Flask with **FastAPI** for better performance.

---

## ✨ **Author**

**Your Name**  
📧 **Contact**: aryanpahari037@gmail.com 

---

🚀 **Happy Predicting!** 🎯
```

---

### **Key Changes**
1. **File Descriptions Section**: Added a dedicated section for file descriptions, ensuring each file is clearly explained.
2. **Improved Structure**: Organized the content into logical sections for better navigation.
3. **Visual Enhancements**: Added emojis and placeholders for images/videos to make the README more engaging.

Let me know if you need further adjustments! 🚀