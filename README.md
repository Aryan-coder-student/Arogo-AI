I've updated the **README.md** with your contact details, GitHub profile, and project repository link. Here's the final version:  

```markdown
# Machine Learning Model Training Pipeline

## Project Overview
This project focuses on training, evaluating, and predicting mental health treatment needs using machine learning models. It includes:
- **Data Preprocessing**
- **Model Training & Hyperparameter Tuning**
- **Flask API for Real-Time Predictions**
- **Streamlit UI for Interactive Use**
- **DVC for Data Version Control**
- **MLflow for Experiment Tracking**

---

## 📂 File Descriptions

### 🛠 **Main Components**
- `pre_process.py` - Cleans and encodes survey data.
- `train.py` - Trains multiple ML models.
- `model.py` - Defines ML models & hyperparameter tuning.
- `utils.py` - Helper functions for data transformation.
- `predict_mental_health.py` - Loads trained models for predictions.
- `app.py` - Flask API for real-time predictions.
- `streamlit_app.py` - Interactive web app using **Streamlit**.
- `config.yaml` - Configuration file for paths & model settings.
- `dvc.yaml` - Defines the **DVC pipeline** for automation.

---

## 📊 **DVC Pipeline**
This **DVC pipeline** automates preprocessing and model training:

```yaml
stages:
  requirements:
    cmd: pip install -r requirements.txt
    deps:
      - requirements.txt

  preprocess:
    cmd: python src/pre_process.py
    deps:
      - src/pre_process.py
      - src/utils.py
      - data/bronze/survey.csv
    outs:
      - data/gold/train.csv
      - data/gold/test.csv

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - src/model.py
      - data/gold/train.csv
      - data/gold/test.csv
    outs:
      - models/
```

---

## 🚀 **How to Run the Project**

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the **DVC Pipeline**
```bash
dvc repro
```

### 3️⃣ Start the **Flask API**
```bash
python app.py
```
**API Endpoint:** `http://localhost:5000/predict`

### 4️⃣ Run the **Streamlit App**
```bash
streamlit run streamlit_app.py
```
**UI:** `http://localhost:8501`

---

## 📡 **Making API Predictions**
Use **Postman** or **cURL** to send a **POST** request to `/predict`:

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

#### **cURL Command:**
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @input.json
```

---

## 📺 **Screenshots & Videos**
### 🖼 **Streamlit App UI**
![Streamlit Screenshot](screenshots/streamlit_ui.png)

### 🎥 **Demo Video**
[Watch Demo](videos/demo.mp4)

---

## 📊 **Model Evaluation & Insights**
The **Streamlit app** provides visualizations for:
- 📌 **LIME-based explanations**
- 📌 **Feature importance graphs**
- 📌 **Confusion matrices**
- 📌 **ROC curves & AUC scores**
- 📌 **Classification reports**

📍 **Example: Feature Importance Graph**
![Feature Importance](screenshots/feature_importance.png)

---

## ✅ **Tracking with Git & DVC**
```bash
git add .
git commit -m "Added Streamlit app and model evaluation"
dvc push
```

---

## 🔧 **Dependencies**
- **Python** (>=3.8)
- **Flask** - API development
- **Streamlit** - Interactive UI
- **Pandas, NumPy, Scikit-learn, Imbalanced-learn**
- **MLflow** - Experiment tracking
- **DVC** - Data version control
- **Matplotlib, Seaborn** - Data visualization
- **LIME** - Model explanations

---

## 📌 **Future Enhancements**
✅ Deploy API & UI to **AWS/GCP/Azure**  
✅ Add **Docker support**  
✅ Improve **UI aesthetics**  
✅ Integrate **FastAPI** for faster response times  

---

## ✨ **Author**
👨‍💻 **Aryan Pahari**  
📧 **Email**: aryanpahari037@gmail.com  
🔗 **GitHub**: [Aryan-coder-student](https://github.com/Aryan-coder-student)  
📂 **Project Repo**: [Arogo AI](https://github.com/Aryan-coder-student/Arogo-AI.git)  

---

🚀 **Happy Predicting!** 🎯
```

### 🔥 **What’s Updated?**
✅ **Your GitHub Profile & Repo**  
✅ **Your Email Contact**  
✅ **Refined Formatting**  
✅ **Step-by-Step Guide for API & UI**  
✅ **Screenshots & Video Placeholders**  

Let me know if you need more tweaks! 🚀💡