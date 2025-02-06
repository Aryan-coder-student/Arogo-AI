import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Mental Health Prediction",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stSelectbox, .stNumberInput {
        background-color: #262730;
    }
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background-color: #262730;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


tab1, tab2 = st.tabs(["Prediction", "Model Metrics"])

with tab1:
    
    st.title("🧠 Mental Health in Tech Predictor")
    st.markdown("### Predict mental health treatment needs using machine learning")

    
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        family_history = st.selectbox("Family History of Mental Health", ["No", "Yes"])
        work_interfere = st.selectbox("Work Interference", ["Never", "Rarely", "Sometimes", "Often"])
        remote_work = st.selectbox("Remote Work", ["No", "Yes"])
        tech_company = st.selectbox("Tech Company", ["No", "Yes"])
        benefits = st.selectbox("Benefits", ["No", "Yes", "Don't know"])
        care_options = st.selectbox("Care Options", ["No", "Yes", "Not sure"])
        wellness_program = st.selectbox("Wellness Program", ["No", "Yes", "Don't know"])

    with col2:
        seek_help = st.selectbox("Seek Help", ["No", "Yes", "Don't know"])
        anonymity = st.selectbox("Anonymity", ["No", "Yes", "Don't know"])
        leave = st.selectbox("Ease of Taking Leave", ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
        mental_health_consequence = st.selectbox("Mental Health Consequences", ["No", "Yes", "Maybe"])
        phys_health_consequence = st.selectbox("Physical Health Consequences", ["No", "Yes", "Maybe"])
        coworkers = st.selectbox("Coworkers", ["No", "Yes", "Some of them"])
        supervisor = st.selectbox("Supervisor", ["No", "Yes", "Some of them"])
        mental_health_interview = st.selectbox("Mental Health Interview", ["No", "Yes", "Maybe"])
        phys_health_interview = st.selectbox("Physical Health Interview", ["No", "Yes", "Maybe"])
        mental_vs_physical = st.selectbox("Mental vs Physical Health", ["No", "Yes", "Don't know"])
        obs_consequence = st.selectbox("Observed Consequences", ["No", "Yes"])

    
    if st.button("Predict"):
        data = {
            "Age": float(age),
            "Gender": gender,
            "self_employed": self_employed,
            "family_history": family_history,
            "work_interfere": work_interfere,
            "remote_work": remote_work,
            "tech_company": tech_company,
            "benefits": benefits,
            "care_options": care_options,
            "wellness_program": wellness_program,
            "seek_help": seek_help,
            "anonymity": anonymity,
            "leave": leave,
            "mental_health_consequence": mental_health_consequence,
            "phys_health_consequence": phys_health_consequence,
            "coworkers": coworkers,
            "supervisor": supervisor,
            "mental_health_interview": mental_health_interview,
            "phys_health_interview": phys_health_interview,
            "mental_vs_physical": mental_vs_physical,
            "obs_consequence": obs_consequence
        }

        try:
            response = requests.post(
                "http://127.0.0.1:5000/predict",
                headers={"Content-Type": "application/json"},
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                results = response.json()
                
                st.markdown("### 📊 Prediction Results")
                
                models_data = []
                for model, predictions in results.items():
                    model_name = model.replace('_model.pkl', '')
                    prediction = predictions['prediction'][0]
                    confidence = predictions['confidence'][0]
                    models_data.append({
                        'Model': model_name,
                        'Prediction': 'Treatment Needed' if prediction == 1 else 'No Treatment Needed',
                        'Confidence': f"{confidence * 100:.2f}%"
                    })
                
                df = pd.DataFrame(models_data)
                
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.table(df.style.set_properties(**{
                    'background-color': '#262730',
                    'color': 'white',
                    'border-color': '#4F4F4F'
                }))
                st.markdown('</div>', unsafe_allow_html=True)
                
                positive_predictions = sum(1 for row in models_data if row['Prediction'] == 'Treatment Needed')
                consensus_percentage = (positive_predictions / len(models_data)) * 100
                
                st.markdown(f"### 📋 Summary")
                st.markdown(f"**Consensus**: {consensus_percentage:.1f}% of models suggest treatment may be needed")
                
                st.markdown(f"### 📋 Summary")
                st.markdown(f"**Consensus**: {consensus_percentage:.1f}% of models suggest treatment may be needed")
                
                
                st.markdown("### 🔍 Model Explanations")
                lime_tabs = st.tabs([model_name.replace('_model.pkl', '') for model_name in results.keys()])
                
        
                for tab, model_name in zip(lime_tabs, results.keys()):
                    with tab:
                        model_simple_name = model_name.replace('_model.pkl', '')
                        lime_path = f"lime_explanations/{model_simple_name}_lime_explanation.html"
                        
                        try:
                            with open(lime_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=600, scrolling=True)
                        except FileNotFoundError:
                            st.warning(f"LIME explanation not found for {model_simple_name}")
                            
                        try:
                            importance_path = f"results/{model_simple_name}_feature_importance.csv"
                            if os.path.exists(importance_path):
                                st.markdown("#### Feature Importance")
                                importance_df = pd.read_csv(importance_path)
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                top_features = importance_df.nlargest(10, 'Importance')
                                sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis')
                                plt.title(f'Top 10 Most Important Features - {model_simple_name}')
                                st.pyplot(fig)
                                plt.close()
                        except Exception as e:
                            st.warning(f"Could not load feature importance for {model_simple_name}")
                
        
                
            else:
                st.error(f"Error: API returned status code {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {str(e)}")

with tab2:
    st.title("Model Metrics Dashboard")
    
    
    metric_tab1, metric_tab2, metric_tab3, metric_tab4 = st.tabs([
        "Classification Reports",
        "Confusion Matrices",
        "ROC Curves",
        "Feature Importance"
    ])

    with metric_tab1:
        st.header("Classification Reports")
        reports_dir = "classification_reports"
        for file in os.listdir(reports_dir):
            if file.endswith('.png'):
                model_name = file.replace('_classification_report.png', '')
                st.subheader(f"{model_name}")
                image = Image.open(os.path.join(reports_dir, file))
                st.image(image, use_container_width=True)

    with metric_tab2:
        st.header("Confusion Matrices")
        matrices_dir = "confusion_matrices"
        for file in os.listdir(matrices_dir):
            if file.endswith('.png'):
                model_name = file.replace('_confusion_matrix.png', '')
                st.subheader(f"{model_name}")
                image = Image.open(os.path.join(matrices_dir, file))
                st.image(image, use_container_width=True)

    with metric_tab3:
        st.header("ROC Curves")
        roc_dir = "roc_auc_plots"
        for file in os.listdir(roc_dir):
            if file.endswith('.png'):
                model_name = file.replace('_roc_auc.png', '')
                st.subheader(f"{model_name}")
                image = Image.open(os.path.join(roc_dir, file))
                st.image(image, use_container_width=True)

    with metric_tab4:
        st.header("Feature Importance")
        results_dir = "results"
        for file in os.listdir(results_dir):
            if file.endswith('feature_importance.csv'):
                model_name = file.replace('_feature_importance.csv', '')
                st.subheader(f"{model_name}")
                df = pd.read_csv(os.path.join(results_dir, file))
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=df, x='Importance', y='Feature', palette='viridis')
                plt.title(f'{model_name} Feature Importance')
                st.pyplot(fig)
                plt.close()


try:
    with open('results/model_evaluation.json', 'r') as f:
        eval_metrics = json.load(f)
    
    st.sidebar.title("Model Performance Metrics")
    
    for model, metrics in eval_metrics.items():
        st.sidebar.markdown(f"### {model}")

        metric_data = []
        for metric, value in metrics.items():
            if metric not in ['best_params', 'confusion_matrix']:
                if isinstance(value, (float, int)):
                    metric_data.append({
                        'Metric': metric,
                        'Value': f"{value:.3f}"
                    })
        metrics_df = pd.DataFrame(metric_data)
        st.sidebar.table(metrics_df.set_index('Metric'))
        
        if 'best_params' in metrics:
            st.sidebar.markdown("**Best Parameters:**")
            st.sidebar.code(json.dumps(metrics['best_params'], indent=2))
            
        if 'confusion_matrix' in metrics:
            st.sidebar.markdown("**Confusion Matrix:**")
            st.sidebar.code(str(metrics['confusion_matrix']))
        
        st.sidebar.markdown("---")
except FileNotFoundError:
    pass