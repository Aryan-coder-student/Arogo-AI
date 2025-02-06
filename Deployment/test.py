import requests
import json

# Define the URL and headers
url = "http://127.0.0.1:5000/predict"
headers = {
    "Content-Type": "application/json"
}

# Define the data
data = {
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

try:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response)
    if response.status_code == 200:
        print("Prediction Response:")
        print(response.json())
    else:
        print(f"Error occurred: Status code {response.status_code}")
        print("Response content:")
        print(response.text)
        
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
