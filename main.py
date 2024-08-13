# main.py
import streamlit as st
import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading

# Load the model and scalers
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('standscaler.pkl', 'rb') as f:
    sc = pickle.load(f)
with open('minmaxscaler.pkl', 'rb') as f:
    mx = pickle.load(f)

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CropRecommendationInput(BaseModel):
    N: int
    P: int
    K: int
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/predict/")
async def predict_crop(data: CropRecommendationInput):
    feature_list = [data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    
    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)
    
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        return {"crop": crop}
    else:
        return {"error": "Could not determine the best crop with the provided data."}

def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=start_fastapi).start()

# Streamlit app
st.title('Crop Recommendation System 🌱')

# Input fields
N = st.number_input('Nitrogen', min_value=0, max_value=100, step=1)
P = st.number_input('Phosphorus', min_value=0, max_value=100, step=1)
K = st.number_input('Potassium', min_value=0, max_value=100, step=1)
temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, step=0.1)
humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, step=0.1)
ph = st.number_input('pH', min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=500.0, step=0.1)

# Prediction
if st.button('Get Recommendation'):
    feature_list = [N, P, K, temperature, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    
    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)
    
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        st.success(f"The best crop to be cultivated is: {crop}")
    else:
        st.error("Sorry, we could not determine the best crop to be cultivated with the provided data.")
