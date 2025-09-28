from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import io
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf
import requests, statistics
from datetime import datetime, timedelta
import os

# --------- App Setup ---------
app = FastAPI(title="Crop & Plant Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (change for production)
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Load Models ---------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_crop.pkl")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenetv2_plantvillage.h5")

crop_model = joblib.load(RF_MODEL_PATH)
disease_model = tf.keras.models.load_model(CNN_MODEL_PATH)

CLASS_NAMES = [
    "Pepper_bell__Bacterial_spot",
    "Pepper_bell__healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_1",
    "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow",
    "Tomato_Tomato_mosaic",
    "Tomato_healthy"
]

# --------- SoilWeatherAnalyzer ---------
class SoilWeatherAnalyzer:
    def __init__(self):
        self.data_sources = []

    def get_soilgrids_data(self, lat, lon):
        url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        params = {
            'lon': lon, 'lat': lat,
            'property': ['phh2o', 'soc', 'clay'],
            'depth': ['0-5cm'], 'value': ['mean']
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            result = {}
            for layer in data.get('properties', {}).get('layers', []):
                if layer['name'] in ['phh2o', 'soc', 'clay']:
                    val = layer['depths'][0]['values']['mean']
                    if val == -32768:
                        continue
                    result[layer['name']] = round(val, 2)

            if 'soc' in result:
                soc = result['soc']
                soc_kg_ha = soc * 1.4 * 5 * 10
                result['nitrogen'] = round(soc_kg_ha / 11, 2)

            if 'clay' in result:
                clay = result['clay']
                result['phosphorus'] = round(15 + clay * 0.2, 2)
                result['potassium'] = round(120 + clay * 1.5, 2)

            self.data_sources.append("SoilGrids API")
            return result
        except Exception as e:
            print(f"SoilGrids error: {e}")
            return None

    def get_weather_data(self, lat, lon):
        end_date = datetime.today()
        start_date = end_date - timedelta(days=9)
        url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?"
               f"parameters=T2M,RH2M,PRECTOTCORR&community=AG"
               f"&longitude={lon}&latitude={lat}"
               f"&start={start_date.strftime('%Y%m%d')}"
               f"&end={end_date.strftime('%Y%m%d')}&format=JSON")
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            params = r.json()['properties']['parameter']
            temps = [v for v in params['T2M'].values() if v != -999.0]
            hums = [v for v in params['RH2M'].values() if v != -999.0]
            rains = [v for v in params['PRECTOTCORR'].values() if v != -999.0]
            if not (temps and hums and rains):
                return {'temperature': None, 'humidity': None, 'rainfall': None}
            self.data_sources.append("NASA POWER API")
            return {
                'temperature': round(statistics.mean(temps), 2),
                'humidity': round(statistics.mean(hums), 2),
                'rainfall': round(sum(rains), 2)
            }
        except Exception as e:
            print(f"NASA API error: {e}")
            return {'temperature': None, 'humidity': None, 'rainfall': None}

    def get_regional_average(self, lat, lon):
        regions = [
            {'name': 'North India', 'lat_range': (24, 36), 'lon_range': (74, 85),
             'ph': 7.2, 'nitrogen': 22.5, 'phosphorus': 16.8, 'potassium': 145.0},
            {'name': 'South India', 'lat_range': (8, 20), 'lon_range': (74, 85),
             'ph': 6.8, 'nitrogen': 24.3, 'phosphorus': 18.2, 'potassium': 155.0},
        ]
        for region in regions:
            if (region['lat_range'][0] <= lat <= region['lat_range'][1] and
                    region['lon_range'][0] <= lon <= region['lon_range'][1]):
                self.data_sources.append(region['name'])
                return region
        return None

    def integrate_soil(self, soil, reg):
        if soil is None and reg is None:
            self.data_sources.append("India Defaults")
            return {'ph': 7.0, 'N': 22, 'P': 17, 'K': 150}
        return {
            'ph': soil.get('phh2o') if soil and 'phh2o' in soil else (reg['ph'] if reg else 7.0),
            'N': soil.get('nitrogen') if soil and 'nitrogen' in soil else (reg['nitrogen'] if reg else 22),
            'P': soil.get('phosphorus') if soil and 'phosphorus' in soil else (reg['phosphorus'] if reg else 17),
            'K': soil.get('potassium') if soil and 'potassium' in soil else (reg['potassium'] if reg else 150),
        }

    def get_features(self, lat, lon):
        self.data_sources = []
        soil = self.get_soilgrids_data(lat, lon)
        reg = self.get_regional_average(lat, lon)
        soil_data = self.integrate_soil(soil, reg)
        weather = self.get_weather_data(lat, lon)
        return {
            'N': soil_data['N'], 'P': soil_data['P'], 'K': soil_data['K'],
            'ph': soil_data['ph'],
            'temperature': weather['temperature'],
            'humidity': weather['humidity'],
            'rainfall': weather['rainfall'],
            'data_sources': self.data_sources
        }

# --------- Schemas ---------
class RecommendInput(BaseModel):
    N: float
    P: float
    K: float
    ph: float
    humidity: float
    temp: float
    rainfall: float

# --------- Endpoints ---------
@app.get("/")
def root():
    return {"status": "API running"}

@app.get("/api/features")
def get_features(lat: float = Query(...), lon: float = Query(...)):
    analyzer = SoilWeatherAnalyzer()
    return analyzer.get_features(lat, lon)

@app.post("/api/recommend")
def recommend(data: RecommendInput):
    features = np.array([[data.N, data.P, data.K,
                          data.temp, data.humidity,
                          data.ph, data.rainfall]])
    pred = crop_model.predict(features)[0]
    probs = getattr(crop_model, "predict_proba", None)
    conf = float(np.max(probs(features)[0])) if probs else 0.9
    return {"recommended_crop": str(pred), "confidence": conf}

@app.post("/api/detect")
async def detect(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = disease_model.predict(arr)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    disease_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}"
    return {"disease": disease_name, "confidence": conf}

# --------- Run ---------
if __name__ == "__main__":
    # ✅ host=0.0.0.0 lets your phone connect via LAN
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
