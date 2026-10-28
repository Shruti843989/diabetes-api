from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Diabetes Prediction API")
model = joblib.load("model.joblib")


class PatientData(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float


@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PatientData):
    features = [[
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree_function,
        data.age,
    ]]
    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])
    return {
        "prediction": pred,
        "result": "Diabetic" if pred == 1 else "Not Diabetic",
        "probability_diabetic": round(prob, 4),
    }
