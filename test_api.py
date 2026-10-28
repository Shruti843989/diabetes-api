from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

SAMPLE = {
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50,
}


def test_home():
    r = client.get("/")
    assert r.status_code == 200


def test_health():
    r = client.get("/health")
    assert r.json() == {"status": "ok"}


def test_predict():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability_diabetic"] <= 1.0


def test_predict_invalid_input():
    r = client.post("/predict", json={"glucose": "abc"})
    assert r.status_code == 422
