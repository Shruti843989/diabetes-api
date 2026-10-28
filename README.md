# Diabetes Prediction MLOps

Pima Indians Diabetes dataset (Kaggle) -> RandomForest -> FastAPI -> Docker -> GitHub Actions CI.

## Run locally (Windows PowerShell)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs

## Docker
```
docker build -t diabetes-api .
docker run -p 8000:8000 diabetes-api
```

## Sample request
POST /predict
```
{"pregnancies":6,"glucose":148,"blood_pressure":72,"skin_thickness":35,
 "insulin":0,"bmi":33.6,"diabetes_pedigree_function":0.627,"age":50}
```
