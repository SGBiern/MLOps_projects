"""
This module tests the root and the prediction end points.
Author: Sanggyu Biern
Date: 16th Mar. 2023
"""
from fastapi.testclient import TestClient

from main import app
client = TestClient(app)

def test_get_root():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {
        'Hi': 'This app predicts weather income exceeds $50k/yr based on census data.'}
    
    
def test_post_predict_up():
    r = client.post('/predict_income',  json={
        "age": 37,
        "workclass": "Private",
        "fnlgt": 280464,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 80,
        "native_country": "United-States"
    }
                   )
    
    assert r.status_code == 200
    assert r.json() == {'Income prediction': '<50K'}
    
def test_post_predict_down():
    r = client.post('/predict_income', json={
        "age": 28,
        "workclass": "Private",
        "fnlgt": 183175,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Divorced",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
                   )
    
    assert r.status_code == 200
    assert r.json() == {"Income prediction": "<=50K"}