"""
This module includes unit tests for the ML model
Author: Sanggyu Biern
Date: 16th Mar. 2023
"""

from pathlib import Path
import logging
import pandas as pd
import pytest
import src
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

DATA_PATH = '../data/census.csv'
MODEL_PATH = '../model/model.pkl'

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(name='data')
def data():
    """
    Fixture will be utilized by the unit tests
    """
    yield pd.read_csv(DATA_PATH)
    
def test_load_data(data):
    """ Check the data recieved. """
    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] > 0
    assert data.shape[1] > 0
    
def test_model():
    """ Check model type """
    model = src.utils.load_artifact(MODEL_PATH)
    assert isinstance(mode, RandomForestClassifier)
    
def test_process_data(data):
    """ Test the data split """
    train, _ = train_test_split(data, test_size=0.2)
    X, y, _, _ = src.utils.process_data(train, cat_features, label='salary')
    assert len(X) == len(y)