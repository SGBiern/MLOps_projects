"""
This module contains utility functions for training a machine learning model.
Author: Sanggyu Biern
Date: 16th Mar. 2023
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def load_data(path):
    """
    Returns a dataframe from a given path
    """
    return pd.read_csv(path)

def load_artifact(path):
    """
    Loads an artifact
    """
    return joblib.load(path)

def process_data(X, categorical_features=[], label=None, training=True, encoder=None, lb=None):
    """
    Processes the data for the ML pipeline.
    
    Inputs:
    =======
        X : pd.DataFrame
            Dataframing containing the features and labels.
        categorical_features: list[str]
            List containing the names of the categorical features (default=[])
        label: str
            Name of the label column in `X`. If None, then an empty array will be returned
            for y (default=None)
        training: bool
            Indicator if training mode or inference/validation mode. (default=True)
        encoder : sklearn.preprocessing._encoders.OneHotEncoder
            Trained sklearn OneHotEncoder, only used if training=False (default=None)
        lb : sklearn.preprocessing._label.LabelBinarizer
            Trained sklearn LabelBinarizer, only used if training=False. (default=None)
            
    Returns:
    ========
        X : np.array
            Processed data
        y : np.array
            Processed labels if label=True, otherwise empty np.array.
        encoder : sklearn.preprocessing._encoders.OneHotEncoder
            Trained OneHotEncoder if training is True, otherwise returns the encoder passed
            in.
        lb : sklearn.preprocessing._label.LabelBinarizer
            Trained LabelBinarizer if training is True, otherwise returns the binarizer
            passed in.
    """
    
    if label is not None:
        y = X[label]
        X.drop(columns=[label], axis=1, inplace=True)
    else:
        y = np.array([])
    
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)
    
    if training:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.vales).ravel()
        except AttributeError:
            pass
   
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb

def get_cat_features():
    """ Return a list of categorical features"""
    
    cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
    ]

    return cat_features

            