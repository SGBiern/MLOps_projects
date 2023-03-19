"""
Testing module that will check the churn_library.py procedure.
Artifact produced will be in logs folders.
Author: Sanggyu Biern
Date: 15th Feb. 2023
"""

import os
import logging

from churn_library import import_data, perform_eda, encoder_helper, \
    perform_feature_engineering, train_models

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_eda(perform_eda):
    '''
    test perform eda function - test creation of images related eda
    '''

    try:
        assert os.path.isfile('images/eda/Churn.jpg')
        assert os.path.isfile('images/eda/Customer_Age.jpg')
        assert os.path.isfile('images/eda/Marital_Status.jpg')
        assert os.path.isfile('images/eda/Total_Trans_Ct.jpg')
        assert os.path.isfile('images/eda/Heatmap.jpg')
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

def test_encoder_helper(encoder_helper):
    '''
    test encoder helper - Testing the categorical columns were removed,
                        and the response parameter is working properly
    '''
    try:
        encoder_helper(df, cat_vars, response="Churn")
        logging.info("Testing test_encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: It appears there is something wrong in the function")
        raise err

    try:
        df_new = encoder_helper(df, cat_vars, response="Churn")
        for i in df_new.columns[df_new.columns.str.contains('_Churn')]:
            assert df_new[i].dtype != "O"
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The encoder not add new columns")
        raise err

    try:
        df_new = encoder_helper(df, cat_vars, response="Churn")
        assert sum(df_new.columns.str.contains('_Churn')) == len(cat_vars)
    except AssertionError as err:
        logging.error("Testing encoder_helper: The response not work")
        raise err



def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering - Testing the split of train-test dataset
    '''
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df_encoder, response="Churn")
        assert x_train.shape[0] > x_test.shape[0]
        assert y_train.shape[0] > y_test.shape[0]
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The function not split the dataframe")
        raise err


def test_train_models(train_models):
    '''
    test train_models - Testing
    '''
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df_encoder, response="Churn")
        train_models(x_train, x_test, y_train, y_test)
        assert os.path.isfile('./models/rfc_model.pkl')
        assert os.path.isfile('./models/logistic_model.pkl')
        assert os.path.isfile('images/results/Random_Forest.jpg')
        assert os.path.isfile('images/results/Logistic_Regression.jpg')
        assert os.path.isfile('./images/results/Feature_Importance.jpg')
        assert os.path.isfile('./images/results/Roc_Curves.jpg')
        logging.info('Testing testing_model: SUCCESS')
    except FileNotFoundError as err:
        logging.error('Teesting train_models: The files were not found')
        raise err


if __name__ == "__main__":
    df = import_data("./data/bank_data.csv")
    cat_vars = [var for var in df.columns if df[var].dtype == "object"]
    test_import(import_data)
    test_eda(perform_eda(df))
    test_encoder_helper(encoder_helper)
    df_encoder = encoder_helper(df, cat_vars, response="Churn")
    test_perform_feature_engineering(perform_feature_engineering)
    test_train_models(train_models)
    