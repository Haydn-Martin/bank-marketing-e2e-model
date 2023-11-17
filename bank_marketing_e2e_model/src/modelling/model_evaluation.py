import os
import json
from numpy import ndarray
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


class EvaluationPipeline:
    def __init__(self, log_reg_pipeline: Pipeline, test_data_path: str, model_info_path: str):
        self.log_reg_pipeline = log_reg_pipeline
        self.test_data_path = test_data_path
        self.model_info_path = model_info_path

    '''
    A class to evaluate a logistic regression model based on a given metric
    and save model characteristics in a given directory.

    Methods:
        make_predictions: Uses an sklearn pipeline and test features to make predictions 
        evaluate_predictions: Evaluates pipeline predictions using the roc_auc metric
        save_model_performance: Saves the model evaluation score in a given directory
    '''

    def make_predictions(self) -> ndarray:
        # Read test data
        test_data = pd.read_csv(self.test_data_path)
        # Get x test to make predictions on
        x_test = test_data.drop(columns=['target'])
        # Use pipeline to make predictions
        return self.log_reg_pipeline.predict(x_test)

    def evaluate_predictions(self) -> float:
        # Read test data
        test_data = pd.read_csv(self.test_data_path)
        # Get y test to evaluate predictions
        y_test = test_data['target']
        # Use previous method to get predictions
        preds = self.make_predictions()
        # Use predictions to evaluate model
        return roc_auc_score(y_test, preds)

    def save_model_performance(self):
        # Save model performance
        eval_score_path = os.path.join(self.model_info_path,
                                       'model_performance',
                                       'model_metrics.json')
        # Create evaluation dict
        eval_dict = {
            'roc_auc': self.evaluate_predictions()
        }
        # Save in JSON format
        with open(eval_score_path, 'w') as metrics_file:
            json.dump(eval_dict, metrics_file)
