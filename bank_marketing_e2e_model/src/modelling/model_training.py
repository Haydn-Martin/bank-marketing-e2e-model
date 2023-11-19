from ..utils.utils import save_df_to_path

import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TrainingPipeline:
    def __init__(self, hyper_params: dict, raw_data_path: str, train_data_path: str,
                 test_data_path: str, model_path: str):
        self.hyper_params = hyper_params
        self.raw_data_path = raw_data_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.model_path = model_path

    '''
    A class to fetch training data and use a specified model configuration
    to train a logistic regression to predict if a client will subscribe to a term deposit.

    Methods:
        data_split: Fetches raw training data and saves to training and test data 
        train_model: Trains a logistic regression model on training 
        save_model: Saves the model pipeline in binary format in a specified directory
    '''

    def data_split(self, test_size: float = 0.3, random_state: int = 420):
        # Read raw training data
        raw_training_data = pd.read_csv(self.raw_data_path)
        # Get features
        x_values = raw_training_data.drop(columns=['target'])
        # Get binary target
        y_values = raw_training_data['target'].map(dict(yes=1, no=0))
        # Split data to train, test sets
        x_train, x_test, y_train, y_test = train_test_split(x_values,
                                                            y_values,
                                                            test_size=test_size,
                                                            random_state=random_state)
        # Save training data in appropriate file path
        x_train['target'] = y_train  # add target
        save_df_to_path(x_train, self.train_data_path)
        # Save test data in appropriate file path
        x_test['target'] = y_test  # add target
        save_df_to_path(x_test, self.test_data_path)

    def train_model(self) -> Pipeline:
        # Read training data
        all_training = pd.read_csv(self.train_data_path)
        # Split to x and y
        x_training = all_training.drop(columns=['target'])
        y_training = all_training['target']
        # Define categorical and numerical columns
        cat_cols = [col for col in x_training.columns if x_training[col].dtype == 'object']
        num_cols = [col for col in x_training.columns if x_training[col].dtype != 'object']
        # Define OHE mini-pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        # Define Scalar mini-pipeline
        non_categorical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler(with_mean=False))
        ])
        # Combine preprocessing steps using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, cat_cols),
                ('num', non_categorical_transformer, num_cols)
            ])
        # Create a full pipeline including preprocessing and modeling
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', LogisticRegression(**self.hyper_params))])
        # Fit to training data
        pipeline.fit(x_training, y_training)
        # Return pipeline
        return pipeline

    def save_model(self):
        # Save model pipeline for making inferences
        joblib.dump(self.train_model(),
                    self.model_path)
