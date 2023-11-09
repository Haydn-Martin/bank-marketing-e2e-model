from ucimlrepo import fetch_ucirepo
from ucimlrepo.dotdict import dotdict
from pandas import DataFrame
import os


class FetchUCIData:
    def __init__(self, uci_repo):
        self.uci_repo = uci_repo
    '''
    A class to fetch data from UCI using the python package fetch_ucirepo.
    
    This class returns a pandas dataframe of combined features and target column.
    
    Methods:
        fetch_raw_data: Fetches raw UCI dataset given a repo id
        format_data: Returns a pandas dataframe given raw UCI dataset 
    '''

    def fetch_raw_data(self) -> dotdict:
        try:
            return fetch_ucirepo(id=self.uci_repo)
        except Exception as e:
            print(f"Error fetching raw data: {e}")

    def format_data(self) -> DataFrame:
        # Get raw data
        raw_data = self.fetch_raw_data()
        # Get features and targets
        x_vals = raw_data.data.features
        y_vals = raw_data.data.targets
        # Combine to one df
        x_vals['target'] = y_vals
        return x_vals


class SaveRawTrainingData:
    def __init__(self, raw_training_df, raw_data_store_path):
        self.raw_training_df = raw_training_df
        self.raw_data_store_path = raw_data_store_path
    '''
    A class to save a pandas dataframe in a specified directory.
    
    Methods:
        save_raw_training_data: Saves a training data pandas df in a specified raw 
                                training data directory 
    '''

    def save_raw_training_data(self):
        # File path to store data
        try:
            # File path to store data
            file_path = os.path.join(self.raw_data_store_path,
                                     'raw_training_data.csv')
            # Save pandas df as csv to this dir
            self.raw_training_df.to_csv(file_path,
                                        index=False)
        except Exception as e:
            print(f"Error saving file: {e}")
