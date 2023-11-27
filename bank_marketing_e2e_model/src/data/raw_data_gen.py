"""
This module contains classes to generate the raw
data required to train the model.
"""


from pandas import DataFrame

from ucimlrepo import fetch_ucirepo
from ucimlrepo.dotdict import dotdict


class FetchUCIData:
    """
    A class to fetch data from UCI using the python package fetch_ucirepo.

    This class returns a pandas dataframe of combined features and target column.

    Methods:
        fetch_raw_data: Fetches raw UCI dataset given a repo id
        format_data: Returns a pandas dataframe given raw UCI dataset
    """

    def __init__(self, uci_repo: int):
        self.uci_repo = uci_repo

    def fetch_raw_data(self) -> dotdict:
        """Fetches raw UCI dataset given a repo id"""
        try:
            return fetch_ucirepo(id=self.uci_repo)
        except Exception as e:
            print(f"Error fetching raw data: {e}")
            # Return an empty dotdict
            return dotdict()

    def format_data(self) -> DataFrame:
        """Returns a pandas dataframe given raw UCI dataset"""
        # Get raw data
        raw_data = self.fetch_raw_data()
        # Get features and targets
        x_vals = raw_data.data.features
        y_vals = raw_data.data.targets
        # Combine to one df
        x_vals['target'] = y_vals
        return x_vals


class SaveRawTrainingData:
    """
    A class to save a pandas dataframe in a specified directory.

    Methods:
        save_raw_training_data: Saves a training data pandas df in a specified raw
                                training data directory
    """
    def __init__(self, raw_training_df: DataFrame, raw_data_store_path: str):
        self.raw_training_df = raw_training_df
        self.raw_data_store_path = raw_data_store_path

    def save_raw_training_data(self):
        """
        Saves a training data pandas df in a specified raw
        training data directory
        """
        # Save pandas df as csv to this dir
        self.raw_training_df.to_csv(self.raw_data_store_path,
                                    index=False)
