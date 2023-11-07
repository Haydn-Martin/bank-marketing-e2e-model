from ucimlrepo import fetch_ucirepo
from ucimlrepo.dotdict import dotdict
from pandas import DataFrame


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

    def fetch_raw_data(self) -> dotdict.dotdict:
        return fetch_ucirepo(id=self.uci_repo)

    def format_data(self) -> DataFrame:
        # Get raw data
        raw_data = self.fetch_raw_data()
        # Get features and targets
        x_vals = raw_data.data.features
        y_vals = raw_data.data.targets
        # Combine to one df
        x_vals['target'] = y_vals
        return x_vals
