from pathlib import Path
from pandas import DataFrame

# Saves a pandas df as a csv in a specified path
def save_df_to_path(df: DataFrame, path: str):
    Path(path).parent.mkdir(parents=True,
                            exist_ok=True)  # create file path
    df.to_csv(path,
              index=False)  # save to file path