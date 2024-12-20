from sklearn.preprocessing import StandardScaler
import pandas as pd
from io import StringIO

def process_dataset(dataset_json: str) -> pd.DataFrame:
    """
    Preprocessing function :
    1. Renaming columns
    2. Removing missing values.

    Args:
    - dataset_json (str): JSON string representing the dataset to be processed.

    Returns:
    - pd.DataFrame: Preprocessed dataset.

    Raises: 
    - ValueError: If an error occurs during processing.
    """
    try:
        dataset = pd.read_json(StringIO(dataset_json))

        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        dataset.columns = column_names

        dataset = dataset.dropna()

        return dataset
    except Exception as e:
        raise ValueError(f"Error processing the dataset: {str(e)}")