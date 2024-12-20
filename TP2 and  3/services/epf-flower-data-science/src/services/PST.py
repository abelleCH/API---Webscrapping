from sklearn.model_selection import train_test_split
import joblib, json, os
import pandas as pd
from io import StringIO
from sklearn.ensemble import RandomForestClassifier

MODEL_SAVE_PATH = "src/models/random_forest_model.pkl"
PARAMETERS_FILE_PATH = "src/config/model_parameters.json"

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

def split_dataset(iris_df):
    """
    Splits the Iris dataset into training and test sets and returns them.
    """
    numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = iris_df[numeric_columns]
    y = iris_df['species'].astype('category')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train

def load_model_parameters(file_path: str):
    """
    Loads the model parameters from a JSON file.
    
    Args:
        file_path (str): The path to the JSON file containing the model parameters.
    
    Returns:
        dict: Dictionary with the model parameters.
    """
    try:
        with open(file_path, 'r') as f:
            params = json.load(f)
        
        if isinstance(params.get("n_estimators", None), int) and \
           isinstance(params.get("max_depth", None), (int, type(None))) and \
           isinstance(params.get("max_features", None), (str, int, float)):
            return params
        else:
            return {}
    except Exception as e:
        return {}

def train_model(X_train, y_train):
    """
    Trains a RandomForest model using parameters defined in the JSON file.
    
    Args:
        X_train (DataFrame): Training data (features).
        y_train (Series): Training labels (target).
    """
    model_params = load_model_parameters(PARAMETERS_FILE_PATH)

    if not model_params:
        return

    try:
        model = RandomForestClassifier(**model_params)

        if len(X_train) != len(y_train):
            return
        
        model.fit(X_train, y_train)

        if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH))

        joblib.dump(model, MODEL_SAVE_PATH)
    except Exception as e:
        pass
