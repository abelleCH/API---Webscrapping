import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import MagicMock
import json

from PST import split_dataset, load_model_parameters, train_model  # Remplacez 'your_module' par le nom réel du module

# Sample Iris dataset for testing
data = {
    'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0],
    'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6],
    'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4],
    'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2],
    'species': ['setosa', 'setosa', 'setosa', 'setosa', 'setosa']
}

iris_df = pd.DataFrame(data)

@pytest.fixture
def mock_model_params():
    """Fixture to mock model parameters."""
    return {
        "n_estimators": 100,
        "max_depth": 3,
        "max_features": "auto"
    }

@pytest.fixture
def mock_model():
    """Fixture to mock a RandomForest model."""
    return MagicMock(spec=RandomForestClassifier)

def test_split_dataset():
    """Test the split_dataset function."""
    X_train, y_train = split_dataset(iris_df)
    assert X_train.shape[0] == 4, "Training set should have 80% of the data."
    assert y_train.shape[0] == 4, "Training set labels should match the training data size."
    assert len(X_train) == len(y_train), "Features and labels should have the same length."

def test_load_model_parameters_success(mock_model_params):
    """Test loading model parameters successfully."""
    mock_open = MagicMock()
    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_model_params)
    
    with mock_open("dummy_path", 'r') as f:
        params = load_model_parameters("dummy_path")
        
    assert params == mock_model_params, "The model parameters should match the mocked data."

def test_load_model_parameters_failure():
    """Test failure when loading model parameters."""
    mock_open = MagicMock()
    mock_open.return_value.__enter__.return_value.read.return_value = "invalid json"
    
    with mock_open("dummy_path", 'r') as f:
        params = load_model_parameters("dummy_path")
        
    assert params == {}, "If the JSON is invalid, it should return an empty dictionary."

def test_train_model_success(mock_model, mock_model_params):
    """Test the train_model function."""
    model = mock_model
    model_params = mock_model_params
    
    # Mock the load_model_parameters to return the mocked parameters
    load_model_parameters = MagicMock(return_value=model_params)
    
    X_train, y_train = split_dataset(iris_df)
    
    # Call the train_model function
    train_model(X_train, y_train)
    
    
