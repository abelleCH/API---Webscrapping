import pytest
from cleaning import process_dataset

dataset_json = """
[
  {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "species": "setosa"},
  {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2, "species": "setosa"},
  {"sepal_length": 4.7, "sepal_width": 3.2, "petal_length": 1.3, "petal_width": 0.2, "species": "setosa"}
]
"""

@pytest.fixture
def processed_dataset():
    return process_dataset(dataset_json)

def test_process_dataset_shape(processed_dataset):
    assert processed_dataset.shape == (3, 5), "Dataset should have 3 rows and 5 columns."

def test_process_dataset_no_missing_values(processed_dataset):
    assert processed_dataset.isnull().sum().sum() == 0, "There should be no missing values."

def test_process_dataset_columns(processed_dataset):
    expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    assert list(processed_dataset.columns) == expected_columns, f"Expected columns: {expected_columns}"
