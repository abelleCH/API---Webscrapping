import unittest
from unittest.mock import MagicMock
import pandas as pd
import json
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib, os
from PST import process_dataset, split_dataset, load_model_parameters, train_model

class TestModelPipeline(unittest.TestCase):

    def setUp(self):
        self.dataset_json = json.dumps([
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "species": "setosa"},
            {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2, "species": "setosa"},
            {"sepal_length": 4.7, "sepal_width": 3.2, "petal_length": 1.3, "petal_width": 0.2, "species": "setosa"}
        ])

        self.model_parameters = {
            "n_estimators": 100,
            "max_depth": 5,
            "max_features": "sqrt"
        }

        self.expected_df = pd.DataFrame({
            "sepal_length": [5.1, 4.9],
            "sepal_width": [3.5, 3.0],
            "petal_length": [1.4, 1.4],
            "petal_width": [0.2, 0.2],
            "species": ["setosa", "setosa"]
        })

    def processed_dataset(self):
        return process_dataset(self.dataset_json)

    def test_process_dataset_shape(self):
        processed_data = self.processed_dataset()
        self.assertEqual(processed_data.shape, (3, 5), "Dataset should have 3 rows and 5 columns.")

    def test_process_dataset_no_missing_values(self):
        processed_data = self.processed_dataset()
        self.assertEqual(processed_data.isnull().sum().sum(), 0, "There should be no missing values.")

    def test_process_dataset_columns(self):
        processed_data = self.processed_dataset()
        expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        self.assertListEqual(list(processed_data.columns), expected_columns, f"Expected columns: {expected_columns}")

    def test_split_dataset(self):
        iris_df = self.expected_df.copy()
        X_train, y_train = split_dataset(iris_df)

        self.assertListEqual(list(X_train.columns), ["sepal_length", "sepal_width", "petal_length", "petal_width"])
        self.assertEqual(len(X_train), len(y_train))

    def test_load_model_parameters(self):
        file_path = "test_parameters.json"
        with open(file_path, "w") as file:
            json.dump(self.model_parameters, file, indent=4)

        params=load_model_parameters(file_path)

        self.assertEqual(params, self.model_parameters)
    
    # def test_train_model(self):
    #     X_train = self.expected_df.drop(columns="species")
    #     y_train = self.expected_df["species"]

    #     model_filename = "mock_model.pkl"

    #     model = train_model(X_train, y_train)

    #     self.assertTrue(os.path.exists(model_filename), "Model file should be saved.")

if __name__ == "__main__":
    unittest.main()
