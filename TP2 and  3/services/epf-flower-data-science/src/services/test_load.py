import pytest
from unittest.mock import MagicMock
import pandas as pd
from load import load_dataset_from_url  

def test_load_dataset_from_url_success():
    """
    Test the successful loading of a dataset from a URL.
    """
    url = "http://example.com/test_dataset.csv"

    mock_df = MagicMock(spec=pd.DataFrame)
    mock_df.to_json.return_value = '[{"0":"data1","1":1},{"0":"data2","1":2}]'
    
    pd.read_csv = MagicMock(return_value=mock_df)
    
    result = load_dataset_from_url(url)

    expected_result = '[{"0":"data1","1":1},{"0":"data2","1":2}]'

    assert result == expected_result


def test_load_dataset_from_url_failure():
    """
    Test the failure case when the dataset cannot be loaded due to invalid URL or data issues.
    """
    url = "http://example.com/invalid_dataset.csv"

    pd.read_csv = MagicMock(side_effect=Exception("Failed to load dataset"))

    with pytest.raises(ValueError, match="Error loading dataset from URL"):
        load_dataset_from_url(url)
