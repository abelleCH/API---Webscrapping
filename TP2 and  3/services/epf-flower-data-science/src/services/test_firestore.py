import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from google.cloud import firestore
from firestore import get_parameters, update_parameters, add_parameters

# Mocking the Firestore document reference and its methods

@pytest.fixture
def mock_firestore_client():
    with patch.object(firestore, 'Client') as mock:
        yield mock

def test_get_parameters(mock_firestore_client):
    mock_doc_ref = MagicMock()
    mock_firestore_client.return_value.collection.return_value.document.return_value = mock_doc_ref
    mock_doc_ref.get.return_value.exists = True
    mock_doc_ref.get.return_value.to_dict.return_value = {"param1": "value1", "param2": "value2"}

    params = get_parameters()

    assert params == {"param1": "value1", "param2": "value2"}
    mock_firestore_client.return_value.collection.return_value.document.return_value.get.assert_called_once()

def test_get_parameters_not_found(mock_firestore_client):
    mock_doc_ref = MagicMock()
    mock_firestore_client.return_value.collection.return_value.document.return_value = mock_doc_ref
    mock_doc_ref.get.return_value.exists = False

    with pytest.raises(HTTPException):
        get_parameters()

def test_update_parameters(mock_firestore_client):
    mock_doc_ref = MagicMock()
    mock_firestore_client.return_value.collection.return_value.document.return_value = mock_doc_ref
    mock_doc_ref.get.return_value.exists = True
    mock_doc_ref.get.return_value.to_dict.return_value = {"params": {"param1": "value1"}}

    new_params = {"param2": "value2"}
    response = update_parameters(new_params)

    assert response["message"] == "Parameters updated successfully."
    assert response["params"] == {"param1": "value1", "param2": "value2"}
    mock_firestore_client.return_value.collection.return_value.document.return_value.set.assert_called_once()

def test_add_parameters(mock_firestore_client):
    mock_doc_ref = MagicMock()
    mock_firestore_client.return_value.collection.return_value.document.return_value = mock_doc_ref
    mock_doc_ref.get.return_value.exists = True
    mock_doc_ref.get.return_value.to_dict.return_value = {"param1": "value1"}

    new_params = {"param2": "value2", "param1": "value1"}
    response = add_parameters(new_params)

    assert response["message"] == "Parameters processed."
    assert response["response"]["param2"] == "Parameter 'param2' added successfully."
    assert response["response"]["param1"] == "Parameter 'param1' already exists. Please update it."
    mock_firestore_client.return_value.collection.return_value.document.return_value.set.assert_called_once()

def test_add_parameters_empty(mock_firestore_client):
    mock_doc_ref = MagicMock()
    mock_firestore_client.return_value.collection.return_value.document.return_value = mock_doc_ref
    mock_doc_ref.get.return_value.exists = False

    new_params = {"param1": "value1"}
    response = add_parameters(new_params)

    assert response["message"] == "Parameters processed."
    assert response["response"]["param1"] == "Parameter 'param1' added successfully."
    mock_firestore_client.return_value.collection.return_value.document.return_value.set.assert_called_once()
