from fastapi import APIRouter, HTTPException
from src.schemas.message import MessageResponse
from pydantic import BaseModel
from fastapi import Query
from typing import Optional
import json
import joblib
import os
import pandas as pd
import src.services.load as load
import src.services.PST as PST
from src.services.PST import *
from src.services.firestore import *
from src.services.loading_config import *
from pydantic import BaseModel

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "src/config/abelleapi-firebase.json"

router = APIRouter()

IRIS_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
MODEL_PATH = "src/models/random_forest_model.pkl"

class Dataset(BaseModel):
    name: str
    url: str

class PredictionRequest(BaseModel):
    features: list

class ParametersRequest(BaseModel):
    params: dict


@router.get("/List", name="List All Datasets")
def list_datasets():
    """
    Lists all datasets present in the configuration file along with their names and URLs.

    Returns:
        dict: A list of datasets with their names and URLs if found, or a message indicating no datasets are found.
    """
    try:
        config = load_config()

        if not config:
            return {"message": "Aucun dataset trouvé dans le fichier de configuration."}

        datasets = [{"name": key, "url": value["url"]} for key, value in config.items()]
        
        return {"message": "Datasets listés avec succès.", "datasets": datasets}
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Le fichier de configuration est manquant."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Une erreur est survenue : {str(e)}"
        )


@router.get("/Info", name="Get dataset info", response_model=MessageResponse)
def get_dataset(dataset_name: str) -> MessageResponse:
    """
    Retrieves information about a specific dataset from the configuration file.

    Args:
        dataset_name (str): The name of the dataset to retrieve information for.
    
    Raises:
        HTTPException: If the dataset does not exist in the configuration file.
    
    Returns:
        MessageResponse: A message indicating the dataset's information or an error if not found.
    """
    try:
        config = load_config()
        
        if dataset_name not in config:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found in the configuration file."
            )
        
        dataset_info = config[dataset_name]
        return MessageResponse(
            message=f"Dataset '{dataset_name}' found: {dataset_info}"
        )
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Configuration file is missing."
        )


@router.post("/Add", name="Add dataset", response_model=MessageResponse)
def add_dataset(name: str, url: str):
    """
    Adds a new dataset to the configuration file with a unique name and URL.

    Args:
        name (str): The unique name of the dataset to be added.
        url (str): The URL where the dataset can be accessed.
    
    Raises:
        HTTPException: If the dataset already exists or an error occurs while saving.
    
    Returns:
        dict: A message indicating whether the dataset was added successfully or already exists.
    """
    try:
        config = load_config()

        if name in config:
            raise HTTPException(
                status_code=400,
                detail=f"Le dataset '{name}' existe déjà dans le fichier de configuration."
            )

        config[name] = {
            "name": name,
            "url": url
        }

        save_config(config)

        return {"message": f"Le dataset '{name}' a été ajouté avec succès."}

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Le fichier de configuration est manquant."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Une erreur est survenue : {str(e)}"
        )
    

@router.put("/Update", name="Update dataset", response_model=MessageResponse)
def update_dataset(name: str, new_url: str):
    """
    Updates the URL of an existing dataset in the configuration file.

    Args:
        name (str): The name of the dataset to update.
        new_url (str): The new URL for the dataset.
    
    Raises:
        HTTPException: If the dataset does not exist in the configuration file.
    
    Returns:
        dict: A message indicating whether the dataset was updated successfully or not found.
    """
    try:
        config = load_config()

        if name not in config:
            raise HTTPException(
                status_code=404,
                detail=f"Le dataset '{name}' n'existe pas dans le fichier de configuration."
            )

        config[name]["url"] = new_url

        save_config(config)

        return {"message": f"Le dataset '{name}' a été mis à jour avec succès."}

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Le fichier de configuration est manquant."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Une erreur est survenue : {str(e)}"
        )


@router.get("/Load", name="Load Dataset")
def load_dataset(url: Optional[str] = Query(None, description="URL of the dataset to load"),
                          dataset_name: Optional[str] = Query(None, description="Name of the dataset to load")):
    
    """
    Loads a dataset either by its URL or by its name from the configuration file.
    
    This endpoint allows you to load a dataset either by directly providing its URL or by specifying its name,
    in which case the URL will be retrieved from the configuration file. The dataset is then loaded as a CSV and returned 
    in JSON format.

    Args:
        url (str, optional): The URL where the dataset is located. If not provided, the `dataset_name` must be specified.
        dataset_name (str, optional): The name of the dataset to load. The URL will be fetched from the configuration file.

    Raises:
        HTTPException: 
            - If neither `url` nor `dataset_name` is provided.
            - If the dataset name is not found in the configuration file.
            - If there is an error loading the dataset from the URL.

    Returns:
        dict: 
            - A message indicating the successful loading of the dataset.
            - The dataset in JSON format.
    """
    try:
        config = load_config()

        if dataset_name:
            dataset_info = config.get(dataset_name)
            if dataset_info:
                url = dataset_info["url"]
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset '{dataset_name}' not found in configuration."
                )

        if not url:
            raise HTTPException(
                status_code=400,
                detail="Either 'url' or 'dataset_name' must be provided."
            )

        dataset_df = pd.read_csv(url, header=None)

        dataset_json = dataset_df.to_json(orient="records")

        return {"message": "Dataset loaded successfully.", "data": json.loads(dataset_json)}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while loading the dataset from the URL: {str(e)}"
        )


@router.post("/PST", name="Process, split and train dataset")
def process_dataset():
    """
    Processes, splits, and trains a model on the dataset. This function processes the dataset,
    splits it into training data, and trains a machine learning model.

    Raises:
        HTTPException: If an error occurs during the processing, splitting, or training of the dataset.
    
    Returns:
        str: A message confirming the completion of the processing, splitting, and training tasks.
    """
    try: 

        dataset = load.load_dataset_from_url(IRIS_DATASET_URL)
        data = PST.process_dataset(dataset)
        X_train, y_train = PST.split_dataset(data)
        PST.train_model(X_train,y_train)
        return "PST done"

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Une erreur est survenue {str(e)}"
        )    


@router.post("/Predict", name="Predict with Trained Model")
def make_prediction(request: PredictionRequest):
    """
    Makes a prediction using the trained model based on the provided feature values.

    Args:
        request (PredictionRequest): A request containing the feature values for prediction.
    
    Raises:
        HTTPException: If an error occurs during the prediction process.
    
    Returns:
        dict: A message confirming the prediction and the predicted values.
    """
    try:
        model = joblib.load(MODEL_PATH)

        input_data = pd.DataFrame([request.features]) 

        prediction = model.predict(input_data)
        print(prediction)

        return {"message": "Prediction successful", "prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )


@router.get("/SeeCollection", name="See firestore collection parameters")
def get_parameters_collection():
    """
    Retrieves collection parameters stored in Firestore.

    Returns:
        dict: A list of parameters stored in Firestore.
    """
    return get_parameters()


@router.put("/UpdateCollection", name="Update parameters in Firestore")
def update_parameters_endpoint(request: ParametersRequest):
    """
    Updates collection parameters in Firestore with the new parameters provided in the request.

    Args:
        request (ParametersRequest): A request containing the parameters to update in Firestore.
    
    Returns:
        dict: A message confirming the successful update of the parameters.
    """
    return update_parameters(request.params)


@router.post("/AddCollection", name="Add new parameters to Firestore")
def add_parameters_endpoint(request: ParametersRequest):
    """
    Adds new collection parameters to Firestore.

    Args:
        request (ParametersRequest): A request containing the parameters to add to Firestore.
    
    Returns:
        dict: A message confirming the successful addition of the parameters.
    """
    return add_parameters(request.params)
