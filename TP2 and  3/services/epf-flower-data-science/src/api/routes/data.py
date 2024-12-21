from fastapi import APIRouter, HTTPException
from src.schemas.message import MessageResponse
from pydantic import BaseModel
from fastapi import Query
import json
import joblib
import os
import pandas as pd
import src.services.load as load
import src.services.PST as PST
from fastapi import HTTPException, APIRouter
from src.services.PST import *
from src.services.firestore import *
from pydantic import BaseModel

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "src/config/abelleapi-firebase.json"

router = APIRouter()

MODEL_PARAMS_FILE_PATH = "src/config/model_parameters.json"
CONFIG_FILE_PATH = "src/config/config.json"
IRIS_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
MODEL_PATH = "src/models/random_forest_model.pkl"

class Dataset(BaseModel):
    name: str
    url: str

def load_config():
    """
    Charge le fichier JSON de configuration.
    """
    if not os.path.exists(CONFIG_FILE_PATH):
        raise FileNotFoundError("The configuration file does not exist.")
    
    with open(CONFIG_FILE_PATH, "r") as file:
        return json.load(file)
    
def save_config(config):
    """
    Sauvegarde le fichier JSON de configuration.
    """
    with open(CONFIG_FILE_PATH, "w") as file:
        json.dump(config, file, indent=4)    

# Fonction pour charger les paramètres du modèle depuis le fichier JSON
def load_model_parameters():
    try:
        with open(MODEL_PARAMS_FILE_PATH, "r") as file:
            return json.load(file)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while loading model parameters: {str(e)}"
        )

@router.get("/list", name="List All Datasets")
def list_datasets():
    """
    Liste tous les datasets présents dans le fichier de configuration avec leurs noms et URLs.
    """
    try:
        # Charger le fichier de configuration
        config = load_config()

        # Vérifier si le fichier est vide
        if not config:
            return {"message": "Aucun dataset trouvé dans le fichier de configuration."}

        # Extraire les datasets sous forme de liste
        datasets = [{"name": key, "url": value["url"]} for key, value in config.items()]
        
        # Retourner les datasets
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


@router.get("/info", name="Get dataset info", response_model=MessageResponse)
def get_dataset(dataset_name: str) -> MessageResponse:
    """
    Récupère les informations d'un dataset à partir du fichier de configuration.
    - `dataset_name`: Le nom du dataset à rechercher dans le fichier de configuration.
    """
    try:
        # Charger la configuration
        config = load_config()
        
        # Vérifier si le dataset existe dans la configuration
        if dataset_name not in config:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found in the configuration file."
            )
        
        # Récupérer les informations du dataset
        dataset_info = config[dataset_name]
        return MessageResponse(
            message=f"Dataset '{dataset_name}' found: {dataset_info}"
        )
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Configuration file is missing."
        )

@router.post("/add", name="Add dataset", response_model=MessageResponse)
def add_dataset(name: str, url: str):
    """
    Ajoute un dataset au fichier de configuration en utilisant un nom et une URL spécifiés dans l'URL.
    - `name`: Le nom unique du dataset.
    - `url`: L'URL du dataset.
    """
    try:
        # Charger la configuration existante
        config = load_config()

        # Vérifier si le dataset existe déjà
        if name in config:
            raise HTTPException(
                status_code=400,
                detail=f"Le dataset '{name}' existe déjà dans le fichier de configuration."
            )

        # Ajouter le nouveau dataset
        config[name] = {
            "name": name,
            "url": url
        }

        # Sauvegarder les modifications dans le fichier
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
    
@router.put("/update", name="Update dataset", response_model=MessageResponse)
def update_dataset(name: str, new_url: str):
    """
    Modifie l'URL d'un dataset existant dans le fichier de configuration.
    - `name`: Le nom du dataset à modifier.
    - `new_url`: La nouvelle URL du dataset.
    """
    try:
        # Charger la configuration existante
        config = load_config()

        # Vérifier si le dataset existe
        if name not in config:
            raise HTTPException(
                status_code=404,
                detail=f"Le dataset '{name}' n'existe pas dans le fichier de configuration."
            )

        # Modifier les informations du dataset
        config[name]["url"] = new_url

        # Sauvegarder les modifications dans le fichier
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

@router.get("/load", name="Load Dataset")
def load_dataset_from_url(url: str = Query(..., description="URL of the dataset to load")):
    """
    Charge un dataset à partir de l'URL fournie et le retourne sous forme de JSON.
    """
    try:
        # Lire le fichier CSV directement depuis l'URL fournie
        dataset_df = pd.read_csv(url, header=None)

        # Convertir le DataFrame en JSON
        dataset_json = dataset_df.to_json(orient="records")

        # Retourner le JSON du dataset
        return {"message": "Dataset loaded successfully.", "data": json.loads(dataset_json)}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Une erreur est survenue lors du chargement du dataset depuis l'URL : {str(e)}"
        )

@router.post("/PST", name="Process, split and train dataset")
def process_dataset():
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


class PredictionRequest(BaseModel):
    features: list  # List of feature values (depending on your dataset's features)

# Endpoint to make predictions
@router.post("/predict", name="Predict with Trained Model")
def make_prediction(request: PredictionRequest):
    """
    Makes a prediction using the trained model based on the provided features.
    - `features`: List of feature values for prediction.
    """
    try:
        model = joblib.load(MODEL_PATH)

        input_data = pd.DataFrame([request.features])  # Convert the list of features into a DataFrame

        # Make prediction using the model
        prediction = model.predict(input_data)
        print(prediction)

        # Return the prediction as JSON
        return {"message": "Prediction successful", "prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )


class ParametersRequest(BaseModel):
    params: dict

@router.get("/SeeCollection", name="See firestore collection parameters")
def get_parameters_collection():
    """
    Récupère les paramètres stockés dans Firestore.
    """
    return get_parameters()

@router.put("/UpdateCollection", name="Update parameters in Firestore")
def update_parameters_endpoint(request: ParametersRequest):
    """
    Met à jour les paramètres dans Firestore avec les paramètres envoyés dans la requête.
    """
    return update_parameters(request.params)

@router.post("/AddCollection", name="Add new parameters to Firestore")
def add_parameters_endpoint(request: ParametersRequest):
    """
    Ajoute de nouveaux paramètres dans Firestore.
    """
    return add_parameters(request.params)
