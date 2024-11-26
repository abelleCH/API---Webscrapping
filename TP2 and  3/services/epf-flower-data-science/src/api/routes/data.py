from fastapi import APIRouter, HTTPException
from src.schemas.message import MessageResponse
from pydantic import BaseModel
import json
import os

router = APIRouter()

class Dataset(BaseModel):
    name: str
    url: str

# Chemin vers le fichier de configuration
CONFIG_FILE_PATH = "src/config/config.json"

# Charger le fichier de configuration
def load_config():
    """
    Charge le fichier JSON de configuration.
    """
    if not os.path.exists(CONFIG_FILE_PATH):
        raise FileNotFoundError("The configuration file does not exist.")
    
    with open(CONFIG_FILE_PATH, "r") as file:
        return json.load(file)
    
# Sauvegarder le fichier de configuration
def save_config(config):
    """
    Sauvegarde le fichier JSON de configuration.
    """
    with open(CONFIG_FILE_PATH, "w") as file:
        json.dump(config, file, indent=4)    


@router.get("/dataset", name="Get dataset info", response_model=MessageResponse)
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
async def add_dataset(name: str, url: str):
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