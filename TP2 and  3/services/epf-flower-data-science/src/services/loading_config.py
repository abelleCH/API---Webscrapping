from fastapi import HTTPException, APIRouter
import os, json

CONFIG_FILE_PATH = "src/config/config.json"
MODEL_PARAMS_FILE_PATH = "src/config/model_parameters.json"

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

def load_model_parameters():
    try:
        with open(MODEL_PARAMS_FILE_PATH, "r") as file:
            return json.load(file)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while loading model parameters: {str(e)}"
        )