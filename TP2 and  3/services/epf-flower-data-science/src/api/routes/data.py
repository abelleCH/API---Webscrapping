from fastapi import APIRouter, HTTPException
from src.schemas.message import MessageResponse
from pydantic import BaseModel
import json
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

IRIS_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Step 7: Endpoint pour charger le dataset Iris
@router.get("/load", name="Load Iris Dataset")
def load_iris_dataset():
    """
    Charge le dataset Iris à partir de l'URL et le retourne sous forme de JSON.
    """
    try:
        # Télécharger le dataset Iris (par exemple depuis UCI ou un fichier local)
        # Définir les colonnes en fonction du format Iris
        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        
        # Lire le fichier CSV directement depuis l'URL ou depuis un fichier local
        iris_df = pd.read_csv(IRIS_DATASET_URL, header=None, names=column_names)

        # Convertir le DataFrame en JSON
        iris_json = iris_df.to_json(orient="records")

        # Retourner le JSON du dataset Iris
        return {"message": "Iris dataset loaded successfully.", "data": json.loads(iris_json)}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Une erreur est survenue lors du chargement du dataset Iris : {str(e)}"
        )


@router.post("/process", name="Process dataset")
def process_dataset():
    """
    Perform necessary preprocessing on the dataset before it can be used for model training.
    This may include handling missing values, encoding categorical variables, normalizing data, etc.
    """
    try:
        # Load the Iris dataset
        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        iris_df = pd.read_csv(IRIS_DATASET_URL, header=None, names=column_names)

        # 1. Handle missing values (if any, drop them in this case)
        iris_df = iris_df.dropna()

        # 2. Encode categorical variables (the 'species' column)
        iris_df['species'] = iris_df['species'].astype('category').cat.codes

        # 3. Normalize the numeric columns
        scaler = StandardScaler()
        numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        iris_df[numeric_columns] = scaler.fit_transform(iris_df[numeric_columns])

        # Return the processed DataFrame as a JSON object for verification
        iris_json = iris_df.to_json(orient="records")
        return {"message": "Dataset processed successfully.", "data": json.loads(iris_json)}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the dataset: {str(e)}"
        )


# Step 9: Endpoint to split the dataset into training and testing sets
@router.post("/split", name="Split dataset into train and test")
def split_dataset():
    """
    Split the Iris dataset into training and testing sets and return both as JSON.
    """
    try:
        # Load the Iris dataset
        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        iris_df = pd.read_csv(IRIS_DATASET_URL, header=None, names=column_names)

        # Process the dataset: handling missing values, encoding categorical data, and normalizing
        iris_df = iris_df.dropna()
        iris_df['species'] = iris_df['species'].astype('category').cat.codes
        scaler = StandardScaler()
        numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        iris_df[numeric_columns] = scaler.fit_transform(iris_df[numeric_columns])

        # Split the dataset into features (X) and target (y)
        X = iris_df[numeric_columns]
        y = iris_df['species']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert the split datasets to JSON format
        X_train_json = X_train.to_json(orient="split")
        X_test_json = X_test.to_json(orient="split")
        y_train_json = y_train.to_json(orient="split")
        y_test_json = y_test.to_json(orient="split")

        # Return the split datasets as JSON
        return {
            "message": "Dataset split into train and test sets successfully.",
            "train": {
                "X": json.loads(X_train_json),
                "y": json.loads(y_train_json)
            },
            "test": {
                "X": json.loads(X_test_json),
                "y": json.loads(y_test_json)
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while splitting the dataset: {str(e)}"
        )
