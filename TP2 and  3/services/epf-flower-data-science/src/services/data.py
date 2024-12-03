from sklearn.model_selection import train_test_split
import joblib
import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Chemins vers les fichiers
MODEL_SAVE_PATH = "src/models/random_forest_model.pkl"
PARAMETERS_FILE_PATH = "src/config/model_parameters.json"

def split_dataset(iris_df):
    """
    Sépare le dataset Iris en ensembles d'entraînement et de test et retourne ces derniers.
    """
    numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # Séparer les données en features (X) et target (y)
    X = iris_df[numeric_columns]
    print(X.shape)
    y = iris_df['species']
    print(y.shape)

    # Diviser le dataset en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape,y_train.shape)

    return X_train, y_train

def load_model_parameters(file_path: str):
    """
    Charge les paramètres du modèle depuis un fichier JSON.
    
    Args:
        file_path (str): Le chemin vers le fichier JSON contenant les paramètres du modèle.
    
    Returns:
        dict: Dictionnaire avec les paramètres du modèle.
    """
    try:
        with open(file_path, 'r') as f:
            params = json.load(f)
        
        # Vérifier que les paramètres sont sous forme correcte
        if isinstance(params.get("n_estimators", None), int) and \
           isinstance(params.get("max_depth", None), (int, type(None))) and \
           isinstance(params.get("max_features", None), (str, int, float)):
            return params
        else:
            print("Erreur : les paramètres du modèle sont mal formatés.")
            return {}
    except Exception as e:
        print(f"Erreur lors du chargement des paramètres : {e}")
        return {}

def train_model(X_train, y_train):
    """
    Entraîne un modèle RandomForest en utilisant les paramètres définis dans un fichier JSON.
    
    Args:
        X_train (DataFrame): Les données d'entraînement (features).
        y_train (Series): Les labels d'entraînement (target).
    """
    # Charger les paramètres depuis le fichier JSON
    model_params = load_model_parameters(PARAMETERS_FILE_PATH)

    if not model_params:
        print("Les paramètres sont incorrects ou manquants.")
        return

    print("Paramètres du modèle chargés :", model_params)

    try:
        # Créer le modèle avec les paramètres chargés
        model = RandomForestClassifier(**model_params)
        
        # Vérification des dimensions de X_train et y_train
        if len(X_train) != len(y_train):
            print(f"Erreur : nombre d'échantillons incohérent entre X_train ({len(X_train)}) et y_train ({len(y_train)})")
            return
        
        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Vérifier si le répertoire existe, sinon le créer
        if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH))

        # Sauvegarder le modèle
        joblib.dump(model, MODEL_SAVE_PATH)
        print(f"Modèle entraîné et sauvegardé sous {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Une erreur est survenue lors de l'entraînement du modèle : {e}")
