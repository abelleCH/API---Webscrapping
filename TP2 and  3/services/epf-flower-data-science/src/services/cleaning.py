from sklearn.preprocessing import StandardScaler
import pandas as pd
from io import StringIO

def process_dataset(dataset_json: str) -> dict:
    """
    Effectue le pré-traitement nécessaire sur le dataset avant qu'il puisse être utilisé pour l'entraînement du modèle.
    Cela inclut la gestion des valeurs manquantes, l'encodage des variables catégorielles, la normalisation des données, etc.
    
    Args:
        dataset_json (str): Le dataset sous forme de JSON.
    
    Returns:
        dict: Le dataset traité sous forme de JSON.
    """
    try:
        # Convertir le JSON en DataFrame
        dataset = pd.read_json(StringIO(dataset_json))

        # Renommer les colonnes (si nécessaire)
        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        dataset.columns = column_names

        # 1. Gérer les valeurs manquantes (si présentes, les supprimer)
        dataset = dataset.dropna()

        # 2. Encoder les variables catégorielles (la colonne 'species')
        dataset['species'] = dataset['species'].astype('category').cat.codes

        # 3. Normaliser les colonnes numériques
        scaler = StandardScaler()
        numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])

        return dataset
    
    except Exception as e:
        print(f"Une erreur est survenue lors du traitement du dataset: {e}")
        return None
