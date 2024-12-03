from sklearn.model_selection import train_test_split
import joblib, os, json
from sklearn.ensemble import RandomForestClassifier


MODEL_SAVE_PATH = "src/models/random_forest_model.pkl"
PARAMETERS_FILE_PATH = "src/config/model_parameters.json"

from sklearn.model_selection import train_test_split
import json

def split_dataset(iris_df):
    """
    Split the Iris dataset into training and testing sets and return both as JSON.
    """
    numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # Séparer les données en features (X) et target (y)
    X = iris_df[numeric_columns]
    y = iris_df['species']

    # Diviser le dataset en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertir les datasets divisés en format JSON
    X_train_json = X_train.to_json(orient="records")
    X_test_json = X_test.to_json(orient="records")
    y_train_json = y_train.to_json(orient="records")
    y_test_json = y_test.to_json(orient="records")

    # Retourner les datasets divisés sous forme de JSON
    return json.loads(X_train_json), json.loads(y_train_json), json.loads(X_test_json), json.loads(y_test_json)



def load_model_parameters(file_path: str):
    """
    Charge les paramètres du modèle depuis un fichier JSON.
    
    Args:
        file_path (str): Le chemin vers le fichier JSON contenant les paramètres du modèle.
    
    Returns:
        dict: Dictionnaire avec les paramètres du modèle.
    """
    with open(file_path, 'r') as f:
        params = json.load(f)

    return params

def train_model(X_train, y_train):
    """
    Entraîne un modèle RandomForest en utilisant les paramètres définis dans un fichier JSON.
    
    Args:
        X_train (DataFrame): Les données d'entraînement (features).
        y_train (Series): Les labels d'entraînement (target).
    """
    model_params = load_model_parameters(PARAMETERS_FILE_PATH)

    # Vérification de type avant de passer les paramètres au modèle
    if not isinstance(model_params.get("n_estimators", 100), int):
        print("Erreur : 'n_estimators' doit être un entier.")
        return

    if not isinstance(model_params.get("max_depth", None), (int, type(None))):
        print("Erreur : 'max_depth' doit être un entier ou None.")
        return

    if not isinstance(model_params.get("max_features", "auto"), (str, int, float)):
        print("Erreur : 'max_features' doit être une chaîne, un entier ou un float.")
        return

    try:
        # Créer le modèle avec les paramètres chargés
        model = RandomForestClassifier(**model_params)
        
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

