import pandas as pd 
def load_dataset_from_url(url):
    """
    Charge un dataset à partir de l'URL fournie et le retourne sous forme de JSON.
    """
        # Lire le fichier CSV directement depuis l'URL fournie
    dataset_df = pd.read_csv(url, header=None)

        # Convertir le DataFrame en JSON
    dataset_json = dataset_df.to_json(orient="records")

        # Retourner le JSON du dataset
    return dataset_json