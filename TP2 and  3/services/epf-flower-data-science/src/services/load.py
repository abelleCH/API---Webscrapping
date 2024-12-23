import pandas as pd
import os
from pathlib import Path
from fastapi import HTTPException
os.environ["KAGGLE_CONFIG_DIR"] = "src/config"
from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = 'src/data/'  
CONFIG_FILE_PATH = 'src/config/config.json'  

def download_kaggle_dataset(url: str):
    """
    Downloads a Kaggle dataset from the specified URL, extracts the files, 
    and returns the data as a JSON object.

    Args:
    - url (str): The URL of the Kaggle dataset to download.

    Returns:
    - json_data (list): A list of records (dict) representing the dataset.

    Raises:
    - HTTPException: If any error occurs during the download or data processing.
    """
    try:
        os.environ['KAGGLE_CONFIG_DIR'] = 'src/config/kaggle.json'  
        api = KaggleApi()
        api.authenticate()

        dataset_spec = url.split('/')[-2] + '/' + url.split('/')[-1]

        dataset_name = url.split('/')[-1]
        destination = Path(DATA_DIR) / dataset_name  
        destination.mkdir(parents=True, exist_ok=True) 

        print(f"Downloading dataset {dataset_spec} to {destination}...")
        api.dataset_download_files(dataset_spec, path=str(destination), unzip=True)

        print(f"Dataset downloaded to: {destination}")
        csv_file = next((file for file in destination.glob("*.csv")), None)
        
        if csv_file is None:
            raise HTTPException(status_code=404, detail="No CSV file found in the downloaded dataset.")

        df = pd.read_csv(csv_file)
        if df.empty:
            raise HTTPException(status_code=404, detail="The CSV file is empty.")

        json_data = df.to_dict(orient="records")

        return json_data

    except HTTPException as e:
        raise e  
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while downloading the dataset: {str(e)}"
        )
