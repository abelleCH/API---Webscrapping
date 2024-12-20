import pandas as pd 
def load_dataset_from_url(url):
    """
    Loads a dataset from the provided URL and returns it as a JSON.

    Args:
        url (str): The URL pointing to the dataset (CSV file).

    Returns:
        dict: The dataset as a JSON object in 'records' orientation.

    Raises:
        ValueError: If the dataset at the URL is not in the expected format or cannot be read.
    """

    try:
        dataset_df = pd.read_csv(url, header=None)

        dataset_json = dataset_df.to_json(orient="records")

        return dataset_json
    except Exception as e:
            raise ValueError(f"Error loading dataset from URL: {e}")