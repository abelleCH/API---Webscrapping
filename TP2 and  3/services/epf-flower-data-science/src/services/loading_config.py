import os, json

def load_config(config_file_path):
    """
    Loads a JSON configuration file.

    Args:
        config_file_path (str): The path to the configuration file.

    Returns:
        dict: The configuration data as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    if not os.path.exists(config_file_path):
        raise FileNotFoundError("The configuration file does not exist.")
    
    with open(config_file_path, "r") as file:
        return json.load(file)
    
def save_config(config, config_file_path):
    """
    Saves a configuration dictionary to a JSON file.

    Args:
        config (dict): The configuration data to save.
        config_file_path (str): The path to the configuration file.

    Returns:
        None
    """
    with open(config_file_path, "w") as file:
        json.dump(config, file, indent=4)
