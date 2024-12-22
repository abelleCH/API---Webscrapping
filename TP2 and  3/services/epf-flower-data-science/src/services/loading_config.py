import os, json

def load_config(config_file_path):
    """
    Charge le fichier JSON de configuration.
    """
    if not os.path.exists(config_file_path):
        raise FileNotFoundError("The configuration file does not exist.")
    
    with open(config_file_path, "r") as file:
        return json.load(file)
    
def save_config(config, config_file_path):
    """
    Sauvegarde le fichier JSON de configuration.
    """
    with open(config_file_path, "w") as file:
        json.dump(config, file, indent=4)    