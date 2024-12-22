import os, json
import pytest
from loading_config import load_config, save_config

def test_load_config_success():
    """
    Test the successful loading of the JSON configuration file.
    """
    config_data = {"key1": "value1", "key2": "value2"}
    config_file_path = "test_config.json"

    # Write the mock configuration data to a file
    with open(config_file_path, "w") as file:
        json.dump(config_data, file, indent=4)

    result = load_config(config_file_path)

    assert result == config_data

    os.remove(config_file_path)

def test_load_config_file_not_found():
    """
    Test the case when the configuration file does not exist.
    """
    config_file_path = "non_existent_config.json"

    with pytest.raises(FileNotFoundError, match="The configuration file does not exist."):
        load_config(config_file_path)

def test_save_config_success():
    """
    Test the successful saving of a JSON configuration file.
    """
    config_data = {"key1": "value1", "key2": "value2"}
    config_file_path = "test_save_config.json"

    save_config(config_data, config_file_path)

    with open(config_file_path, "r") as file:
        saved_data = json.load(file)

    assert saved_data == config_data

    os.remove(config_file_path)
