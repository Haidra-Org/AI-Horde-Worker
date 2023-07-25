import os
from pathlib import Path

import yaml

from worker.consts import BRIDGE_CONFIG_FILE


def set_worker_env_vars_from_config():
    config_file_as_path = Path(BRIDGE_CONFIG_FILE)
    if config_file_as_path.exists():
        with open(config_file_as_path, "rt", encoding="utf-8", errors="ignore") as configfile:
            config = yaml.safe_load(configfile)
            if "cache_home" in config:
                os.environ["AIWORKER_CACHE_HOME"] = config["cache_home"]
            if "temp_dir" in config:
                os.environ["AIWORKER_TEMP_DIR"] = config["temp_dir"]
        return True
    return False


def get_models_to_load():
    config_file_as_path = Path(BRIDGE_CONFIG_FILE)
    if config_file_as_path.exists():
        with open(config_file_as_path, "rt", encoding="utf-8", errors="ignore") as configfile:
            config = yaml.safe_load(configfile)
            return config.get("models_to_load", [])
    return None
