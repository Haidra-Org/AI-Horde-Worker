import os
from pathlib import Path

import yaml

from worker.consts import BRIDGE_CONFIG_FILE


def set_aiworker_cache_home_from_config():
    config_file_as_path = Path(BRIDGE_CONFIG_FILE)
    if config_file_as_path.exists():
        with open(config_file_as_path, "rt", encoding="utf-8", errors="ignore") as configfile:
            config = yaml.safe_load(configfile)
            if "cache_home" in config:
                os.environ["AIWORKER_CACHE_HOME"] = config["cache_home"]
        return True
    return False
