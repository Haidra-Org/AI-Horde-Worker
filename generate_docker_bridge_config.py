import os
import random

import yaml

bridge_data_file = "bridgeData.yaml"

if os.path.isfile(bridge_data_file):
    print("bridgeData.yaml file already exists.")
    exit(0)


def get_list_environment_variable(env_var_name, default_values=None):
    env_var_value = os.getenv(env_var_name, "")
    if env_var_value == "":
        return default_values if default_values else []
    items = [item.strip() for item in env_var_value.split(",")]
    if len(items) == 1:
        return [items[0]]
    return items


def get_bool_env(env_var_name, default_value):
    value = os.getenv(env_var_name, default_value)
    if value.lower() == "false":
        return False
    if value.lower() == "true":
        return True
    raise ValueError(f"The value of {env_var_name} must be 'true' or 'false', but was {value}.")


def get_int_env(env_var_name, default_value):
    value = os.getenv(env_var_name, default_value)
    return int(value)


def get_worker_name():
    """
    HORDE_WORKER_NAME environment variable is used if it is set

    if unset, a custom prefix can be set using the envionrment variable HORDE_WORKER_PREFIX
    otherwise the default of DockerWorker will be used

    a random string of numbers will be attached to the end of the prefix to allow easy
    deployment of ephemeral containers
    """
    worker_name = os.getenv("HORDE_WORKER_NAME")
    if not worker_name:
        worker_name_prefix = os.getenv("HORDE_WORKER_PREFIX", "DockerWorker")
        worker_name = worker_name_prefix + "#" + "".join(random.choices("0123456789", k=10))
    return worker_name


config = {
    "horde_url": os.getenv("HORDE_URL", "https://stablehorde.net"),
    "worker_name": get_worker_name(),
    "api_key": os.getenv("HORDE_API_KEY", "0000000000"),
    "priority_usernames": get_list_environment_variable("HORDE_PRIORITY_USERNAMES"),
    "max_threads": get_int_env("HORDE_MAX_THREADS", "1"),
    "queue_size": get_int_env("HORDE_QUEUE_SIZE", "0"),
    "require_upfront_kudos": get_bool_env("HORDE_REQUIRE_UPFRONT_KUDOS", "false"),
    "max_power": get_int_env("HORDE_MAX_POWER", "8"),
    "nsfw": get_bool_env("HORDE_NSFW", "true"),
    "censor_nsfw": get_bool_env("HORDE_CENSOR_NSFW", "false"),
    "blacklist": get_list_environment_variable("HORDE_BLACKLIST"),
    "censorlist": get_list_environment_variable("HORDE_CENSORLIST"),
    "allow_img2img": get_bool_env("HORDE_ALLOW_IMG2IMG", "true"),
    "allow_painting": get_bool_env("HORDE_ALLOW_PAINTING", "true"),
    "allow_unsafe_ip": get_bool_env("HORDE_ALLOW_UNSAFE_IP", "true"),
    "allow_post_processing": get_bool_env("HORDE_ALLOW_POST_PROCESSING", "true"),
    "allow_controlnet": get_bool_env("HORDE_ALLOW_CONTROLNET", "false"),
    "dynamic_models": get_bool_env("HORDE_DYNAMIC_MODELS", "true"),
    "number_of_dynamic_models": get_int_env("HORDE_NUMBER_OF_DYNAMIC_MODELS", "3"),
    "max_models_to_download": get_int_env("HORDE_MAX_MODELS_TO_DOWNLOAD", "10"),
    "stats_output_frequency": get_int_env("HORDE_STATS_OUTPUT_FREQUENCY", "30"),
    "nataili_cache_home": os.getenv("HORDE_NATAILI_CACHE_HOME", "/cache/nataili"),
    "low_vram_mode": get_bool_env("HORDE_LOW_VRAM_MODE", "true"),
    "enable_model_cache": get_bool_env("HORDE_ENABLE_MODEL_CACHE", "false"),
    "always_download": get_bool_env("HORDE_ALWAYS_DOWNLOAD", "false"),
    "ray_temp_dir": os.getenv("HORDE_RAY_TEMP_DIR", "/cache/ray"),
    "disable_voodoo": get_bool_env("HORDE_DISABLE_VOODOO", "false"),
    "disable_terminal_ui": get_bool_env("HORDE_DISABLE_TERMINAL_UI", "false"),
    "models_to_load": get_list_environment_variable(
        "HORDE_MODELS_TO_LOAD",
        [
            "stable_diffusion_2.1",
            "stable_diffusion",
        ],
    ),
    "models_to_skip": get_list_environment_variable(
        "HORDE_MODELS_TO_SKIP",
        [
            "stable_diffusion_inpainting",
        ],
    ),
    "forms": get_list_environment_variable(
        "HORDE_FORMS",
        [
            "caption",
            "nsfw",
        ],
    ),
}

with open(bridge_data_file, "w") as file:
    print("Created bridgeData.yaml")
    yaml.dump(config, file)
