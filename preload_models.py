import os
import pathlib

from worker.utils.set_envs import get_models_to_load, set_worker_env_vars_from_config

set_worker_env_vars_from_config()  # Get `cache_home` from `bridgeConfig.yaml` into the environment variable

import hordelib  # noqa: E402

hordelib.initialise()

from hordelib.shared_model_manager import MODEL_CATEGORY_NAMES, SharedModelManager  # noqa: E402

print("Model directory to use: ")
AIWORKER_CACHE_HOME = os.environ.get("AIWORKER_CACHE_HOME", None)
if AIWORKER_CACHE_HOME is None:
    print("AIWORKER_CACHE_HOME is not set.")
    print("Please set `cache_home` in your bridge data to the directory where you want to store models.")
    print("Or, set AIWORKER_CACHE_HOME in your environment variables.")
    exit(1)
cache_home_path = pathlib.Path(AIWORKER_CACHE_HOME).resolve()

SharedModelManager.load_model_managers(
    [
        MODEL_CATEGORY_NAMES.blip,
        MODEL_CATEGORY_NAMES.clip,
        MODEL_CATEGORY_NAMES.codeformer,
        MODEL_CATEGORY_NAMES.compvis,
        MODEL_CATEGORY_NAMES.controlnet,
        MODEL_CATEGORY_NAMES.esrgan,
        MODEL_CATEGORY_NAMES.gfpgan,
        MODEL_CATEGORY_NAMES.safety_checker,
    ],
)
if SharedModelManager.manager.compvis is None:
    print("CompVis model manager is not loaded.")
    exit(1)


def preload_models():
    if SharedModelManager.manager.compvis is None:
        raise Exception("CompVis model manager is not loaded.")

    all_model_names = SharedModelManager.manager.compvis.model_reference.keys()
    all_downloaded_models = SharedModelManager.manager.compvis.available_models

    all_models_not_downloaded = set(all_model_names) - set(all_downloaded_models)
    all_models_not_downloaded = list(all_models_not_downloaded)

    if len(all_models_not_downloaded) == 0:
        print("All models are downloaded.")
        return

    models_to_load = get_models_to_load()
    if not models_to_load:
        print("No models to load.")
        return

    models_to_download = set(models_to_load) - set(all_downloaded_models)

    if len(models_to_download) == 0:
        print("All models to load are downloaded.")
        return

    print(
        f"This is going to download {len(models_to_download)} models. They are at least 2gb each. Are you sure? (y/n)",
    )

    if input() != "y":
        return

    for model_name in models_to_download:
        print(f"Downloading {model_name}...")
        if not SharedModelManager.manager.compvis.download_model(model_name):
            print(f"Failed to download {model_name}.")
            continue
        SharedModelManager.manager.compvis.load(model_name)
        SharedModelManager.manager.compvis.move_to_disk_cache(model_name)
        SharedModelManager.manager.compvis.unload_model(model_name)
        print(f"Downloaded {model_name}.")


def build_cache(models):
    if SharedModelManager.manager.compvis is None:
        raise Exception("CompVis model manager is not loaded.")

    print("Building cache for models...")
    print("This is going to write at least 2.2 gb per model to your tmp_dir.")
    tmp_dir = os.environ.get("AIWORKER_TEMP_DIR", "./tmp")
    print(f"tmp_dir: {tmp_dir}")
    models_num = len(SharedModelManager.manager.available_models)
    print(f"Models to build cache for: {models_num}")
    print("Are you sure? (y/n)")
    if input() != "y":
        return

    for downloaded_model in models:
        if SharedModelManager.manager.compvis.have_model_cache(downloaded_model):
            continue

        if not SharedModelManager.manager.compvis.load(downloaded_model):  # noqa: SIM102
            if not SharedModelManager.manager.download_model(downloaded_model):
                print(f"Failed to download {downloaded_model}.")
                continue
        print(f"Building cache for {downloaded_model}...")
        SharedModelManager.manager.compvis.move_to_disk_cache(downloaded_model)
        SharedModelManager.manager.compvis.unload_model(downloaded_model)


if __name__ == "__main__":
    preload_models()
    build_cache(SharedModelManager.manager.compvis.available_models)
