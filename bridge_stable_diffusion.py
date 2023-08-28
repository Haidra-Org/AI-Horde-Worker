"""This is the bridge, It connects the horde with the ML processing"""
import os

# isort: off
# We need to import the argparser first, as it sets the necessry Switches
from worker.argparser.stable_diffusion import args
from worker.utils.set_envs import set_worker_env_vars_from_config

set_worker_env_vars_from_config()  # Get `cache_home` from `bridgeconfig.yaml` into the environment variable

import hordelib

# We need to remove these, to avoid comfyUI trying to use them
hordelib.initialise()
from hordelib.horde import SharedModelManager
from hordelib.consts import MODEL_CATEGORY_NAMES

# isort: on
from worker.bridge_data.stable_diffusion import StableDiffusionBridgeData
from worker.logger import logger, quiesce_logger, set_logger_verbosity
from worker.workers.stable_diffusion import StableDiffusionWorker


def check_for_old_dir():
    models_folder = "models/custom"
    compvis_folder = "nataili/compvis"
    if not os.path.exists(compvis_folder):
        os.makedirs(compvis_folder)
    if os.path.exists(models_folder):
        print(f"{models_folder} folder exists.")
        answer = input(f"Do you want to move its contents to {compvis_folder} (recommended) [Y/N]? ")
        if answer.lower() == "y":
            import shutil

            contents = os.listdir(models_folder)
            for item in contents:
                shutil.move(os.path.join(models_folder, item), compvis_folder)
            print(f"Contents of {models_folder} have been moved to {compvis_folder}.")
            answer = input("Do you want to delete the models/ folder (not needed anymore) [Y/N]? ")
            if answer.lower() == "y":
                shutil.rmtree("models/")
                print("models/ folder has been deleted.")
            else:
                print("old models/ folder left intact.")
        else:
            print("Existing custom models left in their previous location.")


def main():
    set_logger_verbosity(args.verbosity)
    quiesce_logger(args.quiet)
    # TODO: Remove check after fully deprecating arg.
    if args.skip_md5:
        logger.warning("DeprecationWarning: `--skip_md5` has been deprecated. Please use `--skip_checksum` instead.")

    bridge_data = StableDiffusionBridgeData()
    try:
        bridge_data.reload_data()

        SharedModelManager.load_model_managers(
            [
                MODEL_CATEGORY_NAMES.blip,
                MODEL_CATEGORY_NAMES.clip,
                MODEL_CATEGORY_NAMES.compvis,
                MODEL_CATEGORY_NAMES.controlnet,
                MODEL_CATEGORY_NAMES.codeformer,
                MODEL_CATEGORY_NAMES.gfpgan,
                MODEL_CATEGORY_NAMES.esrgan,
                MODEL_CATEGORY_NAMES.safety_checker,
                MODEL_CATEGORY_NAMES.lora,
                MODEL_CATEGORY_NAMES.ti,
            ],
        )

        worker = StableDiffusionWorker(SharedModelManager.manager, bridge_data)

        worker.model_manager = SharedModelManager.manager

        annotators_preloaded_successfully = False
        if bridge_data.allow_controlnet:
            annotators_preloaded_successfully = SharedModelManager.preloadAnnotators()
        if not annotators_preloaded_successfully:
            logger.warning("Annotators were not preloaded. ControlNet will not be available.")
            bridge_data.allow_controlnet = False

        worker.start()

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")


if __name__ == "__main__":
    check_for_old_dir()
    main()
