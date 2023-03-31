"""This is the bridge, It connects the horde with the ML processing"""
import os

# isort: off
# We need to import the argparser first, as it sets the necessry Switches
from worker.argparser.stable_diffusion import args

# isort: on

from nataili.model_manager.super import ModelManager
from nataili.util.logger import logger, quiesce_logger, set_logger_verbosity

from worker.bridge_data.stable_diffusion import StableDiffusionBridgeData
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
    model_manager = ModelManager(
        clip=True,
        compvis=True,
        diffusers=True,
        esrgan=True,
        gfpgan=True,
        safety_checker=True,
        codeformer=True,
        controlnet=True,
    )
    try:
        worker = StableDiffusionWorker(model_manager, bridge_data)
        worker.start()
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")


if __name__ == "__main__":
    check_for_old_dir()
    main()
