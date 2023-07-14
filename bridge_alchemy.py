"""This is the bridge, It connects the horde with the ML processing"""
# isort: off
# We need to import the argparser first, as it sets the necessary Switches
from worker.argparser.stable_diffusion import args
from worker.utils.set_envs import set_worker_env_vars_from_config

set_worker_env_vars_from_config()  # Get `cache_home` from `bridgeconfig.yaml` into the environment variable
import hordelib

# We need to remove these, to avoid comfyUI trying to use them
hordelib.initialise()
from hordelib.horde import SharedModelManager
from hordelib.shared_model_manager import MODEL_CATEGORY_NAMES

# isort: on

from worker.bridge_data.interrogation import InterrogationBridgeData
from worker.logger import logger, quiesce_logger, set_logger_verbosity
from worker.workers.interrogation import InterrogationWorker

if __name__ == "__main__":
    set_logger_verbosity(args.verbosity)
    quiesce_logger(args.quiet)

    bridge_data = InterrogationBridgeData()
    bridge_data.reload_data()
    SharedModelManager.load_model_managers(
        [
            MODEL_CATEGORY_NAMES.safety_checker,
            MODEL_CATEGORY_NAMES.clip,
            MODEL_CATEGORY_NAMES.blip,
            MODEL_CATEGORY_NAMES.codeformer,
            MODEL_CATEGORY_NAMES.esrgan,
            MODEL_CATEGORY_NAMES.gfpgan,
        ],
    )
    try:
        worker = InterrogationWorker(SharedModelManager.manager, bridge_data)
        worker.start()
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")
