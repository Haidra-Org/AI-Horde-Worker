"""This is the bridge, It connects the horde with the ML processing"""
from nataili.model_manager import ModelManager
from nataili.util import logger, quiesce_logger, set_logger_verbosity
from worker.argparser.stable_diffusion import args
from worker.bridge_data.stable_diffusion import StableDiffusionBridgeData
from worker.workers.stable_diffusion import StableDiffusionWorker

if __name__ == "__main__":
    set_logger_verbosity(args.verbosity)
    quiesce_logger(args.quiet)

    bridge_data = StableDiffusionBridgeData()
    model_manager = ModelManager(disable_voodoo=bridge_data.disable_voodoo.active)
    model_manager.init()
    try:
        worker = StableDiffusionWorker(model_manager, bridge_data)
        worker.start()
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")
