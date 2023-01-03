"""This is the bridge, It connects the horde with the ML processing"""
from nataili.model_manager import ModelManager
from nataili.util import logger, quiesce_logger, set_logger_verbosity
from worker.argparser import args
from worker.workers.prompt import StableDiffusionWorker
from worker.bridge_data.stable_diffusion import StableDiffusionBridgeData



if __name__ == "__main__":
    set_logger_verbosity(args.verbosity)
    if args.log_file:
        logger.add("koboldai_bridge_log.log", retention="7 days", level="warning")  # Automatically rotate too big file
    quiesce_logger(args.quiet)
    logger.add("logs/worker.log", retention="1 days", level=10)

    bridge_data = StableDiffusionBridgeData()
    model_manager = ModelManager(disable_voodoo=bridge_data.disable_voodoo.active)
    model_manager.init()
    try:
        worker = PromptWorker(model_manager, bridge_data)
        worker.start()
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")
