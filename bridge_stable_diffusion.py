"""This is the bridge, It connects the horde with the ML processing"""
from nataili.model_manager.super import ModelManager
from nataili.util.logger import logger, quiesce_logger, set_logger_verbosity
from worker.argparser.stable_diffusion import args
from worker.bridge_data.stable_diffusion import StableDiffusionBridgeData
from worker.workers.stable_diffusion import StableDiffusionWorker

def main():
    set_logger_verbosity(args.verbosity)
    quiesce_logger(args.quiet)

    bridge_data = StableDiffusionBridgeData()
    model_manager = ModelManager(
        compvis=True,
        diffusers=True,
        esrgan=True,
        gfpgan=True,
        safety_checker=True,
        codeformer=True,
    )
    try:
        worker = StableDiffusionWorker(model_manager, bridge_data)
        worker.start()
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")

if __name__ == "__main__":
    main()
