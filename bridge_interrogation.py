"""This is the bridge, It connects the horde with the ML processing"""
# This needs to load first as it sets the disable_voodoo switches
from worker.argparser.interrogation import args  # isort: skip
from nataili.model_manager import ModelManager
from nataili.util import logger, quiesce_logger, set_logger_verbosity
from worker.bridge_data.interrogation import InterrogationBridgeData
from worker.workers.interrogation import InterrogationWorker

if __name__ == "__main__":
    set_logger_verbosity(args.verbosity)
    quiesce_logger(args.quiet)

    bridge_data = InterrogationBridgeData()
    model_manager = ModelManager(disable_voodoo=bridge_data.disable_voodoo.active)
    model_manager.init()
    try:
        worker = InterrogationWorker(model_manager, bridge_data)
        worker.start()
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")
