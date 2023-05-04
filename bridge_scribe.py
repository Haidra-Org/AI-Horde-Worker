"""This is the bridge, It connects the horde with the ML processing"""
# isort: off
# We need to import the argparser first, as it sets the necessary Switches
from worker.argparser.scribe import args

# isort: on

from worker.bridge_data.scribe import KoboldAIBridgeData
from worker.logger import logger, quiesce_logger, set_logger_verbosity
from worker.workers.scribe import ScribeWorker


def main():
    set_logger_verbosity(args.verbosity)
    quiesce_logger(args.quiet)
    bridge_data = KoboldAIBridgeData()
    try:
        worker = ScribeWorker(bridge_data)
        worker.start()
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")


if __name__ == "__main__":
    main()
