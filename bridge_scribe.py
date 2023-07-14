"""This is the bridge, It connects the horde with the ML processing"""
# isort: off
# We need to import the argparser first, as it sets the necessary Switches
from worker.argparser.scribe import args
from worker.utils.set_envs import set_worker_env_vars_from_config

set_worker_env_vars_from_config()  # Get `cache_home` from `bridgeconfig.yaml` into the environment variable

from worker.bridge_data.scribe import KoboldAIBridgeData  # noqa: E402
from worker.logger import logger, quiesce_logger, set_logger_verbosity  # noqa: E402
from worker.workers.scribe import ScribeWorker  # noqa: E402

# isort: on


def main():
    set_logger_verbosity(args.verbosity)
    quiesce_logger(args.quiet)
    bridge_data = KoboldAIBridgeData()
    bridge_data.reload_data()
    try:
        worker = ScribeWorker(bridge_data)
        worker.start()
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")


if __name__ == "__main__":
    main()
