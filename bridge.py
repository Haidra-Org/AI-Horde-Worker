import time
from concurrent.futures import ThreadPoolExecutor

import requests

from bridge import BridgeData, HordeJob, args, bridge_stats, disable_voodoo
from nataili.model_manager import ModelManager
from nataili.util import logger, quiesce_logger, set_logger_verbosity


def reload_data(bd):
    bd.reload_data()
    bd.check_models(model_manager)
    bd.reload_models(model_manager)


@logger.catch(reraise=True)
def bridge(model_manager, bd):
    running_jobs = []
    run_count = 0
    logger.stats("Starting new stats session")
    reload_data(bd)
    with ThreadPoolExecutor(max_workers=bd.max_threads) as executor:
        loop_count = 0
        while True:
            if len(model_manager.get_loaded_models_names()) == 0:
                time.sleep(2)
                logger.info("No models loaded. Waiting for the first model to be up before polling the horde")
                continue
            loop_count += 1
            if loop_count % 10 == 0:
                reload_data(bd)
                loop_count = 0

            pop_count = 0
            while len(running_jobs) < bd.max_threads:
                pop_count += 1
                if pop_count > 10:  # Just to allow reload to fire
                    break
                new_job = HordeJob(model_manager, bd)
                pop = new_job.get_job_from_server()  # This sleeps itself, so no need for extra
                if pop is None:
                    continue
                logger.info("Got a new job from the horde")
                running_jobs.append(executor.submit(new_job.start_job, pop))

            for job in running_jobs:
                if job.done():
                    logger.debug("Job finished successfully")
                    running_jobs.remove(job)
                    run_count += 1

                if job.exception():
                    logger.debug("Job failed with exception")
                    logger.exception(job.exception())
                    running_jobs.remove(job)

                # check if any job has run for more than 60 seconds
                if job.running() and job.running_for() > 180:
                    running_jobs.remove(job)
                    job.cancel()
                    logger.warning("Cancelled job as was running for more than 180 seconds: %s", job.running_for())

            if run_count % 100 == 0:
                logger.stats(f"Stats this session:\n{bridge_stats.get_pretty_stats()}")
                try:
                    models_data = requests.get(bridge_data.horde_url + "/api/v2/status/models", timeout=10).json()
                    models_data.sort(key=lambda x: (x["eta"], x["queued"] / x["performance"]), reverse=True)
                    top_5 = [x["name"] for x in models_data[:5]]
                    logger.stats(f"Top 5 models by load: {', '.join(top_5)}")
                except Exception as e:
                    logger.debug(f"Failed to get models_req: {e}")
                run_count = 0

            time.sleep(0.1)  # Give the CPU a break


if __name__ == "__main__":

    set_logger_verbosity(args.verbosity)
    if args.log_file:
        logger.add("koboldai_bridge_log.log", retention="7 days", level="warning")  # Automatically rotate too big file
    quiesce_logger(args.quiet)
    logger.add("logs/bridge.log", retention="1 days", level=10)
    model_manager = ModelManager(disable_voodoo=disable_voodoo.active)
    model_manager.init()
    bridge_data = BridgeData()
    try:
        bridge(model_manager, bridge_data)
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")
