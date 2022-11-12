import time

import requests

from bridge import BridgeData, HordeJob, args, bridge_stats, disable_voodoo
from nataili.model_manager import ModelManager
from nataili.util import logger, quiesce_logger, set_logger_verbosity


@logger.catch(reraise=True)
def bridge(model_manager, bd):
    running_jobs = []
    run_count = 0
    logger.stats("Starting new stats session")
    while True:
        bd.reload_data()
        bd.check_models(model_manager)
        bd.reload_models(model_manager)
        polling_jobs = 0
        if len(running_jobs) < bd.max_threads:
            new_job = HordeJob(model_manager, bd)
            running_jobs.append(new_job)
            # logger.debug(f"started {new_job}")
            continue
        for job in running_jobs:
            if job.is_finished():
                job.delete()
                running_jobs.remove(job)
                # logger.debug(f"removed {job}")
            elif job.is_polling():
                polling_jobs += 1
        if len(running_jobs) and polling_jobs == len(running_jobs):
            found_reason = None
            for j in running_jobs:
                if j.skipped_info is not None:
                    found_reason = j.skipped_info
            if found_reason is not None:
                logger.info(f"Server {bd.horde_url} has no valid generations to do for us.{found_reason}")
        run_count += 1
        if run_count % 240 == 0:
            logger.stats(f"Stats this session:\n{bridge_stats.get_pretty_stats()}")
            try:
                models_data = requests.get(bridge_data.horde_url + "/api/v2/status/models", timeout=10).json()
                models_data.sort(key=lambda x: (x["eta"], x["queued"] / x["performance"]), reverse=True)
                top_5 = [x["name"] for x in models_data[:5]]
                logger.stats(f"Top 5 models by load: {', '.join(top_5)}")
            except Exception as e:
                logger.debug(f"Failed to get models_req: {e}")
            run_count = 0
        time.sleep(0.5)


if __name__ == "__main__":

    set_logger_verbosity(args.verbosity)
    if args.log_file:
        logger.add("koboldai_bridge_log.log", retention="7 days", level="warning")  # Automatically rotate too big file
    quiesce_logger(args.quiet)
    logger.add("bridge.log", retention="1 days", level=10)
    model_manager = ModelManager(disable_voodoo=disable_voodoo.active)
    model_manager.init()
    bridge_data = BridgeData()
    try:
        bridge(model_manager, bridge_data)
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")
