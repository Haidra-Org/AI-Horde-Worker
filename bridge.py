import time

from bridge import BridgeData, HordeJob, args, disable_voodoo
from nataili.model_manager import ModelManager
from nataili.util import logger, quiesce_logger, set_logger_verbosity


@logger.catch(reraise=True)
def bridge(model_manager, bd):
    running_jobs = []
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
