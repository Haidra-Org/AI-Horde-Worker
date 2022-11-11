import getpass
import os
import time

from nataili.util import logger, quiesce_logger, set_logger_verbosity
from bridge import args, disable_voodoo, BridgeData, HordeJob
from nataili.model_manager import ModelManager


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
                if j.skipped_info != None:
                    found_reason = j.skipped_info
            if found_reason:
                logger.info(f"Server {bd.horde_url} has no valid generations to do for us.{found_reason}")
        time.sleep(0.5)

def check_mm_auth(model_manager):
    if model_manager.has_authentication():
        return
    if args.hf_token:
        hf_auth = {"username": "USER", "password": args.hf_token}
        model_manager.set_authentication(hf_auth=hf_auth)
        return
    try:
        from creds import hf_password, hf_username
    except ImportError:
        hf_username = input("Please type your huggingface.co username: ")
        hf_password = getpass.getpass("Please type your huggingface.co Access Token or password: ")
    hf_auth = {"username": hf_username, "password": hf_password}
    model_manager.set_authentication(hf_auth=hf_auth)


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
