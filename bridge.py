"""This is the worker, it's the main workhorse that deals with getting requests, and spawning data processing"""
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import requests

from nataili.model_manager import ModelManager
from nataili.util import logger, quiesce_logger, set_logger_verbosity
from worker.argparser import args
from worker.bridge_data import BridgeData
from worker.job import HordeJob
from worker.stats import bridge_stats


def reload_data(this_bridge_data):
    """This is just a utility function to reload the configuration"""
    this_bridge_data.reload_data()
    this_bridge_data.check_models(model_manager)
    this_bridge_data.reload_models(model_manager)


@logger.catch(reraise=True)
def bridge(this_model_manager, this_bridge_data):
    """This is the worker, it's the main workhorse that deals with getting requests, and spawning data processing"""
    running_jobs = []
    waiting_jobs = []
    run_count = 0
    last_config_reload = 0  # Means immediate config reload
    logger.stats("Starting new stats session")
    reload_data(this_bridge_data)
    try:
        should_stop = False
        while True:  # This is just to allow it to loop through this and handle shutdowns correctly
            should_restart = False
            with ThreadPoolExecutor(max_workers=this_bridge_data.max_threads) as executor:
                while True:
                    if should_restart:
                        break
                    try:
                        if time.time() - last_config_reload > 60:
                            this_model_manager.download_model_reference()
                            reload_data(this_bridge_data)
                            executor._max_workers = this_bridge_data.max_threads
                            logger.stats(f"Stats this session:\n{bridge_stats.get_pretty_stats()}")
                            if this_bridge_data.dynamic_models:
                                try:
                                    all_models_data = requests.get(
                                        bridge_data.horde_url + "/api/v2/status/models", timeout=10
                                    ).json()
                                    # We remove models with no queue from our list of models to load dynamically
                                    models_data = [md for md in all_models_data if md["queued"] > 0]
                                    models_data.sort(key=lambda x: (x["eta"], x["queued"]), reverse=True)
                                    top_5 = [x["name"] for x in models_data[:5]]
                                    logger.stats(f"Top 5 models by load: {', '.join(top_5)}")
                                    total_models = this_bridge_data.predefined_models.copy()
                                    new_dynamic_models = []
                                    running_models = get_running_models(running_jobs)
                                    # Sometimes a dynamic model is wwaiting in the queue,
                                    # and we do not wan to unload it
                                    # However we also don't want to keep it loaded
                                    # + the full amount of dynamic models
                                    # as we may run out of RAM/VRAM.
                                    # So we reduce the amount of dynamic models
                                    # based on how many previous dynamic models we need to keep loaded
                                    needed_previous_dynamic_models = 0
                                    for model_name in running_models:
                                        if model_name not in this_bridge_data.predefined_models:
                                            needed_previous_dynamic_models += 1
                                    for model in models_data:
                                        if model["name"] in this_bridge_data.models_to_skip:
                                            continue
                                        if model["name"] in total_models:
                                            continue
                                        if (
                                            len(new_dynamic_models) + needed_previous_dynamic_models
                                            >= this_bridge_data.number_of_dynamic_models
                                        ):
                                            break
                                        # If we've limited the amount of models to download,
                                        # then we skip models which are not already downloaded
                                        if (
                                            this_model_manager.count_available_models_by_types()
                                            >= this_bridge_data.max_models_to_download
                                            and model["name"] not in this_model_manager.get_available_models()
                                        ):
                                            continue
                                        total_models.append(model["name"])
                                        new_dynamic_models.append(model["name"])
                                    logger.info(
                                        "Dynamically loading new models to attack the relevant queue: {}",
                                        new_dynamic_models,
                                    )
                                    # Ensure we don't unload currently queued models
                                    this_bridge_data.model_names = list(set(total_models + running_models))
                                # pylint: disable=broad-except
                                except Exception as err:
                                    logger.warning("Failed to get models_req to do dynamic model loading: {}", err)
                                    trace = "".join(traceback.format_exception(type(err), err, err.__traceback__))
                                    logger.trace(trace)
                            last_config_reload = time.time()

                        if len(this_model_manager.get_loaded_models_names()) == 0:
                            time.sleep(5)
                            logger.info(
                                "No models loaded. Waiting for the first model to be up before polling the horde"
                            )
                            continue

                        # Add job to queue if we have space
                        if len(waiting_jobs) < this_bridge_data.queue_size:
                            job = pop_job(this_model_manager, this_bridge_data)
                            if job:
                                waiting_jobs.append((job))
                                # The job sends the current models loaded in the MM to
                                # the horde. That model might end up unloaded if it's dynamic
                                # so we need to ensure it will be there next iteration.
                                if job.current_model not in this_bridge_data.model_names:
                                    this_bridge_data.model_names.append(job.current_model)

                        # Start new jobs
                        while len(running_jobs) < this_bridge_data.max_threads:
                            job = None
                            # Queue disabled
                            if this_bridge_data.queue_size == 0:
                                job = pop_job(this_model_manager, this_bridge_data)
                            # Queue enabled
                            elif len(waiting_jobs) > 0:
                                job = waiting_jobs.pop(0)
                            else:
                                break
                            # Run the job
                            if job:
                                job_model = job.current_model
                                logger.debug("Starting job for model: {}", job_model)
                                running_jobs.append((executor.submit(job.start_job), time.monotonic(), job))
                                logger.debug("job submitted")
                            else:
                                logger.debug("No job to start")

                        # Check if any jobs are done
                        for (job_thread, start_time, job) in running_jobs:
                            runtime = time.monotonic() - start_time
                            if job_thread.done():
                                if job_thread.exception(timeout=1):
                                    logger.error("Job failed with exception, {}", job_thread.exception())
                                    logger.exception(job_thread.exception())
                                run_count += 1
                                logger.debug(
                                    f"Job finished successfully in {runtime:.3f}s (Total Completed: {run_count})"
                                )
                                running_jobs.remove((job_thread, start_time, job))
                                continue

                            # check if any job has run for more than 180 seconds
                            if job_thread.running() and job.is_stale():
                                logger.warning("Restarting all jobs, as a job is stale " f": {runtime:.3f}s")
                                for (
                                    inner_job_thread,
                                    inner_start_time,
                                    inner_job,
                                ) in running_jobs:  # Sometimes it's already removed
                                    running_jobs.remove((inner_job_thread, inner_start_time, inner_job))
                                    job_thread.cancel()
                                executor.shutdown(wait=False)
                                should_restart = True
                                break
                        time.sleep(0.02)  # Give the CPU a break
                    except KeyboardInterrupt:
                        should_stop = True
                        break

            if should_stop:
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, shutting down")

    logger.info("Shutting down bridge")
    executor.shutdown(wait=False)
    for job in running_jobs:
        job.cancel()
    logger.info("Shutting down bridge - Done")


# Helper functions
def pop_job(this_model_manager, this_bridge_data):
    new_job = HordeJob(this_model_manager, this_bridge_data)
    pop = new_job.get_job_from_server()  # This sleeps itself, so no need for extra
    if pop:
        logger.debug("Got a new job from the horde for model: {}", new_job.current_model)
        return new_job
    return None


def get_running_models(running_jobs):
    running_models = []
    for (job_thread, start_time, job) in running_jobs:
        running_models.append(job.current_model)
    # logger.debug(running_models)
    return running_models


if __name__ == "__main__":
    set_logger_verbosity(args.verbosity)
    if args.log_file:
        logger.add("koboldai_bridge_log.log", retention="7 days", level="warning")  # Automatically rotate too big file
    quiesce_logger(args.quiet)
    logger.add("logs/worker.log", retention="1 days", level=10)

    bridge_data = BridgeData()
    model_manager = ModelManager(disable_voodoo=bridge_data.disable_voodoo.active)
    model_manager.init()
    try:
        bridge(model_manager, bridge_data)
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")
