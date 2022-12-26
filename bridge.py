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
            with ThreadPoolExecutor(max_workers=this_bridge_data.max_threads) as executor:
                while True:
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
                                    for model in models_data:
                                        if model["name"] in this_bridge_data.models_to_skip:
                                            continue
                                        if model["name"] in total_models:
                                            continue
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
                                        if len(new_dynamic_models) >= this_bridge_data.number_of_dynamic_models:
                                            break
                                    logger.info(
                                        "Dynamically loading new models to attack the relevant queue: {}",
                                        new_dynamic_models,
                                    )
                                    this_bridge_data.model_names = total_models
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
                            job, pop = pop_job(this_model_manager, this_bridge_data)
                            if pop:
                                waiting_jobs.append((job, pop))

                        # Start new jobs
                        while len(running_jobs) < this_bridge_data.max_threads:
                            job, pop = (None, None)
                            # Queue disabled
                            if this_bridge_data.queue_size == 0:
                                job, pop = pop_job(this_model_manager, this_bridge_data)
                            # Queue enabled
                            elif len(waiting_jobs) > 0:
                                job, pop = waiting_jobs.pop(0)
                            else:
                                break
                            # Run the job
                            if pop:
                                job_model = pop.get("model", "Unknown")
                                logger.debug("Starting job for model: {}", job_model)
                                running_jobs.append((executor.submit(job.start_job, pop), time.monotonic()))
                                logger.debug("job submitted")
                            else:
                                logger.debug("No job to start")

                        # Check if any jobs are done
                        for (job, start_time) in running_jobs:
                            runtime = time.monotonic() - start_time
                            if job.done():
                                if job.exception(timeout=1):
                                    logger.error("Job failed with exception, {}", job.exception())
                                    logger.exception(job.exception())
                                run_count += 1
                                logger.debug(
                                    f"Job finished successfully in {runtime:.3f}s (Total Completed: {run_count})"
                                )
                                running_jobs.remove((job, start_time))
                                continue

                            # check if any job has run for more than 180 seconds
                            if job.running() and runtime > 180:
                                logger.warning(
                                    "Restarting all jobs, as a job was running "
                                    f"for more than 180 seconds: {runtime:.3f}s"
                                )
                                for (inner_job, inner_start_time) in running_jobs:  # Sometimes it's already removed
                                    running_jobs.remove((inner_job, inner_start_time))
                                    job.cancel()
                                executor.shutdown(wait=False)
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
        job_model = pop.get("model", "Unknown")
        logger.debug("Got a new job from the horde for model: {}", job_model)
        return new_job, pop
    return None, None


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
