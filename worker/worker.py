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

class Worker:

    def __init__(self, this_model_manager, this_bridge_data):
        self.model_manager = this_model_manager
        self.bridge_data = this_bridge_data
        self.running_jobs = []
        self.waiting_jobs = []
        self.run_count = 0
        self.last_config_reload = 0
        self.should_stop = False
        self.should_restart = False
        self.consecutive_executor_restarts = 0
        self.consecutive_failed_jobs = 0
        self.executor = None
        self.reload_data()
        logger.stats("Starting new stats session")

    def start(self):
        while True:  # This is just to allow it to loop through this and handle shutdowns correctly
            self.should_restart = False
            self.consecutive_failed_jobs = 0
            with ThreadPoolExecutor(max_workers=self.bridge_data.max_threads) as self.executor:
                while True:
                    if self.should_restart:
                        self.executor.shutdown(wait=False)
                        break
                    try:
                        self.process_jobs()
                    except KeyboardInterrupt:
                        self.should_stop = True
                    break
            if self.should_stop:
                logger.init("Worker", status="Shutting Down")
                try:
                    for job in self.running_jobs:
                        job.cancel()
                    self.executor.shutdown(wait=False)
                # In case it's already shut-down
                except Exception as e:
                    logger.init_err(f"Worker Exception: {e}", status="Shut Down")
                    pass
                logger.init_ok("Worker", status="Shut Down")
                break

    def reload_data(self):
        """This is just a utility function to reload the configuration"""
        self.bridge_data.reload_data()
        self.bridge_data.check_models(model_manager)
        self.bridge_data.reload_models(model_manager)

    def process_jobs(self):
        if time.time() - self.last_config_reload > 60:
            self.reload_bridge_data()
        if self.can_process_jobs():
            time.sleep(5)
            return
        # Add job to queue if we have space
        if len(self.waiting_jobs) < self.bridge_data.queue_size:
            self.add_job_to_queue()
        # Start new jobs
        while len(self.running_jobs) < self.bridge_data.max_threads:
            if not self.start_job():
                break
        # Check if any jobs are done
        for (job_thread, start_time, job) in self.running_jobs:
            self.check_running_jobs_status(job_thread, start_time, job)
            if self.should_restart or self.should_stop:
                break
        # Give the CPU a break
        time.sleep(0.02)  

    # Setting it as it's own function so that it can be overriden
    def can_process_jobs(self):
        can_do = len(self.model_manager.get_loaded_models_names()) > 0
        if not can_do:
            logger.info(
                "No models loaded. Waiting for the first model to be up before polling the horde"
            )
        return can_do

    # We want this to be extendable as well
    def add_job_to_queue(self):
        '''Picks up a job from the horde and adds it to the local queue'''
        job = self.pop_job()
        if job:
            self.waiting_jobs.append((job))
            # The job sends the current models loaded in the MM to
            # the horde. That model might end up unloaded if it's dynamic
            # so we need to ensure it will be there next iteration.
            if job.current_model not in self.bridge_data.model_names:
                self.bridge_data.model_names.append(job.current_model)

    def pop_job(self):
        new_job = HordeJob(self.model_manager, self.bridge_data)
        pop = new_job.get_job_from_server()  # This sleeps itself, so no need for extra
        if pop:
            logger.debug("Got a new job from the horde for model: {}", new_job.current_model)
            return new_job
        return None

    def start_job(self):
        job = None
        # Queue disabled
        if self.bridge_data.queue_size == 0:
            job = pop_job(self.model_manager, self.bridge_data)
        # Queue enabled
        elif len(self.waiting_jobs) > 0:
            job = self.waiting_jobs.pop(0)
        else:
            #  This causes a break on the main loop outside
            return False
        # Run the job
        if job:
            job_model = job.current_model
            logger.debug("Starting job for model: {}", job_model)
            self.running_jobs.append((self.executor.submit(job.start_job), time.monotonic(), job))
            logger.debug("job submitted")
        else:
            logger.debug("No job to start")
        return True    

    def check_running_job_status(self, job_thread, start_time, job):
        runtime = time.monotonic() - start_time
        if job_thread.done():
            if job_thread.exception(timeout=1):
                logger.error("Job failed with exception, {}", job_thread.exception())
                logger.exception(job_thread.exception())
                if self.consecutive_executor_restarts > 0:
                    logger.critical(
                        "Worker keeps crashing after thread executor restart. "
                        "Cannot be salvaged. Aborting!"
                    )
                    self.should_stop = True
                    return
                self.consecutive_failed_jobs += 1
                if self.consecutive_failed_jobs >= 5:
                    logger.critical(
                        "Too many consecutive jobs have failed. "
                        "Restarting thread executor and hope we recover..."
                    )
                    self.should_restart = True
                    self.consecutive_executor_restarts += 1
                    return
            else:
                self.consecutive_failed_jobs = 0
                self.consecutive_executor_restarts = 0
            self.run_count += 1
            logger.debug(
                f"Job finished successfully in {runtime:.3f}s (Total Completed: {self.run_count})"
            )
            self.running_jobs.remove((job_thread, start_time, job))
            return

        # check if any job has run for more than 180 seconds
        if job_thread.running() and job.is_stale():
            logger.warning("Restarting all jobs, as a job is stale " f": {runtime:.3f}s")
            for (
                inner_job_thread,
                inner_start_time,
                inner_job,
            ) in self.running_jobs:  # Sometimes it's already removed
                self.running_jobs.remove((inner_job_thread, inner_start_time, inner_job))
                job_thread.cancel()
            self.should_restart = True
            return

    def get_running_models(self):
        running_models = []
        for (job_thread, start_time, job) in self.running_jobs:
            running_models.append(job.current_model)
        # logger.debug(running_models)
        return running_models

    def reload_bridge_data(self):
        self.model_manager.download_model_reference()
        self.reload_data()
        self.executor._max_workers = self.bridge_data.max_threads
        logger.stats(f"Stats this session:\n{bridge_stats.get_pretty_stats()}")
        if self.bridge_data.dynamic_models:
            try:
                self.calculate_dynamic_models()
            # pylint: disable=broad-except
            except Exception as err:
                logger.warning("Failed to get models_req to do dynamic model loading: {}", err)
                trace = "".join(traceback.format_exception(type(err), err, err.__traceback__))
                logger.trace(trace)
        self.last_config_reload = time.time()        

    def calculate_dynamic_models(self):
        all_models_data = requests.get(
            bridge_data.horde_url + "/api/v2/status/models", timeout=10
        ).json()
        # We remove models with no queue from our list of models to load dynamically
        models_data = [md for md in all_models_data if md["queued"] > 0]
        models_data.sort(key=lambda x: (x["eta"], x["queued"]), reverse=True)
        top_5 = [x["name"] for x in models_data[:5]]
        logger.stats(f"Top 5 models by load: {', '.join(top_5)}")
        total_models = self.bridge_data.predefined_models.copy()
        new_dynamic_models = []
        running_models = self.get_running_models(self.running_jobs)
        # Sometimes a dynamic model is wwaiting in the queue,
        # and we do not wan to unload it
        # However we also don't want to keep it loaded
        # + the full amount of dynamic models
        # as we may run out of RAM/VRAM.
        # So we reduce the amount of dynamic models
        # based on how many previous dynamic models we need to keep loaded
        needed_previous_dynamic_models = 0
        for model_name in running_models:
            if model_name not in self.bridge_data.predefined_models:
                needed_previous_dynamic_models += 1
        for model in models_data:
            if model["name"] in self.bridge_data.models_to_skip:
                continue
            if model["name"] in total_models:
                continue
            if (
                len(new_dynamic_models) + needed_previous_dynamic_models
                >= self.bridge_data.number_of_dynamic_models
            ):
                break
            # If we've limited the amount of models to download,
            # then we skip models which are not already downloaded
            if (
                self.model_manager.count_available_models_by_types()
                >= self.bridge_data.max_models_to_download
                and model["name"] not in self.model_manager.get_available_models()
            ):
                continue
            total_models.append(model["name"])
            new_dynamic_models.append(model["name"])
        logger.info(
            "Dynamically loading new models to attack the relevant queue: {}",
            new_dynamic_models,
        )
        # Ensure we don't unload currently queued models
        self.bridge_data.model_names = list(set(total_models + running_models))
