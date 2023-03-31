"""This is the worker, it's the main workhorse that deals with getting requests, and spawning data processing"""
import traceback

import requests
from nataili.util.logger import logger

from worker.jobs.poppers import StableDiffusionPopper
from worker.jobs.stable_diffusion import StableDiffusionHordeJob
from worker.workers.framework import WorkerFramework


class StableDiffusionWorker(WorkerFramework):
    def __init__(self, this_model_manager, this_bridge_data):
        super().__init__(this_model_manager, this_bridge_data)
        self.PopperClass = StableDiffusionPopper
        self.JobClass = StableDiffusionHordeJob

    # Setting it as it's own function so that it can be overriden
    def can_process_jobs(self):
        loaded_models = len(self.model_manager.compvis.get_loaded_models_names()) + len(
            self.model_manager.diffusers.get_loaded_models_names(),
        )
        can_do = loaded_models > 0
        if not can_do:
            logger.info("No models loaded. Waiting for the first model to be up before polling the horde")
        return can_do

    # We want this to be extendable as well
    def add_job_to_queue(self):
        job = super().add_job_to_queue()
        if job and job.current_model not in self.bridge_data.model_names:
            # The job sends the current models loaded in the MM to
            # the horde. That model might end up unloaded if it's dynamic
            # so we need to ensure it will be there next iteration.
            self.bridge_data.model_names.append(job.current_model)

    def pop_job(self):
        return super().pop_job()

    def get_running_models(self):
        return [job.current_model for job_thread, start_time, job in self.running_jobs]

    def calculate_dynamic_models(self):
        all_models_data = requests.get(f"{self.bridge_data.horde_url}/api/v2/status/models", timeout=10).json()
        # We remove models with no queue from our list of models to load dynamically
        models_data = [md for md in all_models_data if md["queued"] > 0]
        models_data.sort(key=lambda x: (x["eta"], x["queued"]), reverse=True)
        top_5 = [x["name"] for x in models_data[:5]]
        logger.stats(f"Top 5 models by load: {', '.join(top_5)}")
        total_models = self.bridge_data.predefined_models.copy()
        new_dynamic_models = []
        running_models = self.get_running_models()
        # Sometimes a dynamic model is wwaiting in the queue,
        # and we do not wan to unload it
        # However we also don't want to keep it loaded
        # + the full amount of dynamic models
        # as we may run out of RAM/VRAM.
        # So we reduce the amount of dynamic models
        # based on how many previous dynamic models we need to keep loaded
        needed_previous_dynamic_models = sum(
            model_name not in self.bridge_data.predefined_models for model_name in running_models
        )
        for model in models_data:
            if model["name"] in self.bridge_data.models_to_skip:
                continue
            if model["name"] in total_models:
                continue
            if len(new_dynamic_models) + needed_previous_dynamic_models >= self.bridge_data.number_of_dynamic_models:
                break
            # If we've limited the amount of models to download,
            # then we skip models which are not already downloaded
            if (
                self.model_manager.count_available_models_by_types() >= self.bridge_data.max_models_to_download
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

    def reload_data(self):
        """This is just a utility function to reload the configuration"""
        super().reload_data()
        self.bridge_data.check_models(self.model_manager)
        self.bridge_data.reload_models(self.model_manager)

    def reload_bridge_data(self):
        super().reload_bridge_data()
        if self.bridge_data.dynamic_models:
            try:
                self.calculate_dynamic_models()
            # pylint: disable=broad-except
            except Exception as err:
                logger.warning("Failed to get models_req to do dynamic model loading: {}", err)
                trace = "".join(traceback.format_exception(type(err), err, err.__traceback__))
                logger.trace(trace)
