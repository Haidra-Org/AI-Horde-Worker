"""This is the worker, it's the main workhorse that deals with getting requests, and spawning data processing"""
from nataili.util.logger import logger

from worker.jobs.interrogation import InterrogationHordeJob
from worker.jobs.poppers import InterrogationPopper
from worker.workers.framework import WorkerFramework


class InterrogationWorker(WorkerFramework):
    def __init__(self, this_model_manager, this_bridge_data):
        super().__init__(this_model_manager, this_bridge_data)
        self.PopperClass = InterrogationPopper
        self.JobClass = InterrogationHordeJob

    # Setting it as it's own function so that it can be overriden
    def can_process_jobs(self):
        can_do = len(self.model_manager.get_loaded_models_names()) > 0
        if not can_do:
            logger.info("No models loaded. Waiting for the first model to be up before polling the horde")
        return can_do

    def pop_job(self):
        return super().pop_job()

    def reload_data(self):
        """This is just a utility function to reload the configuration"""
        super().reload_data()
        self.bridge_data.check_models(self.model_manager)
        self.bridge_data.reload_models(self.model_manager)
