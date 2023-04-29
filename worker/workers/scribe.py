"""This is the scribe worker, it's the main workhorse that deals with getting requests, and spawning data processing"""
import traceback

import requests

from worker.jobs.poppers import ScribePopper
from worker.jobs.scribe import ScribeHordeJob
from worker.logger import logger
from worker.workers.framework import WorkerFramework


class ScribeWorker(WorkerFramework):
    def __init__(self, this_bridge_data):
        super().__init__(None, this_bridge_data)
        self.PopperClass = ScribePopper
        self.JobClass = ScribeHordeJob

    def can_process_jobs(self):
        return self.bridge_data.kai_available

    # We want this to be extendable as well
    def add_job_to_queue(self):
        job = super().add_job_to_queue()

    def pop_job(self):
        return super().pop_job()

    def get_running_models(self):
        running_job_models = [job.current_model for job_thread, start_time, job in self.running_jobs]
        queued_jobs_models = [job.current_model for job in self.waiting_jobs]
        return list(set(running_job_models + queued_jobs_models))