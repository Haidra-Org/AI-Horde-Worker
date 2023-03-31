"""This is the worker, it's the main workhorse that deals with getting requests, and spawning data processing"""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from nataili.util.logger import logger

from worker.stats import bridge_stats
from worker.ui import TerminalUI


class WorkerFramework:
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
        self.ui = None
        self.last_stats_time = time.time()
        self.reload_data()
        logger.stats("Starting new stats session")
        # These two should be filled in by the extending classes
        self.PopperClass = None
        self.JobClass = None

    @logger.catch(reraise=True)
    def start(self):
        self.exit_rc = 1

        # Setup UI if requested
        if self.bridge_data.enable_terminal_ui:
            # Don't allow this if auto-downloading is not enabled as how will the user see download prompts?
            if not self.bridge_data.always_download:
                logger.warning("Terminal UI can not be enabled without also enabling 'always_download'")
            else:
                ui = TerminalUI(self.bridge_data.worker_name, self.bridge_data.api_key, self.bridge_data.horde_url)
                self.ui = threading.Thread(target=ui.run, daemon=True)
                self.ui.start()

        while True:  # This is just to allow it to loop through this and handle shutdowns correctly
            self.should_restart = False
            self.consecutive_failed_jobs = 0
            with ThreadPoolExecutor(max_workers=self.bridge_data.max_threads) as self.executor:
                while not self.should_stop:
                    if self.should_restart:
                        self.executor.shutdown(wait=False)
                        break
                    try:
                        if self.ui and not self.ui.is_alive():
                            # UI Exited, we should probably exit
                            raise KeyboardInterrupt
                        self.process_jobs()
                    except KeyboardInterrupt:
                        self.should_stop = True
                        self.exit_rc = 0
                        break
                if self.should_stop:
                    logger.init("Worker", status="Shutting Down")
                    sys.exit(self.exit_rc)

    def process_jobs(self):
        if time.time() - self.last_config_reload > 60:
            self.reload_bridge_data()
        if not self.can_process_jobs():
            time.sleep(5)
            return
        # Add job to queue if we have space
        if len(self.waiting_jobs) < self.bridge_data.queue_size:
            self.add_job_to_queue()
        # Start new jobs
        while len(self.running_jobs) < self.bridge_data.max_threads and self.start_job():
            pass
        # Check if any jobs are done
        for job_thread, start_time, job in self.running_jobs:
            self.check_running_job_status(job_thread, start_time, job)
            if self.should_restart or self.should_stop:
                break
        # Give the CPU a break
        time.sleep(0.02)

    def can_process_jobs(self):
        """This function returns true when this worker can start polling for jobs from the AI Horde
        This function MUST be overriden, according to the logic for this worker type"""
        return False

    def add_job_to_queue(self):
        """Picks up a job from the horde and adds it to the local queue
        Returns the job object created, if any"""
        if jobs := self.pop_job():
            self.waiting_jobs.extend(jobs)

    def pop_job(self):
        """Polls the AI Horde for new jobs and creates as many Job classes needed
        As the amount of jobs returned"""
        job_popper = self.PopperClass(self.model_manager, self.bridge_data)
        pops = job_popper.horde_pop()
        if not pops:
            return None
        new_jobs = []
        for pop in pops:
            new_job = self.JobClass(self.model_manager, self.bridge_data, pop)
            new_jobs.append(new_job)
        return new_jobs

    def start_job(self):
        """Starts a job previously picked up from the horde
        Returns True to continue starting jobs until queue is full
        Returns False to break out of the loop and poll the horde again"""
        job = None
        # Queue disabled
        if self.bridge_data.queue_size == 0:
            if jobs := self.pop_job():
                job = jobs[0]
        elif len(self.waiting_jobs) > 0:
            job = self.waiting_jobs.pop(0)
        else:
            #  This causes a break on the main loop outside
            return False
        # Run the job
        if job:
            self.running_jobs.append((self.executor.submit(job.start_job), time.monotonic(), job))
            logger.debug("New job processing")
        else:
            logger.debug("No new job to start")
        return True

    def check_running_job_status(self, job_thread, start_time, job):
        """Polls the AI Horde for new jobs and creates a Job class"""
        runtime = time.monotonic() - start_time
        if job_thread.done():
            if job_thread.exception(timeout=1) or job.is_faulted():
                if job_thread.exception(timeout=1):
                    logger.error("Job failed with exception, {}", job_thread.exception())
                    logger.exception(job_thread.exception())
                if self.consecutive_executor_restarts > 0:
                    logger.critical(
                        "Worker keeps crashing after thread executor restart. " "Cannot be salvaged. Aborting!",
                    )
                    self.should_stop = True
                    return
                self.consecutive_failed_jobs += 1
                if self.consecutive_failed_jobs >= 5:
                    logger.critical(
                        "Too many consecutive jobs have failed. " "Restarting thread executor and hope we recover...",
                    )
                    self.should_restart = True
                    self.consecutive_executor_restarts += 1
                    return
            else:
                self.consecutive_failed_jobs = 0
                self.consecutive_executor_restarts = 0
            self.run_count += 1
            logger.debug(f"Job finished successfully in {runtime:.3f}s (Total Completed: {self.run_count})")
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

        # Check periodically if any interesting stats should be announced
        if self.bridge_data.stats_output_frequency and time.time() - self.last_stats_time > min(
            self.bridge_data.stats_output_frequency,
            30,
        ):
            self.last_stats_time = time.time()
            logger.info(f"Estimated average kudos per hour: {bridge_stats.stats.get('kudos_per_hour', 0)}")

    def reload_data(self):
        """This is just a utility function to reload the configuration"""
        self.bridge_data.reload_data()

    def reload_bridge_data(self):
        self.reload_data()
        self.executor._max_workers = self.bridge_data.max_threads
        self.last_config_reload = time.time()
