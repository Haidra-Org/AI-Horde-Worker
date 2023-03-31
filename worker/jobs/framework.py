"""Get and process a job from the horde"""
import copy
import json
import sys
import threading
import time

import requests
from nataili.util.logger import logger

from worker.enums import JobStatus


class HordeJobFramework:
    """Get and process a job from the horde"""

    retry_interval = 1

    def __init__(self, mm, bd, pop):
        self.model_manager = mm
        self.bridge_data = copy.deepcopy(bd)
        self.pop = pop
        self.loop_retry = 0
        self.status = JobStatus.INIT
        self.start_time = time.time()
        self.process_time = time.time()
        self.stale_time = None
        self.submit_dict = {}
        self.headers = {"apikey": self.bridge_data.api_key}

    def is_finished(self):
        """Check if the job is finished"""
        return self.status not in [JobStatus.WORKING, JobStatus.POLLING, JobStatus.INIT]

    def is_polling(self):
        """Check if the job is polling"""
        return self.status in [JobStatus.POLLING]

    def is_finalizing(self):
        """True if generation has finished even if upload is still remaining"""
        return self.status in [JobStatus.FINALIZING]

    def is_stale(self):
        """Check if the job is stale"""
        if time.time() - self.start_time > 1200:
            return True
        if not self.stale_time:
            return False
        # Jobs which haven't started yet are not considered stale.
        if self.status != JobStatus.WORKING:
            return False
        return time.time() > self.stale_time

    def is_faulted(self):
        """Check if the job is faulted"""
        return self.status == JobStatus.FAULTED

    @logger.catch(reraise=True)
    def start_job(self):
        """Starts a job from a pop request
        This method MUST be extended with the specific logic for this worker
        At the end it MUST create a new thread to submit the results to the horde"""
        # Pop new request from the Horde
        if self.pop is None:
            self.pop = self.get_job_from_server()

        if self.pop is None:
            logger.error(
                f"Something has gone wrong with {self.bridge_data.horde_url}. Please inform its administrator!",
            )
            time.sleep(self.retry_interval)
            self.status = JobStatus.FAULTED
            # The extended function should return as well
            return
        self.process_time = time.time()
        self.status = JobStatus.WORKING
        # Continue with the specific worker logic from here
        # At the end, you must call self.start_submit_thread()

    def start_submit_thread(self):
        """Starts a thread with submit_job so that we don't wait for the upload to complete
        # Not a daemon, so that it can survive after this class is garbage collected"""
        submit_thread = threading.Thread(target=self.submit_job, args=())
        submit_thread.start()
        logger.debug("Finished job in threadpool")

    def submit_job(self, endpoint):
        """Submits the job to the server to earn our kudos.
        This method MUST be extended with the specific logic for this worker
        At the end it MUST set the job state to DONE"""
        if self.status == JobStatus.FAULTED:
            self.submit_dict = {
                "id": self.current_id,
                "state": "faulted",
                "generation": "faulted",
                "seed": -1,
            }
        else:
            self.status = JobStatus.FINALIZING
            self.prepare_submit_payload()
        self.status = JobStatus.FINALIZING
        # Submit back to horde
        while self.is_finalizing():
            if self.loop_retry > 10:
                logger.error(f"Exceeded retry count {self.loop_retry} for job id {self.current_id}. Aborting job!")
                self.status = JobStatus.FAULTED
                break
            self.loop_retry += 1
            try:
                logger.debug(
                    f"posting payload with size of {round(sys.getsizeof(json.dumps(self.submit_dict)) / 1024,1)} kb",
                )
                submit_req = requests.post(
                    self.bridge_data.horde_url + endpoint,
                    json=self.submit_dict,
                    headers=self.headers,
                    timeout=60,
                )
                logger.debug(f"Upload completed in {submit_req.elapsed.total_seconds()}")
                try:
                    submit = submit_req.json()
                except json.decoder.JSONDecodeError:
                    logger.error(
                        f"Something has gone wrong with {self.bridge_data.horde_url} during submit. "
                        f"Please inform its administrator!  (Retry {self.loop_retry}/10)",
                    )
                    time.sleep(self.retry_interval)
                    continue
                if submit_req.status_code == 404:
                    logger.warning("The job we were working on got stale. Aborting!")
                    self.status = JobStatus.FAULTED
                    break
                if not submit_req.ok:
                    if submit_req.status_code == 400:
                        logger.warning(
                            f"During gen submit, server {self.bridge_data.horde_url} "
                            f"responded with status code {submit_req.status_code}: "
                            f"Job took {round(time.time() - self.start_time,1)} seconds since queued "
                            f"and {round(time.time() - self.process_time,1)} since start."
                            f"{submit['message']}. Aborting job!",
                        )
                        self.status = JobStatus.FAULTED
                        break
                    logger.warning(
                        f"During gen submit, server {self.bridge_data.horde_url} "
                        f"responded with status code {submit_req.status_code}: "
                        f"{submit['message']}. Waiting for 2 seconds...  (Retry {self.loop_retry}/10)",
                    )
                    if "errors" in submit:
                        logger.warning(f"Detailed Request Errors: {submit['errors']}")
                    time.sleep(2)
                    continue
                logger.info(
                    f'Submitted job with id {self.current_id} and contributed for {submit_req.json()["reward"]}. '
                    f"Job took {round(time.time() - self.start_time,1)} seconds since queued "
                    f"and {round(time.time() - self.process_time,1)} since start.",
                )
                self.post_submit_tasks(submit_req)
                self.status = JobStatus.DONE
                break
            except requests.exceptions.ConnectionError:
                logger.warning(
                    f"Server {self.bridge_data.horde_url} unavailable during submit. "
                    f"Waiting 10 seconds...  (Retry {self.loop_retry}/10)",
                )
                time.sleep(10)
                continue
            except requests.exceptions.ReadTimeout:
                logger.warning(
                    f"Server {self.bridge_data.horde_url} timed out during submit. "
                    f"Waiting 10 seconds...  (Retry {self.loop_retry}/10)",
                )
                time.sleep(10)
                continue

    def prepare_submit_payload(self):
        """Should be overriden and prepare a self.submit_dict dictionary with the payload needed
        for this job to be submitted"""
        self.submit_dict = {}

    def post_submit_tasks(self, submit_req):
        """Optional job which will execute only if the submit is successfull"""
