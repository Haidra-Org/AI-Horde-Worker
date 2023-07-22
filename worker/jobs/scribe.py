"""Get and process a job from the horde"""
import json
import time
import traceback

import requests

from worker.enums import JobStatus
from worker.jobs.framework import HordeJobFramework
from worker.logger import logger
from worker.stats import bridge_stats


class ScribeHordeJob(HordeJobFramework):
    """Process a scribe job from the horde"""

    def __init__(self, mm, bd, pop):
        # mm will always be None for the scribe
        super().__init__(mm, bd, pop)
        self.current_model = None
        self.seed = None
        self.text = None
        self.current_model = self.bridge_data.model
        self.current_id = self.pop["id"]
        self.current_payload = self.pop["payload"]
        self.current_payload["quiet"] = True
        self.requested_softprompt = self.current_payload.get("softprompt")
        self.censored = None

    @logger.catch(reraise=True)
    def start_job(self):
        """Starts a Scribe job from a pop request"""
        logger.debug("Starting job in threadpool for model: {}", self.current_model)
        
        super().start_job()
        if self.status == JobStatus.FAULTED:
            self.start_submit_thread()
            return
        self.stale_time = time.time() + (self.current_payload.get("max_length", 80) / 2) + 10
        # These params will always exist in the payload from the horde
        gen_payload = self.current_payload
        if "width" in gen_payload or "length" in gen_payload or "steps" in gen_payload:
            logger.error(f"Stable Diffusion payload detected. Aborting. ({gen_payload})")
            self.status = JobStatus.FAULTED
            self.start_submit_thread()
            return
        try:
            logger.info(
                f"Starting generation for id {self.current_id}: {self.current_model} @ "
                f"{self.current_payload['max_length']}:{self.current_payload['max_context_length']} "
                f"Prompt length is {len(self.current_payload['prompt'])} characters",
            )
            time_state = time.time()
            if self.requested_softprompt != self.bridge_data.current_softprompt:
                logger.debug(f"Requested softprompt: {self.requested_softprompt}")
                # requests.put(
                #     self.bridge_data.kai_url + "/api/latest/config/soft_prompt/",
                #     json={"value": self.requested_softprompt},
                # )
                time.sleep(1)  # Wait a second to unload the softprompt
            loop_retry = 0
            gen_success = False
            while not gen_success and loop_retry < 5:
                try:
                    gen_req = {
                        "json": {
                            "results": []
                        },
                        "status_code": 200,
                    }

                    for _ in range(self.current_payload['n']):
                        req =  requests.post(
                            self.bridge_data.kai_url + "/api/v1/generate",
                            json={
                                "prompt": self.current_payload["prompt"],
                                "max_new_tokens": self.current_payload["max_length"],
                                "temperature": self.current_payload["temperature"] if "temperature" in self.current_payload else 0.7,
                                "top_p": self.current_payload["top_p"] if "top_p" in self.current_payload else 0.9,
                                "top_k": self.current_payload["top_k"] if "top_k" in self.current_payload else 20,
                                "top_a": self.current_payload["top_a"] if "top_a" in self.current_payload else 0,
                                "repetition_penalty": self.current_payload["rep_pen"] if "rep_pen" in self.current_payload else 0,
                                "repetition_penalty_range": self.current_payload["rep_pen_range"] if "rep_pen_range" in self.current_payload else 0,
                                "penalty_alpha": self.current_payload["rep_pen_slope"] if "rep_pen_slope" in self.current_payload else 0,
                                "typical_p": self.current_payload["typical"] if "typical" in self.current_payload else 1,
                                "tfs": self.current_payload["tfs"] if "tfs" in self.current_payload else 1,
                            },
                            timeout=300,
                        )
                        if req.status_code != 200:
                            gen_req["status_code"] = req.status_code
                            break

                        gen_req["json"]["results"].append(req.json()["results"][0])
                except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
                    logger.error(f"Worker {self.bridge_data.kai_url} unavailable. Retrying in 3 seconds...")
                    loop_retry += 1
                    time.sleep(3)
                    continue
                if type(gen_req["json"]) is not dict:
                    logger.error(
                        (
                            f"KAI instance {self.bridge_data.kai_url} API unexpected response on generate: {gen_req}. "
                            "Retrying in 3 seconds..."
                        ),
                    )
                    time.sleep(3)
                    loop_retry += 1
                    continue
                if gen_req["status_code"] == 503:
                    logger.debug(
                        f"KAI instance {self.bridge_data.kai_url} Busy (attempt {loop_retry}). Will try again...",
                    )
                    time.sleep(3)
                    loop_retry += 1
                    continue
                if gen_req["status_code"] == 422:
                    logger.error(
                        f"KAI instance {self.bridge_data.kai_url} reported validation error.",
                    )
                    self.status = JobStatus.FAULTED
                    self.start_submit_thread()
                    return
                try:
                    req_json = gen_req["json"]
                except json.decoder.JSONDecodeError:
                    logger.error(
                        (
                            f"Something went wrong when trying to generate on {self.bridge_data.kai_url}. "
                            "Please check the health of the KAI worker. Retrying 3 seconds...",
                        ),
                    )
                    loop_retry += 1
                    time.sleep(3)
                    continue
                try:
                    self.text = req_json["results"][0]["text"]
                except KeyError:
                    logger.error(
                        (
                            f"Unexpected response received from {self.bridge_data.kai_url}: {req_json}. "
                            "Please check the health of the KAI worker. Retrying in 3 seconds..."
                        ),
                    )
                    logger.debug(self.current_payload)
                    loop_retry += 1
                    time.sleep(3)
                    continue
                gen_success = True
            self.seed = 0
            logger.info(
                f"Generation for id {self.current_id} finished successfully"
                f" in {round(time.time() - time_state,1)} seconds.",
            )
        except Exception as err:
            stack_payload = gen_payload
            stack_payload["request_type"] = "text2text"
            stack_payload["model"] = self.current_model
            stack_payload["prompt"] = "PROMPT REDACTED"
            logger.error(
                "Something went wrong when processing request. "
                "Please check your trace.log file for the full stack trace. "
                f"Payload: {stack_payload}",
            )
            trace = "".join(traceback.format_exception(type(err), err, err.__traceback__))
            logger.trace(trace)
            self.status = JobStatus.FAULTED
            self.start_submit_thread()
            return
        self.start_submit_thread()

    def submit_job(self, endpoint="/api/v2/generate/text/submit"):
        """Submits the job to the server to earn our kudos."""
        super().submit_job(endpoint=endpoint)

    def prepare_submit_payload(self):
        self.submit_dict = {
            "id": self.current_id,
            "generation": self.text,
            "seed": self.seed,
        }
        if self.censored:
            self.submit_dict["state"] = self.censored

    def post_submit_tasks(self, submit_req):
        bridge_stats.update_inference_stats(self.current_model, submit_req.json()["reward"])
