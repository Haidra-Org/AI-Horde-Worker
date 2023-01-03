"""Get and process a job from the horde"""
import time
import traceback

import requests

from PIL import Image, UnidentifiedImageError
from io import BytesIO
from nataili.blip.caption import Caption
from nataili.util import logger
from worker.enums import JobStatus
from worker.stats import bridge_stats
from worker.jobs.framework import HordeJobFramework

class InterrogationHordeJob(HordeJobFramework):
    """Get and process an image interrogation job from the horde"""

    def __init__(self, mm, bd, pop):
        super().__init__(mm, bd, pop)
        self.current_form = self.pop["form"]
        self.current_id = self.pop["id"]
        self.current_payload = self.pop["payload"]
        self.image = self.pop["image"]
        # We allow a generation a plentiful 5 seconds per form before we consider it stale
        self.stale_time = time.time() + (len(self.current_forms) * 5)
        self.result = None

    @logger.catch(reraise=True)
    def start_job(self):
        """Starts a Stable Diffusion job from a pop request"""
        logger.debug("Starting job in threadpool for model: {}", self.current_model)
        super().start_job()
        if self.status == JobStatus.FAULTED:
            return
        interrogator = None
        payload_kwargs = {}
        if self.current_form == "caption":
            interrogator = Caption(mm.loaded_models["BLIP_Large"]["model"], mm.loaded_models["BLIP_Large"]["device"])
            payload_kwargs = {
                "num_beams": self.current_payload.get("num_beams",7),
                "min_length": self.current_payload.get("min_length",10),
                "max_length": self.current_payload.get("max_length", self.current_payload.get("min_length",10) + 20),
                "top_p": self.current_payload.get("top_p", 0.9),
                "repetition_penalty": self.current_payload.get("repetition_penalty", 1.2),
            }
        try:
            logger.debug("Starting interrogation...")
            self.result = interrogator(image, **payload_kwargs)
            logger.debug("Finished interrogation...")
        except RuntimeError as err:
            logger.error(
                "Something went wrong when processing request. "
                "Please check your trace.log file for the full stack trace. "
                f"Form: {self.current_form}"
                f"Payload: {payload_kwargs}"
            )
            trace = "".join(traceback.format_exception(type(err), err, err.__traceback__))
            logger.trace(trace)
            self.status = JobStatus.FAULTED
            return
        generator = None
        self.start_submit_thread()
    

    def submit_job(self, endpoint = "/api/v2/interrogate/submit"):
        """Submits the job to the server to earn our kudos."""
        super().submit_job(endpoint = endpoint)


    def prepare_submit_payload(self):
        # images, seed, info, stats = txt2img(**self.current_payload)
        self.submit_dict = {"id": self.current_id}
        if self.current_form == "caption":
            submit_dict["result"] = {"caption": self.result}
        logger.debug(submit_payload)
