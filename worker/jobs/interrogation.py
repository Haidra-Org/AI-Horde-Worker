"""Get and process a job from the horde"""
import time
import traceback

import numpy as np
from nataili.blip.caption import Caption
from nataili.clip.interrogate import Interrogator
from nataili.util.logger import logger
from transformers import CLIPFeatureExtractor

from worker.enums import JobStatus
from worker.jobs.framework import HordeJobFramework


class InterrogationHordeJob(HordeJobFramework):
    """Get and process an image interrogation job from the horde"""

    def __init__(self, mm, bd, pop):
        super().__init__(mm, bd, pop)
        self.current_form = self.pop["form"]
        self.current_id = self.pop["id"]
        self.current_payload = self.pop.get("payload", {})
        self.image = self.pop["image"]
        # We allow a generation a plentiful 10 seconds per form before we consider it stale
        self.result = None

    @logger.catch(reraise=True)
    def start_job(self):
        """Starts a Stable Diffusion job from a pop request"""
        logger.debug("Starting job in threadpool for model: {}", self.current_form)
        super().start_job()
        if self.status == JobStatus.FAULTED:
            return
        self.stale_time = time.time() + 10
        interrogator = None
        payload_kwargs = {}
        logger.debug(f"Starting interrogation {self.current_id}")
        if self.current_form == "nsfw":
            safety_checker = self.model_manager.loaded_models["safety_checker"]["model"]
            feature_extractor = CLIPFeatureExtractor()
            image_features = feature_extractor(self.image, return_tensors="pt").to("cpu")
            _, has_nsfw_concept = safety_checker(
                clip_input=image_features.pixel_values, images=[np.asarray(self.image)]
            )
            self.result = has_nsfw_concept and True in has_nsfw_concept
        else:
            if self.current_form == "caption":
                interrogator = Caption(
                    self.model_manager.loaded_models["BLIP_Large"],
                )
                payload_kwargs = {
                    "sample": True,
                    "num_beams": self.current_payload.get("num_beams", 7),
                    "min_length": self.current_payload.get("min_length", 20),
                    "max_length": self.current_payload.get(
                        "max_length", self.current_payload.get("min_length", 20) + 30
                    ),
                    "top_p": self.current_payload.get("top_p", 0.9),
                    "repetition_penalty": self.current_payload.get("repetition_penalty", 1.4),
                }
            if self.current_form == "interrogation":
                interrogator = Interrogator(
                    self.model_manager.loaded_models["ViT-L/14"],
                )
                payload_kwargs = {
                    "rank": self.current_payload.get(
                        "rank", True
                    ),  # TODO: Change after payload onboards rank/similarity
                    "similarity": self.current_payload.get("similarity", False),
                    "top_count": self.current_payload.get("top_count", 5),  # TODO: Add to payload
                }
            try:
                self.result = interrogator(self.image, **payload_kwargs)
            except RuntimeError as err:
                logger.error(
                    "Something went wrong when processing request. "
                    "Please check your trace.log file for the full stack trace. "
                    f"Form: {self.current_form}. "
                    f"Payload: {payload_kwargs}."
                    f"URL: {self.pop['source_image']}."
                )
                trace = "".join(traceback.format_exception(type(err), err, err.__traceback__))
                logger.trace(trace)
                self.status = JobStatus.FAULTED
                return
        logger.info(f"Finished interrogation {self.current_id}")
        interrogator = None
        self.start_submit_thread()

    def submit_job(self, endpoint="/api/v2/interrogate/submit"):
        """Submits the job to the server to earn our kudos."""
        super().submit_job(endpoint=endpoint)

    def prepare_submit_payload(self):
        # images, seed, info, stats = txt2img(**self.current_payload)
        self.submit_dict = {"id": self.current_id}
        if self.current_form == "caption":
            self.submit_dict["result"] = {"caption": self.result}
        if self.current_form == "nsfw":
            self.submit_dict["result"] = {"nsfw": self.result}
        if self.current_form == "interrogation":
            self.submit_dict["result"] = {"interrogation": self.result}
        logger.debug(self.submit_dict)
