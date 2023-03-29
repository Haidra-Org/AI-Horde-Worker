"""Get and process a job from the horde"""
import time
import traceback
from io import BytesIO

import numpy as np
import rembg
import requests
from nataili.blip.caption import Caption
from nataili.clip.interrogate import Interrogator
from nataili.util.logger import logger
from transformers import CLIPFeatureExtractor

from worker.consts import KNOWN_POST_PROCESSORS, KNOWN_UPSCALERS
from worker.enums import JobStatus
from worker.jobs.framework import HordeJobFramework
from worker.post_process import post_process


class InterrogationHordeJob(HordeJobFramework):
    """Get and process an image interrogation job from the horde"""

    def __init__(self, mm, bd, pop):
        super().__init__(mm, bd, pop)
        self.current_form = self.pop["form"]
        self.current_id = self.pop["id"]
        self.current_payload = self.pop.get("payload", {})
        self.image = self.pop["image"]
        self.r2_upload = self.pop.get("r2_upload", False)
        # We allow a generation a plentiful 10 seconds per form before we consider it stale
        self.result = None

    @logger.catch(reraise=True)
    def start_job(self):
        """Starts a Stable Diffusion job from a pop request"""
        logger.debug("Starting job in threadpool for model: {}", self.current_form)
        super().start_job()
        if self.status == JobStatus.FAULTED:
            return
        stale_buffer = 10  # seconds
        if self.current_form in KNOWN_UPSCALERS:
            stale_buffer = self.calculate_upscale_chunks() * 5
        self.stale_time = time.time() + stale_buffer + 5
        interrogator = None
        payload_kwargs = {}
        logger.debug(f"Starting interrogation {self.current_id}")
        if self.current_form == "nsfw":
            safety_checker = self.model_manager.loaded_models["safety_checker"]["model"]
            feature_extractor = CLIPFeatureExtractor()
            image_features = feature_extractor(self.image, return_tensors="pt").to("cpu")
            _, has_nsfw_concept = safety_checker(
                clip_input=image_features.pixel_values,
                images=[np.asarray(self.image)],
            )
            self.result = has_nsfw_concept and True in has_nsfw_concept
        elif self.current_form in KNOWN_POST_PROCESSORS:
            try:
                if self.current_form == "strip_background":
                    session = rembg.new_session("u2net")
                    self.image = rembg.remove(
                        self.image,
                        session=session,
                        only_mask=False,
                        alpha_matting=10,
                        alpha_matting_foreground_threshold=240,
                        alpha_matting_background_threshold=10,
                        alpha_matting_erode_size=10,
                    )
                    del session
                else:
                    strength = self.current_payload.get("facefixer_strength", 0.5)
                    self.image = post_process(self.current_form, self.image, self.model_manager, strength=strength)
                self.result = "R2"
            except (AssertionError, RuntimeError) as err:
                logger.error(
                    "Post-Processor form '{}' encountered an error when working on image . Skipping! {}",
                    self.current_form,
                    err,
                )
                self.status = JobStatus.FAULTED
                self.start_submit_thread()
                return
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
                        "max_length",
                        self.current_payload.get("min_length", 20) + 30,
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
                        "rank",
                        True,
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
                    f"URL: {self.pop['source_image']}.",
                )
                trace = "".join(traceback.format_exception(type(err), err, err.__traceback__))
                logger.trace(trace)
                self.status = JobStatus.FAULTED
                self.start_submit_thread()
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
        elif self.current_form == "interrogation":
            self.submit_dict["result"] = {"interrogation": self.result}
        elif self.current_form == "nsfw":
            self.submit_dict["result"] = {"nsfw": self.result}
        elif self.current_form in KNOWN_POST_PROCESSORS:
            logger.debug(self.r2_upload)
            buffer = BytesIO()
            # We send as WebP to avoid using all the horde bandwidth
            self.image.save(buffer, format="WebP", quality=95)
            if self.r2_upload:
                put_response = requests.put(self.r2_upload, data=buffer.getvalue())
                logger.debug("R2 Upload response: {}", put_response)
            self.submit_dict["result"] = {self.current_form: self.result}
        logger.debug([self.current_form in KNOWN_POST_PROCESSORS, self.current_form, KNOWN_POST_PROCESSORS])
        logger.debug(self.submit_dict)

    def calculate_upscale_chunks(self):
        width, height = self.image.size

        tiles_x = (width + 511) // 512
        tiles_y = (height + 511) // 512
        return tiles_x * tiles_y
