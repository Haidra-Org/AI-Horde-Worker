import copy
import json
import time
from io import BytesIO

import requests
from PIL import Image, UnidentifiedImageError

from nataili.util import logger


class JobPopper:

    retry_interval = 1
    BRIDGE_VERSION = 10

    def __init__(self, mm, bd):
        self.model_manager = mm
        self.bridge_data = copy.deepcopy(bd)
        self.pop = None
        self.headers = {"apikey": self.bridge_data.api_key}
        # This should be set by the extending class
        self.endpoint = None

    def horde_pop(self):
        """Get a job from the horde"""
        try:
            pop_req = requests.post(
                self.bridge_data.horde_url + self.endpoint,
                json=self.pop_payload,
                headers=self.headers,
                timeout=20,
            )
            logger.debug(f"Job pop took {pop_req.elapsed.total_seconds()}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Server {self.bridge_data.horde_url} unavailable during pop. Waiting 10 seconds...")
            time.sleep(10)
            return
        except TypeError:
            logger.warning(f"Server {self.bridge_data.horde_url} unavailable during pop. Waiting 2 seconds...")
            time.sleep(2)
            return
        except requests.exceptions.ReadTimeout:
            logger.warning(f"Server {self.bridge_data.horde_url} timed out during pop. Waiting 2 seconds...")
            time.sleep(2)
            return

        try:
            self.pop = pop_req.json()  # I'll use it properly later
        except json.decoder.JSONDecodeError:
            logger.error(
                f"Could not decode response from {self.bridge_data.horde_url} as json. "
                "Please inform its administrator!"
            )
            time.sleep(2)
            return
        if not pop_req.ok:
            logger.warning(
                f"During gen pop, server {self.bridge_data.horde_url} "
                f"responded with status code {pop_req.status_code}: "
                f"{self.pop['message']}. Waiting for 10 seconds..."
            )
            if "errors" in self.pop:
                logger.warning(f"Detailed Request Errors: {self.pop['errors']}")
            time.sleep(10)
            return
        return [self.pop]

    def report_skipped_info(self):
        job_skipped_info = self.pop.get("skipped")
        if job_skipped_info and len(job_skipped_info):
            self.skipped_info = f" Skipped Info: {job_skipped_info}."
        else:
            self.skipped_info = ""
        logger.info(f"Server {self.bridge_data.horde_url} has no valid generations for us to do.{self.skipped_info}")
        time.sleep(self.retry_interval)


class StableDiffusionPopper(JobPopper):
    def __init__(self, mm, bd):
        super().__init__(mm, bd)
        self.endpoint = "/api/v2/generate/pop"
        self.available_models = self.model_manager.get_loaded_models_names()
        for util_model in ["LDSR", "safety_checker", "GFPGAN", "RealESRGAN_x4plus", "CodeFormers"]:
            if util_model in self.available_models:
                self.available_models.remove(util_model)
        self.pop_payload = {
            "name": self.bridge_data.worker_name,
            "max_pixels": self.bridge_data.max_pixels,
            "priority_usernames": self.bridge_data.priority_usernames,
            "nsfw": self.bridge_data.nsfw,
            "blacklist": self.bridge_data.blacklist,
            "models": self.available_models,
            "allow_img2img": self.bridge_data.allow_img2img,
            "allow_painting": self.bridge_data.allow_painting,
            "allow_unsafe_ip": self.bridge_data.allow_unsafe_ip,
            "threads": self.bridge_data.max_threads,
            "allow_post_processing": self.bridge_data.allow_post_processing,
            "require_upfront_kudos": self.bridge_data.require_upfront_kudos,
            "bridge_version": self.BRIDGE_VERSION,
        }

    def horde_pop(self):

        if not super().horde_pop():
            return
        if not self.pop.get("id"):
            self.report_skipped_info()
            return
        # In the stable diffusion popper, the whole return is always a single payload, so we return it as a list
        return [self.pop]


class InterrogationPopper(JobPopper):
    def __init__(self, mm, bd):
        super().__init__(mm, bd)
        self.endpoint = "/api/v2/interrogate/pop"
        available_forms = []
        self.available_models = self.model_manager.get_loaded_models_names()
        for util_model in ["LDSR", "GFPGAN", "RealESRGAN_x4plus", "CodeFormers"]:
            if util_model in self.available_models:
                self.available_models.remove(util_model)
        if "BLIP_Large" in self.available_models:
            available_forms.append("caption")
        if "safety_checker" in self.available_models:
            available_forms.append("nsfw")
        if "ViT-L/14" in self.available_models:
            available_forms.append("interrogation")
        self.pop_payload = {
            "name": self.bridge_data.worker_name,
            "forms": available_forms,
            "amount": self.bridge_data.queue_size,
            "priority_usernames": self.bridge_data.priority_usernames,
            "threads": self.bridge_data.max_threads,
            "bridge_version": self.BRIDGE_VERSION,
        }

    def horde_pop(self):
        if not super().horde_pop():
            return
        if not self.pop.get("forms"):
            self.report_skipped_info()
            return
        # In the interrogation popper, the forms key contains an array of payloads to execute
        current_image_url = None
        for form in self.pop["forms"]:
            if form["source_image"] != current_image_url:
                current_image_url = form["source_image"]
                img_data = requests.get(current_image_url).content
            try:
                form["image"] = Image.open(BytesIO(img_data)).convert("RGB")
            except UnidentifiedImageError as e:
                logger.error(f"Error when creating image: {e}. Url {current_image_url}, img_data: {img_data}")
        logger.debug(f"Popped {len(self.pop['forms'])} interrogation forms")
        return self.pop["forms"]
