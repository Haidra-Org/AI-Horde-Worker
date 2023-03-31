import base64
import copy
import json
import time
from io import BytesIO

import requests
from nataili.util.logger import logger
from PIL import Image, UnidentifiedImageError

from worker.consts import BRIDGE_VERSION, KNOWN_INTERROGATORS, KNOWN_POST_PROCESSORS, POST_PROCESSORS_NATAILI_MODELS


class JobPopper:
    retry_interval = 1
    BRIDGE_AGENT = f"AI Horde Worker:{BRIDGE_VERSION}:https://github.com/db0/AI-Horde-Worker"

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
            # logger.debug(self.headers)
            # logger.debug(self.pop_payload)
            pop_req = requests.post(
                self.bridge_data.horde_url + self.endpoint,
                json=self.pop_payload,
                headers=self.headers,
                timeout=40,
            )
            # logger.debug(self.pop_payload)
            node = pop_req.headers["horde-node"] if "horde-node" in pop_req.headers else "unknown"
            logger.debug(f"Job pop took {pop_req.elapsed.total_seconds()} (node: {node})")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Server {self.bridge_data.horde_url} unavailable during pop. Waiting 10 seconds...")
            time.sleep(10)
            return None
        except TypeError:
            logger.warning(f"Server {self.bridge_data.horde_url} unavailable during pop. Waiting 2 seconds...")
            time.sleep(2)
            return None
        except requests.exceptions.ReadTimeout:
            logger.warning(f"Server {self.bridge_data.horde_url} timed out during pop. Waiting 2 seconds...")
            time.sleep(2)
            return None
        except requests.exceptions.InvalidHeader:
            logger.warning(
                f"Server {self.bridge_data.horde_url} Something is wrong with the API key you are sending. "
                "Please check your bridgeData api_key variable. Waiting 10 seconds...",
            )
            time.sleep(10)
            return None

        try:
            self.pop = pop_req.json()  # I'll use it properly later
        except json.decoder.JSONDecodeError:
            logger.error(
                f"Could not decode response from {self.bridge_data.horde_url} as json. "
                "Please inform its administrator!",
            )
            time.sleep(2)
            return None
        if not pop_req.ok:
            logger.warning(f"{self.pop['message']} ({pop_req.status_code})")
            if "errors" in self.pop:
                logger.warning(f"Detailed Request Errors: {self.pop['errors']}")
            time.sleep(2)
            return None
        return [self.pop]

    def report_skipped_info(self):
        job_skipped_info = self.pop.get("skipped")
        if job_skipped_info and len(job_skipped_info):
            self.skipped_info = f" Skipped Info: {job_skipped_info}."
        else:
            self.skipped_info = ""
        logger.info(f"Server {self.bridge_data.horde_url} has no valid generations for us to do.{self.skipped_info}")
        time.sleep(self.retry_interval)

    def download_image_data(self, image_url):
        """Returns the image data, not a PIL"""
        try:
            with requests.get(image_url, stream=True, timeout=2) as r:
                size = r.headers.get("Content-Length", 0)
                if int(size) > 5120000:
                    logger.error(f"Provided image ({image_url}) cannot be larger than 5Mb")
                    return None
                mbs = 0
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        if mbs == 0:
                            img_data = chunk
                        else:
                            img_data += chunk
                        mbs += 1
                        if mbs > 5:
                            logger.error(f"Provided image ({image_url}) cannot be larger than 5Mb")
                            return None
        except Exception as err:
            logger.error(err)
            return None
        return img_data

    def convert_image_data_to_pil(self, img_data):
        try:
            return Image.open(BytesIO(img_data)).convert("RGB")
        except UnidentifiedImageError as e:
            logger.error(f"Error when creating image: {e}.")
            return None
        except UnboundLocalError as e:
            logger.error(f"Error when creating image: {e}.")
            return None


class StableDiffusionPopper(JobPopper):
    def __init__(self, mm, bd):
        super().__init__(mm, bd)
        self.endpoint = "/api/v2/generate/pop"
        self.available_models = self.model_manager.get_loaded_models_names()
        for util_model in (
            list(KNOWN_INTERROGATORS)
            + list(POST_PROCESSORS_NATAILI_MODELS)
            + [
                "LDSR",
                "safety_checker",
            ]
        ):
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
            "allow_controlnet": self.bridge_data.allow_controlnet,
            "require_upfront_kudos": self.bridge_data.require_upfront_kudos,
            "bridge_version": BRIDGE_VERSION,
            "bridge_agent": self.BRIDGE_AGENT,
        }

    def horde_pop(self):
        if not super().horde_pop():
            return None
        if not self.pop.get("id"):
            self.report_skipped_info()
            return None
        # In the stable diffusion popper, the whole return is always a single payload, so we return it as a list
        self.pop["source_image"] = self.download_source(self.pop.get("source_image"))
        self.pop["source_mask"] = self.download_source(self.pop.get("source_mask"))
        return [self.pop]

    def download_source(self, source_img):
        if not source_img:
            return None
        if "https" in source_img:
            if img_data := self.download_image_data(source_img):
                img = self.convert_image_data_to_pil(img_data)
                if not img:
                    logger.error("Non-image data when downloading image! Ignoring")
                return img

            logger.warning(f"Could not download source image from R2 {source_img}. Skipping source image.")
            return None

        base64_bytes = source_img.encode("utf-8")
        img_bytes = base64.b64decode(base64_bytes)
        return Image.open(BytesIO(img_bytes))


class InterrogationPopper(JobPopper):
    def __init__(self, mm, bd):
        super().__init__(mm, bd)
        self.endpoint = "/api/v2/interrogate/pop"
        available_forms = []
        self.available_models = self.model_manager.get_loaded_models_names()
        if "BLIP_Large" in self.available_models:
            available_forms.append("caption")
        if "safety_checker" in self.available_models:
            available_forms.append("nsfw")
        if "ViT-L/14" in self.available_models:
            available_forms.append("interrogation")
        if "post-process" in bd.forms:
            available_forms.extend(list(KNOWN_POST_PROCESSORS))
        amount = max(self.bridge_data.queue_size, 1)
        self.pop_payload = {
            "name": self.bridge_data.worker_name,
            "forms": available_forms,
            "amount": amount,
            "priority_usernames": self.bridge_data.priority_usernames,
            "threads": self.bridge_data.max_threads,
            "bridge_version": BRIDGE_VERSION,
            "bridge_agent": self.BRIDGE_AGENT,
            "max_tiles": self.bridge_data.max_power,
        }
        logger.debug(self.pop_payload)

    def horde_pop(self):
        if not super().horde_pop():
            return None
        if not self.pop.get("forms"):
            self.report_skipped_info()
            return None
        # In the interrogation popper, the forms key contains an array of payloads to execute
        current_image_url = None
        non_faulted_forms = []
        for form in self.pop["forms"]:
            # TODO: Convert to use self.download_image_data and self.convert_image_data_to_pil
            if form["source_image"] != current_image_url:
                current_image_url = form["source_image"]
                try:
                    with requests.get(current_image_url, stream=True, timeout=2) as r:
                        size = r.headers.get("Content-Length", 0)
                        if int(size) > 5120000:
                            logger.error(f"Provided image ({current_image_url}) cannot be larger than 5Mb")
                            current_image_url = None
                            continue
                        mbs = 0
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                if mbs == 0:
                                    img_data = chunk
                                else:
                                    img_data += chunk
                                mbs += 1
                                if mbs > 5:
                                    logger.error(f"Provided image ({current_image_url}) cannot be larger than 5Mb")
                                    current_image_url = None
                                    continue
                except Exception as err:
                    logger.error(err)
                    current_image_url = None
                    continue
            try:
                form["image"] = Image.open(BytesIO(img_data)).convert("RGB")
                non_faulted_forms.append(form)
            except UnidentifiedImageError as e:
                logger.error(f"Error when creating image: {e}. Url {current_image_url}")
                continue
            except UnboundLocalError as e:
                logger.error(f"Error when creating image: {e}. Url {current_image_url}")
                continue
        logger.debug(f"Popped {len(non_faulted_forms)} interrogation forms")
        # TODO: Report back to the horde with faulted images
        return non_faulted_forms
