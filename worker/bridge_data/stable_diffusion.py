"""The configuration of the bridge"""
import os
import re
import time

import requests
from nataili.util.logger import logger
from PIL import Image

from worker.argparser.stable_diffusion import args
from worker.bridge_data.framework import BridgeDataTemplate


class StableDiffusionBridgeData(BridgeDataTemplate):
    """Configuration object"""

    POSTPROCESSORS = ["GFPGAN", "RealESRGAN_x4plus", "CodeFormers"]
    INTERROGATORS = ["ViT-L/14"]

    def __init__(self):
        super().__init__(args)
        self._last_top_n_refresh = 0
        self._last_model_db_refresh = 0
        self._all_model_names = []
        self._top_n_model_names = []
        self.max_power = int(os.environ.get("HORDE_MAX_POWER", 8))
        self.nsfw = os.environ.get("HORDE_NSFW", "true") == "true"
        self.censor_nsfw = os.environ.get("HORDE_CENSOR", "false") == "true"
        self.blacklist = list(filter(lambda a: a, os.environ.get("HORDE_BLACKLIST", "").split(",")))
        self.censorlist = list(filter(lambda a: a, os.environ.get("HORDE_CENSORLIST", "").split(",")))
        self.allow_img2img = os.environ.get("HORDE_IMG2IMG", "true") == "true"
        self.allow_painting = os.environ.get("HORDE_PAINTING", "true") == "true"
        self.allow_post_processing = os.environ.get("ALLOW_POST_PROCESSING", "true") == "true"
        self.allow_controlnet = os.environ.get("ALLOW_CONTROLNET", "false") == "false"
        self.model_names = os.environ.get("HORDE_MODELNAMES", "stable_diffusion").split(",")
        self.max_pixels = 64 * 64 * 8 * self.max_power
        self.censor_image_sfw_worker = Image.open("assets/nsfw_censor_sfw_worker.png")
        self.censor_image_censorlist = Image.open("assets/nsfw_censor_censorlist.png")
        self.censor_image_sfw_request = Image.open("assets/nsfw_censor_sfw_request.png")
        self.censor_image_csam = Image.open("assets/nsfw_censor_csam.png")
        self.models_reloading = False
        self.model = None
        self.dynamic_models = True
        self.number_of_dynamic_models = 3
        self.models_to_skip = os.environ.get("HORDE_SKIPPED_MODELNAMES", "stable_diffusion_inpainting").split(",")
        self.predefined_models = self.model_names.copy()
        self.top_n_refresh_frequency = os.environ.get("HORDE_TOP_N_REFRESH", 60 * 60 * 24)
        self.model_database_refresh_frequency = os.environ.get("HORDE_MODEL_DB_REFRESH", 0)

    @logger.catch(reraise=True)
    def reload_data(self):
        """Reloads configuration data"""
        previous_url = self.horde_url
        super().reload_data()

        if not hasattr(self, "models_to_load"):
            self.models_to_load = []

        # Check for magic constants and expand them
        top_n = 0
        for model in self.models_to_load[:]:
            # all models
            match = re.match(r"ALL ((\w*) )?MODELS", model, re.IGNORECASE)
            if match:
                self.models_to_load = self.get_all_models(match.group(2))
                break  # can't be more
            # top n
            match = re.match(r"TOP (\d+)", model, re.IGNORECASE)
            if match:
                self.models_to_load.remove(model)
                if int(match.group(1)) > 0:
                    top_n = int(match.group(1))
        if top_n:
            self.models_to_load.extend(self.get_top_n_models(top_n))

        if not self.dynamic_models:
            self.model_names = self.models_to_load
        else:
            self.predefined_models = self.models_to_load

        if args.max_power:
            self.max_power = args.max_power
        if args.model:
            self.model = [args.model]
        if args.sfw:
            self.nsfw = False
        if args.censor_nsfw:
            self.censor_nsfw = args.censor_nsfw
        if args.blacklist:
            self.blacklist = args.blacklist
        if args.censorlist:
            self.censorlist = args.censorlist
        if args.allow_img2img:
            self.allow_img2img = args.allow_img2img
        if args.allow_painting:
            self.allow_painting = args.allow_painting
        if args.disable_dynamic_models:
            self.dynamic_models = False
        if args.disable_post_processing:
            self.allow_post_processing = False
        if args.disable_controlnet:
            self.allow_controlnet = False
        self.max_power = max(self.max_power, 2)
        self.max_pixels = 64 * 64 * 8 * self.max_power
        # if self.censor_nsfw or (self.censorlist is not None and len(self.censorlist)):
        self.model_names.append("safety_checker")
        self.model_names.insert(0, "ViT-L/14")
        if self.allow_post_processing:
            self.model_names += self.POSTPROCESSORS
        if (not self.initialized and not self.models_reloading) or previous_url != self.horde_url:
            logger.init(
                (
                    f"Username '{self.username}'. Server Name '{self.worker_name}'. "
                    f"Horde URL '{self.horde_url}'. Max Pixels {self.max_pixels}. "
                    "Worker Type: Stable Diffusion"
                ),
                status="Joining Horde",
            )

    def check_extra_conditions_for_download_choice(self):
        return self.dynamic_models

    def _is_valid_stable_diffusion_model(self, model_name):
        if model_name in ["safety_checker", "LDSR"]:
            return False

        if model_name in self.POSTPROCESSORS or model_name in self.INTERROGATORS:
            return False

        if model_name in self.models_to_skip:
            return False

        return True

    # Get all models directly from the server, not from nataili, as nataili
    # may not be loaded, e.g. in webui.
    def get_all_models(self, style=""):

        # Recognise some magic style constants
        nsfw = None
        if style and style.upper() == "SFW":
            style = ""
            nsfw = False
        elif style and style.upper() == "NSFW":
            style = ""
            nsfw = True

        # Never refresh more than once per hour
        if self.model_database_refresh_frequency < 3600:
            self.model_database_refresh_frequency = 3600

        # Should we refresh the model list?
        if (
            self._last_model_db_refresh
            and time.monotonic() - self._last_model_db_refresh < self.model_database_refresh_frequency
        ):
            # No, return cached version
            return self._all_model_names[:]

        # If we're never refreshing and have a cache, just use that
        if not self.model_database_refresh_frequency and self._all_model_names:
            return self._all_model_names[:]

        logger.info("Refreshing the list of all available models")
        data = requests.get(
            "https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/stable_diffusion.json"
        ).json()

        # Get all interesting models
        models = []
        for _, model in data.items():
            if not self._is_valid_stable_diffusion_model(model["name"]):
                continue  # ignore
            if model["type"] != "ckpt":
                continue  # ignore
            if style and "style" in model and style.lower() not in model["style"].lower():
                continue  # ignore
            if nsfw is False and model["nsfw"]:
                continue  # ignore
            if nsfw is True and not model["nsfw"]:
                continue  # ignore
            # Add to the list of models to load
            models.append(model["name"])

        models = sorted(list(set(models)))

        # Move the standard SD model to the top of the list
        if "stable_diffusion" in models:
            models.remove("stable_diffusion")
            models.insert(0, "stable_diffusion")

        # Always have usable model something, no matter what went wrong
        if not models:
            models.append("stable_diffusion")

        # Add inpainting model if inpainting is enabled
        if self.allow_painting:
            models.append("stable_diffusion_inpainting")

        # Save our models
        self._all_model_names = models[:]
        self._last_model_db_refresh = time.monotonic()

        return models

    # Get the top n most popular models from the horde server
    def get_top_n_models(self, top_n, period="day"):

        model_list = []

        # Never refresh more than once per hour
        if self.top_n_refresh_frequency < 3600:
            self.top_n_refresh_frequency = 3600

        # Should we refresh the top n list?
        if self._last_top_n_refresh and time.monotonic() - self._last_top_n_refresh < self.top_n_refresh_frequency:
            # No, use cached data
            model_list = self._top_n_model_names

        # If we're never refreshing and have a cache, just use that
        if not self.top_n_refresh_frequency and self._top_n_model_names:
            model_list = self._top_n_model_names

        # Update the top n model chart
        if not model_list:
            models = {}
            logger.info("Refreshing the most popular model data")
            try:
                req = requests.get(f"{self.horde_url}/api/v2/stats/img/models")
                models = req.json()[period] if req.ok else {}
            except requests.exceptions.RequestException:
                logger.warning("Failed to retrieve the most popular models data.")
            model_list = sorted(((models[model], model) for model in models), reverse=True)
            self._top_n_model_names = model_list
            self._last_top_n_refresh = time.monotonic()

        top = [x[1] for x in self._top_n_model_names[:top_n]]

        # Always return something, no matter what went wrong
        if not top:
            top.append("stable_diffusion")
        return top
