"""The configuration of the bridge"""
import os

from nataili.util.logger import logger
from PIL import Image

from worker.argparser.stable_diffusion import args
from worker.bridge_data.framework import BridgeDataTemplate


class StableDiffusionBridgeData(BridgeDataTemplate):
    """Configuration object"""

    POSTPROCESSORS = ["GFPGAN", "RealESRGAN_x4plus", "CodeFormers"]

    def __init__(self):
        super().__init__(args)
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
        self.models_reloading = False
        self.model = None
        self.dynamic_models = True
        self.number_of_dynamic_models = 3
        self.models_to_skip = os.environ.get("HORDE_SKIPPED_MODELNAMES", "stable_diffusion_inpainting").split(",")
        self.predefined_models = self.model_names.copy()

    @logger.catch(reraise=True)
    def reload_data(self):
        """Reloads configuration data"""
        previous_url = self.horde_url
        bd = super().reload_data()
        if bd:
            try:
                self.max_power = bd.max_power
            except AttributeError:
                pass
            try:
                self.nsfw = bd.nsfw
            except AttributeError:
                pass
            try:
                self.censor_nsfw = bd.censor_nsfw
            except AttributeError:
                pass
            try:
                self.blacklist = bd.blacklist
            except AttributeError:
                pass
            try:
                self.censorlist = bd.censorlist
            except AttributeError:
                pass
            try:
                self.allow_img2img = bd.allow_img2img
            except AttributeError:
                pass
            try:
                self.allow_painting = bd.allow_painting
            except AttributeError:
                pass
            try:
                self.dynamic_models = bd.dynamic_models
            except AttributeError:
                pass
            try:
                self.number_of_dynamic_models = bd.number_of_dynamic_models
            except AttributeError:
                pass
            try:
                self.models_to_skip = bd.models_to_skip
            except AttributeError:
                pass
            if not self.dynamic_models:
                self.model_names = bd.models_to_load
            else:
                self.predefined_models = bd.models_to_load
            try:
                self.allow_post_processing = bd.allow_post_processing
            except AttributeError:
                pass
            try:
                self.allow_controlnet = bd.allow_controlnet
            except AttributeError:
                pass
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
        if self.dynamic_models:
            try:
                from creds import hf_password, hf_username  # noqa F401
            except ImportError:
                logger.warning(
                    "Dynamic models enabled. Please setup creds.py so it won't prompt for authentication later"
                )
        self.max_power = max(self.max_power, 2)
        self.max_pixels = 64 * 64 * 8 * self.max_power
        # if self.censor_nsfw or (self.censorlist is not None and len(self.censorlist)):
        self.model_names.append("safety_checker")
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
