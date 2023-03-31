"""The configuration of the bridge"""
import os

from nataili.util.logger import logger

from worker.argparser.interrogation import args
from worker.bridge_data.framework import BridgeDataTemplate
from worker.consts import POST_PROCESSORS_NATAILI_MODELS


class InterrogationBridgeData(BridgeDataTemplate):
    """Configuration object"""

    def __init__(self):
        super().__init__(args)
        self.forms = os.environ.get("HORDE_INTERROGATION_FORMS", "caption").split(",")
        self.model_names = []

    @logger.catch(reraise=True)
    def reload_data(self):
        """Reloads configuration data"""
        previous_url = self.horde_url
        super().reload_data()
        if args.forms:
            self.forms = args.forms
        # Ensure no duplicates
        self.forms = list(set(self.forms))
        if "nsfw" in self.forms and "safety_checker" not in self.model_names:
            self.model_names.append("safety_checker")
        if "caption" in self.forms and "BLIP_Large" not in self.model_names:
            self.model_names.append("BLIP_Large")
        if "interrogation" in self.forms and "ViT-L/14" not in self.model_names:
            self.model_names.append("ViT-L/14")
        if "post-process" in self.forms:
            self.model_names += list(POST_PROCESSORS_NATAILI_MODELS)
        if (not self.initialized and not self.models_reloading) or previous_url != self.horde_url:
            logger.init(
                (
                    f"Username '{self.username}'. Server Name '{self.worker_name}'. "
                    f"Horde URL '{self.horde_url}'. Forms {self.forms}. "
                    "Worker Type: Interrogation"
                ),
                status="Joining Horde",
            )
