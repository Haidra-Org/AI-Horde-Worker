"""The configuration of the bridge"""
import os

import requests
from loguru import logger

from worker.argparser.scribe import args
from worker.bridge_data.framework import BridgeDataTemplate


class KoboldAIBridgeData(BridgeDataTemplate):
    """Configuration object"""

    def __init__(self):
        super().__init__(args)
        self.kai_available = False
        self.model = None
        self.kai_url = "http://localhost:5000"
        self.max_length = int(os.environ.get("HORDE_MAX_LENGTH", "80"))
        self.max_context_length = int(os.environ.get("HORDE_MAX_CONTEXT_LENGTH", "1024"))
        self.branded_model = os.environ.get("HORDE_BRANDED_MODEL", "false") == "true"
        self.softprompts = {}
        self.current_softprompt = None

        self.nsfw = os.environ.get("HORDE_NSFW", "true") == "true"
        self.blacklist = list(filter(lambda a: a, os.environ.get("HORDE_BLACKLIST", "").split(",")))

    @logger.catch(reraise=True)
    def reload_data(self):
        """Reloads configuration data"""
        previous_url = self.horde_url
        super().reload_data()
        if hasattr(self, "scribe_name") and not self.args.worker_name:
            self.worker_name = self.scribe_name
        if args.kai_url:
            self.kai_url = args.kai_url
        if args.sfw:
            self.nsfw = False
        if args.blacklist:
            self.blacklist = args.blacklist
        self.validate_kai()
        if self.kai_available and not self.initialized and previous_url != self.horde_url:
            logger.init(
                (
                    f"Username '{self.username}'. Server Name '{self.worker_name}'. "
                    f"Horde URL '{self.horde_url}'. KoboldAI Client URL '{self.kai_url}'"
                    "Worker Type: Scribe"
                ),
                status="Joining Horde",
            )

    @logger.catch(reraise=True)
    def validate_kai(self):
        logger.debug("Retrieving settings from KoboldAI Client...")
        try:
            req = requests.get(self.kai_url + "/api/latest/model")
            self.model = req.json()["result"]
            # Normalize huggingface and local downloaded model names
            if "/" not in self.model:
                self.model = self.model.replace("_", "/", 1)
            # Now using the settings from the bridge explicitly
            # req = requests.get(self.kai_url + "/api/latest/config/max_context_length")
            # self.max_context_length = req.json()["value"]
            # req = requests.get(self.kai_url + "/api/latest/config/max_length")
            # self.max_length = req.json()["value"]
            if self.model not in self.softprompts:
                req = requests.get(self.kai_url + "/api/latest/config/soft_prompts_list")
                self.softprompts[self.model] = [sp["value"] for sp in req.json()["values"]]
            req = requests.get(self.kai_url + "/api/latest/config/soft_prompt")
            self.current_softprompt = req.json()["value"]
        except requests.exceptions.JSONDecodeError:
            logger.error(f"Server {self.kai_url} is up but does not appear to be a KoboldAI server.")
            self.kai_available = False
            return
        except requests.exceptions.ConnectionError:
            logger.error(f"Server {self.kai_url} is not reachable. Are you sure it's running?")
            self.kai_available = False
            return
        self.kai_available = True
