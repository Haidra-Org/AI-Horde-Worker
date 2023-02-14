"""The configuration of the bridge"""
import importlib
import os
import random
import sys
import threading

import requests
from nataili import disable_voodoo
from nataili.util.logger import logger

from worker.consts import BRIDGE_VERSION


class BridgeDataTemplate:
    """Configuration object"""

    def __init__(self, args):
        random.seed()
        # I have to pass the args from the extended class, as the framework class doesn't
        # know what kind of polymorphism this worker is using
        self.args = args
        self.horde_url = os.environ.get("HORDE_URL", "https://stablehorde.net")
        # Give a cool name to your instance
        self.worker_name = os.environ.get(
            "HORDE_WORKER_NAME",
            f"Automated Instance #{random.randint(-100000000, 100000000)}",
        )
        # The api_key identifies a unique user in the horde
        self.api_key = os.environ.get("HORDE_API_KEY", "0000000000")
        # Put other users whose prompts you want to prioritize.
        # The owner's username is always included so you don't need to add it here,
        # unless you want it to have lower priority than another user
        self.priority_usernames = list(filter(lambda a: a, os.environ.get("HORDE_PRIORITY_USERNAMES", "").split(",")))
        self.max_threads = int(os.environ.get("HORDE_MAX_THREADS", 1))
        self.queue_size = int(os.environ.get("HORDE_QUEUE_SIZE", 0))
        self.allow_unsafe_ip = os.environ.get("HORDE_ALLOW_UNSAFE_IP", "true") == "true"
        self.require_upfront_kudos = os.environ.get("REQUIRE_UPFRONT_KUDOS", "false") == "true"
        self.initialized = False
        self.username = None
        self.models_reloading = False
        self.max_models_to_download = 10

        self.disable_voodoo = disable_voodoo

    @logger.catch(reraise=True)
    def reload_data(self):
        """Reloads configuration data"""
        previous_api_key = self.api_key
        try:
            # TODO - move this to a yaml file
            import bridgeData as bd

            importlib.reload(bd)
            self.api_key = bd.api_key
            self.worker_name = bd.worker_name
            self.horde_url = bd.horde_url
            self.priority_usernames = bd.priority_usernames
            try:
                self.allow_unsafe_ip = bd.allow_unsafe_ip
            except AttributeError:
                pass
            try:
                self.max_threads = bd.max_threads
            except AttributeError:
                pass
            try:
                self.queue_size = bd.queue_size
            except AttributeError:
                pass
            try:
                self.require_upfront_kudos = bd.require_upfront_kudos
            except AttributeError:
                pass
            try:
                self.max_models_to_download = bd.max_models_to_download
            except AttributeError:
                pass
        except (ImportError, AttributeError) as err:
            logger.warning("bridgeData.py could not be loaded. Using defaults with anonymous account - {}", err)
            bd = None
        if self.args.api_key:
            self.api_key = self.args.api_key
        if self.args.worker_name:
            self.worker_name = self.args.worker_name
        if self.args.horde_url:
            self.horde_url = self.args.horde_url
        if self.args.priority_usernames:
            self.priority_usernames = self.args.priority_usernames
        if self.args.max_threads:
            self.max_threads = self.args.max_threads
        if self.args.queue_size:
            self.queue_size = self.args.queue_size
        if self.args.allow_unsafe_ip:
            self.allow_unsafe_ip = self.args.allow_unsafe_ip
        if not self.initialized or previous_api_key != self.api_key:
            try:
                user_req = requests.get(
                    self.horde_url + "/api/v2/find_user",
                    headers={"apikey": self.api_key},
                    timeout=10,
                )
                user_req = user_req.json()
                self.username = user_req["username"]

            # pylint: disable=broad-except
            except Exception:
                logger.warning(f"Server {self.horde_url} error during find_user. Setting username 'N/A'")
                self.username = "N/A"
        return bd

    @logger.catch(reraise=True)
    def check_models(self, model_manager):
        """Check to see if we have the models needed"""
        if self.models_reloading:
            return
        if not self.initialized:
            logger.init("Models", status="Checking")
        models_exist = True
        not_found_models = []
        for model in self.model_names.copy():
            # logger.info(f"Checking: {model}")
            model_info = model_manager.models.get(model, None)
            if not model_info:
                logger.warning(
                    f"Model name requested {model} in bridgeData is unknown to us. "
                    "Please check your configuration. Aborting!"
                )
                self.model_names.remove(model)
                continue
            if int(model_info.get("min_bridge_version", 0)) > BRIDGE_VERSION:
                logger.warning(
                    f"Model requested {model} in bridgeData is not supported in bridge version {BRIDGE_VERSION}. "
                    "Please upgrade your bridge. Skipping."
                )
                self.model_names.remove(model)
                continue
            if model in model_manager.get_loaded_models_names():
                continue
            if not model_manager.validate_model(model, skip_checksum=self.args.skip_md5):
                logger.debug(f"Model {model} not found or has wrong checksum")
                if (
                    model_manager.count_available_models_by_types() + len(not_found_models)
                    < self.max_models_to_download
                ):
                    models_exist = False
                    not_found_models.append(model)
                else:
                    logger.debug(f"Downloading Model {model} would exceed max_models_to_download. Skipping")
                    self.model_names.remove(model)
        if not models_exist:
            if self.args.yes or self.check_extra_conditions_for_download_choice():
                choice = "y"
            else:
                choice = input(
                    "You do not appear to have downloaded the models needed yet.\n"
                    "You need at least a main model to proceed. "
                    f"Would you like to download your prespecified models?\n\
                y: Download {not_found_models} (default).\n\
                n: Abort and exit\n\
                all: Download all basic models (This can take a significant amount of time and bandwidth)\n\
                Please select an option: "
                )
            if choice not in ["y", "Y", "", "yes", "all", "a"]:
                sys.exit(1)
            model_manager.taint_models(not_found_models)
            if choice in ["all", "a"]:
                model_manager.download_all()
            elif choice in ["y", "Y", "", "yes"]:
                for model in not_found_models:
                    # logger.init(f"Model: {model}", status="Downloading")
                    if not model_manager.download_model(model):
                        logger.message(
                            "Something went wrong when downloading the model and it does not fit the expected "
                            "checksum. Please check that your HuggingFace authentication is correct and that "
                            "you've accepted the model license from the browser. This model will be skipped!"
                        )
                        self.model_names.remove(model)
            model_manager.init()
        if not self.initialized:
            logger.init_ok("Models", status="OK")
        if os.path.exists("./bridgeData.py"):
            if not self.initialized:
                logger.init_ok("Bridge Config", status="OK")
        elif input(
            "You do not appear to have a bridgeData.py. Would you like to create it from the template now? (y/n)"
        ) in ["y", "Y", "", "yes"]:
            with open("bridgeData_template.py", "r") as firstfile, open("bridgeData.py", "a") as secondfile:
                for line in firstfile:
                    secondfile.write(line)
            logger.message(
                "bridgeData.py created. Bridge will exit. "
                "Please edit bridgeData.py with your setup and restart the worker"
            )
            sys.exit(2)

    def check_extra_conditions_for_download_choice(self):
        """Extend if any condition on the specifics for this bridge_data will force a 'y' result"""
        return False

    def reload_models(self, model_manager):
        """Reloads models - Note this is IN A THREAD"""
        if self.models_reloading:
            return
        self.models_reloading = True
        thread = threading.Thread(target=self._reload_models, args=(model_manager,))
        thread.daemon = True
        thread.start()

    @logger.catch(reraise=True)
    def _reload_models(self, model_manager):
        for model in model_manager.get_loaded_models_names():
            if model not in self.model_names:
                logger.init(f"{model}", status="Unloading")
                model_manager.unload_model(model)
        for model in self.model_names:
            if model not in model_manager.get_loaded_models_names():
                success = model_manager.load(model, voodoo=False if self.disable_voodoo.active else True)
                if not success:
                    logger.init_err(f"{model}", status="Error")
            self.initialized = True
        self.models_reloading = False
