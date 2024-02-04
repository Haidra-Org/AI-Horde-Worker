# webui.py
# Simple web configuration for horde worker
import argparse
import contextlib
import datetime
import glob
import math
import os
import pathlib
import shutil
import sys
import time

import gradio as gr
import requests
import yaml


# Helper class to access dictionaries
class DotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr, None)

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        if attr in self:
            del self[attr]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def default(self, attr, value):
        if attr not in self:
            self[attr] = value


class WebUI:
    CONFIG_FILE = "bridgeData.yaml"

    # This formally maps config item key name to gradio label and info.
    # The reverse lookup is also done, gradio label to config item key name.
    INFO = {
        "worker_name": {
            "label": "Worker Name",
            "info": "This is a the name of your worker. It needs to be unique to the whole horde. "
            "You cannot run different workers with the same name. It will be publicly visible.",
        },
        "dreamer_name": {
            "label": "Dreamer Name",
            "info": "(Optional) This is the name of your image generation worker. "
            "It needs to be unique to the whole horde. "
            "Overrides worker_name if specified, and defaults to worker_name if left blank/default",
        },
        "api_key": {
            "label": "API Key",
            "info": "This is your Stable Horde API Key. You can get one free at " "https://stablehorde.net/register ",
        },
        "horde_url": {
            "label": "The URL of the horde API server.",
            "info": "Don't change this unless you know exactly what you are doing.",
        },
        "stats_output_frequency": {
            "label": "Stats Output Frequency",
            "info": "How often, in seconds, that statistics such as kudos per hour are output to "
            "the display by the worker.",
        },
        "threads": {
            "label": "Number of Threads",
            "info": "Most workers leave this at 1. "
            "This determines how many jobs will be processed simultaneously. "
            "Each job requires extra VRAM and will slow the speed of generations. "
            "This should be set to provide generations at a minumum speed of 0.6 megapixels per second. "
            "Expected max per VRAM size: 6Gb = 1 thread, 6-8Gb = 2 threads, 8-12Gb = 3 threads, "
            "12Gb - 24Gb = 4 threads",
        },
        "queue_size": {
            "label": "Job Queue Size",
            "info": "This number determines the number of extra jobs that are collected. "
            "When the worker requests jobs it will request 1 job per thread plus this number. ",
        },
        "allow_unsafe_ip": {"label": "Allow requests from suspicious IP addresses", "info": ""},
        "require_upfront_kudos": {"label": "Accept requests only from users with kudos", "info": ""},
        "blacklist": {
            "label": "Blacklisted Words (Separate with commas)",
            "info": "Any words in here that match a prompt will result in that job not being "
            "accepted by this worker.",
        },
        "censorlist": {
            "label": "Censored Words (Separate with commas)",
            "info": "Any words in here that match a prompt will always result in a censored image " "being returned.",
        },
        "nsfw": {"label": "Enable NSFW", "info": "Allow your worker to accept jobs that contain NSFW " "content."},
        "censor_nsfw": {
            "label": "Censor NSFW images",
            "info": "If this is true, the worker will scan all resulting images for NSFW and censor any detected. "
            "If this is false, the worker will only scan for NSFW on client request. "
            "This does nothing is 'Enable NSFW' is set to True.",
        },
        "cache_home": {
            "label": "Model Directory",
            "info": "Downloaded models files are stored here. The default './' is means the AI-Horde-Worker "
            "directory (check for a folder name 'nataili' after your first run).",
        },
        "temp_dir": {
            "label": "Model Cache Directory",
            "info": "Model cache data is stored here. Downloaded models are processed and copies stored "
            "here if you load too many models to fit in RAM and VRAM.",
        },
        "always_download": {
            "label": "Automatically download required models",
            "info": "Download any required models without asking you first.",
        },
        "dynamic_models": {
            "label": "Enable dynamic models",
            "info": "In addition to any other models you have selected to load, you can select this to "
            "have your worker automatically load whatever models are in high demand on the horde right "
            "now. This constantly checks what models are in highest demand and loads them.",
        },
        "number_of_dynamic_models": {
            "label": "Number of Models to Dynamically Load",
            "info": "This number of high demand models will be dynamically loaded, in addition to any "
            "other models you have selected to load.",
        },
        "max_models_to_download": {
            "label": "Maximum Number of Models to Download",
            "info": "This number is the maximum number of models that the worker will download and run. "
            "Each model can take between 2 GB to 8 GB, ensure you have enough storage space available. "
            "This number includes system models such as the safety checker and the post-processors, so "
            "don't set it too low!",
        },
        "alchemist_name": {
            "label": "Alchemist Name",
            "info": "(Optional) This is the name of your Alchemist. It needs to be unique to the whole horde. "
            "Overrides worker_name if specified, and defaults to worker_name if left blank/default",
        },
        "forms": {
            "label": "Alchemy Worker Features",
            "info": "Enable or disable the different types of requests accepted by this worker if you"
            "run an Alchemy worker (image interrogation and upscaling worker)",
        },
        "allow_img2img": {
            "label": "Allow img2img requests",
            "info": "Enable or disable the processing of img2img jobs.",
        },
        "allow_painting": {
            "label": "Allow inpainting requests",
            "info": "Enable or disable the processing of inpainting jobs.",
        },
        "allow_post_processing": {
            "label": "Allow requests requiring post-processing",
            "info": "Enable or disable the processing of jobs that also require post-processing.",
        },
        "allow_controlnet": {
            "label": "Allow requests requiring ControlNet",
            "info": "Enable or disable the processing of jobs that also require ControlNet.",
        },
        "allow_lora": {
            "label": "Allow LoRas to be used by this worker",
            "info": "Your worker will download the top 10Gb of non-character LoRas and then will ad-hoc download any "
            "LoRa requested which you do not have, and cache that for a number a days",
        },
        "max_lora_cache_size": {
            "label": "Lora Cache Size (In gigabytes!) ",
            "info": "Use this setting to control how much extra space LoRas can take after you downloaded the Top."
            "If a new Lora would exceed this space, an old lora you've downloaded previously will be deleted. "
            "!Note! THIS IS ON TOP OF THE CURATED LORAs, so plan around +5G more than this",
        },
        "disable_terminal_ui": {
            "label": "Disable Terminal UI",
            "info": "Disable the display of the terminal UI, just output lines to the terminal",
        },
        "priority_usernames": {
            "label": "Priority Usernames (Separate with commas)",
            "info": "These users will be prioritized over all others when submitting jobs. "
            "Enter in format username#id e.g. residentchiefnz#3966. You do not need "
            "to add your own name to this list",
        },
        "max_power": {
            "label": "Maximum Image Size",
            "info": "This is the maximum image size your worker can generate. Start small at 512x512. "
            "Larger images use a significant amount of VRAM, if you go too large your worker will crash. "
            "Common numbers are 2 (256x256), 8 (512x512), 18 (768x768), and 32 (1024x1024)",
        },
        "models_on_disk": {
            "label": "Already Downloaded Models To Load",
            "info": "These are models which are already downloaded to your worker.",
        },
        "models_to_load": {
            "label": "Individual Models To Load",
            "info": "You can select individual models to load here. These are loaded in addition to "
            "any other models you have selected, such as 'Top 5' and dynamic models.",
        },
        "models_to_skip": {
            "label": "Models To Skip",
            "info": "Any model you select here will NEVER be downloaded to your worker, regardless of "
            "any other model loading settings. Use this to completely exclude a model from your worker.",
        },
        "special_models_to_load": {
            "label": "Loading Groups of Models",
            "info": "You can select groups of models here. 'All Models' loads all possible models "
            "which will take over 500gb of space in the folder defined by the setting 'cache_home'. "
            "The other options load different subsets of models based on style. You can select "
            "more than one.",
        },
        "special_top_models_to_load": {
            "label": "Automatically Loading Popular Models",
            "info": "Choose to automatically load the top 'n' most popular models of the day.",
        },
        "ram_to_leave_free": {
            "label": "RAM to Leave Free (%)",
            "info": "This is the amount of RAM to leave free for your system to use. You should raise this value "
            "if you expect to run other programs on your computer while running your worker.",
        },
        "vram_to_leave_free": {
            "label": "VRAM to Leave Free (%)",
            "info": "This is the amount of VRAM to leave free for your system to use. ",
        },
        "scribe_name": {
            "label": "Scribe Name",
            "info": "(Optional) This is a the name of your scribe worker. It needs to be unique to the whole horde. "
            "You cannot run different workers with the same name. It will be publicly visible. "
            "Overrides worker_name if specified, and defaults to worker_name if left blank/default",
        },
        "kai_url": {
            "label": "Kai URL",
            "info": "This is the URL of the Kobold AI Client API you want your worker to connect to. "
            "You will probably be running your own Kobold AI Client, and you should enter the URL here.",
        },
        "max_length": {
            "label": "Maximum Length",
            "info": "This is the maximum number of tokens your worker will generate per request.",
        },
        "max_context_length": {
            "label": "Maximum Context Length",
            "info": "The max tokens to use from the prompt.",
        },
        "branded_model": {
            "label": "Branded Model",
            "info": " This will prevent the model from being used from the shared pool, but will ensure that"
            " no other worker can pretend to serve it If you are unsure, leave this as 'None'.",
        },
    }

    models_found_on_disk = None

    def __init__(self):
        self.app = None
        self.models_found_on_disk = []

    def _label(self, name):
        return WebUI.INFO[name]["label"] if name in WebUI.INFO else None

    def _info(self, name):
        return f"{WebUI.INFO[name]['info']} [{name}]" if name in WebUI.INFO else None

    # Label to config item name
    def _cfg(self, label):
        return next(
            (key for key, value in WebUI.INFO.items() if value["label"] == label),
            None,
        )

    def reload_config(self):
        # Sanity check, to ensure Tazlin doesn't give me a hard time
        # about this corner case [jug]
        if os.path.exists("bridgeData.py"):
            print(
                "You have a very old config file. Please run your worker "
                "at least once to update to the new format and then try again "
                "with this webUI",
                file=sys.stderr,
            )
            exit(1)

        if not os.path.exists(WebUI.CONFIG_FILE):
            # Create it from the template
            shutil.copy("bridgeData_template.yaml", WebUI.CONFIG_FILE)

        with open(WebUI.CONFIG_FILE, "rt", encoding="utf-8") as configfile:
            data = yaml.safe_load(configfile)

        return DotDict(data)

    def process_input_list(self, list):
        output = []
        if list != "":
            temp = list.split(",")
            for item in temp:
                trimmed_item = item.strip()
                output.append(trimmed_item)
        return output

    def save_config(self, args):
        args = DotDict(args)

        # Grab the existing config file contents
        config = self.reload_config()

        # Merge values which require some pre-processing
        skipped_keys = ["models_on_disk", "special_models_to_load", "special_top_models_to_load"]
        models_to_load = []
        for key, value in args.items():
            cfgkey = self._cfg(key.label)
            if cfgkey == "priority_usernames" or cfgkey == "blacklist" or cfgkey == "censorlist":
                config[cfgkey] = self.process_input_list(value)
                continue
            if cfgkey == "ram_to_leave_free" or cfgkey == "vram_to_leave_free":
                config[cfgkey] = str(value) + "%"
                continue

            if cfgkey == "special_models_to_load" or cfgkey == "models_on_disk":
                models_to_load.extend(value)
            elif cfgkey == "special_top_models_to_load":
                if value and value != "None":
                    models_to_load.append(value)
            elif cfgkey == "models_to_load":
                models_to_load.extend(value)
            elif cfgkey == "dreamer_name" and (value == "An Awesome Dreamer" or not value):
                skipped_keys.append("dreamer_name")
            elif cfgkey == "scribe_name" and (value == "An Awesome Scribe" or not value):
                skipped_keys.append("scribe_name")
            elif cfgkey == "alchemist_name" and (value == "An Awesome Alchemist" or not value):
                skipped_keys.append("alchemist_name")

            config[cfgkey] = value if cfgkey != "models_to_load" else None

        config["models_to_load"] = models_to_load
        with open(WebUI.CONFIG_FILE, "wt", encoding="utf-8") as configfile:
            yaml.safe_dump({k: v for k, v in config.items() if k not in skipped_keys}, configfile)

        return f"Configuration Saved at {datetime.datetime.now()}"

    def download_models(self, model_location):
        models = None
        try:
            r = requests.get(model_location)
            models = r.json()
            print("Models downloaded successfully")
        except Exception:
            print("Failed to load models")
        return models

    def load_models(self):
        remote_models = (
            "https://raw.githubusercontent.com/Haidra-org/AI-Horde-image-model-reference/main/stable_diffusion.json"
        )
        latest_models = self.download_models(remote_models)
        if not latest_models or not isinstance(latest_models, dict):
            print("Failed to load models")
            latest_models = {}
        aiworker_cache_home = os.environ.get("AIWORKER_CACHE_HOME", None)
        model_cache_folder = aiworker_cache_home if aiworker_cache_home else "./"

        sub_folders = ["", "nataili", "models"]

        sd_models_folders: list[pathlib.Path] = [
            pathlib.Path(model_cache_folder).joinpath(x).joinpath("compvis") for x in sub_folders
        ]
        for sd_models_folder in sd_models_folders:
            if sd_models_folder.exists():
                all_files_in_cache = glob.glob(str(sd_models_folder.joinpath("*.*")))
                all_files_in_cache = [
                    pathlib.Path(x).name for x in all_files_in_cache if x.endswith((".ckpt", ".safetensors"))
                ]
                for model_name, model_info in latest_models.items():
                    model_config_dict: dict = model_info.get("config", None)
                    if not model_config_dict:
                        continue
                    model_file_config_list: list = model_config_dict.get("files")
                    if not model_file_config_list:
                        continue
                    if len(model_file_config_list) == 0:
                        continue
                    model_filename: str | None = None
                    for key in model_file_config_list:
                        model_filename = key.get("path", None)
                        if model_filename and "yaml" not in model_filename:
                            break
                    if model_filename and model_filename in all_files_in_cache:
                        if self.models_found_on_disk is None:
                            self.models_found_on_disk = []
                        self.models_found_on_disk.append(model_name)
                break

        return sorted(latest_models, key=str.casefold)

    def load_workerID(self, worker_name):
        workerID = ""
        workers_URL = "https://stablehorde.net/api/v2/workers"
        r = requests.get(workers_URL)
        worker_json = r.json()
        for item in worker_json:
            if item["name"] == worker_name:
                workerID = item["id"]
        return workerID

    def load_worker_mode(self, worker_name):
        worker_mode = False
        workers_URL = "https://stablehorde.net/api/v2/workers"
        r = requests.get(workers_URL)
        worker_json = r.json()
        for item in worker_json:
            if item["name"] == worker_name:
                worker_mode = item["maintenance_mode"]
        return worker_mode

    def load_worker_stats(self, worker_name):
        worker_stats = ""
        workers_URL = "https://stablehorde.net/api/v2/workers"
        r = requests.get(workers_URL)
        worker_json = r.json()
        for item in worker_json:
            if item["name"] == worker_name:
                worker_stats += "Current MPS:  " + str(item["performance"]).split()[0] + " MPS\n"
                worker_stats += "Total Kudos Earned:  " + str(item["kudos_rewards"]) + "\n"
                worker_stats += "Total Jobs Completed:  " + str(item["requests_fulfilled"])
        return worker_stats

    def update_worker_mode(self, worker_name, worker_id, current_mode, apikey):
        header = {"apikey": apikey}
        payload = {"maintenance": False, "name": worker_name}
        if current_mode == "False":
            payload = {"maintenance": True, "name": worker_name}
        worker_URL = f"https://stablehorde.net/api/v2/workers/{worker_id}"
        requests.put(worker_URL, json=payload, headers=header)
        state = "enabled" if payload["maintenance"] else "disabled"
        return f"Maintenance mode is being {state}, this may take up to 30 seconds to update here. Please wait."

    def _imgsize(self, value):
        try:
            pixels = int(math.sqrt(64 * 64 * 8 * value))
        except ValueError:
            pixels = 0
        return f"Maximum image size of approximately {pixels}x{pixels}"

    def initialise(self):
        config = self.reload_config()

        model_list = self.load_models()
        model_list = [model for model in model_list if model not in self.models_found_on_disk]
        models_on_disk = []
        # Seperate out the magic constants
        models_to_load_all = []
        models_to_load_top = "None"
        models_to_load_individual = []
        for model in config.models_to_load:
            if not model:
                continue
            if model.lower().startswith("top "):
                models_to_load_top = model.title()
            elif model.lower().startswith("all "):
                models_to_load_all.append(model.title())
            elif model in self.models_found_on_disk:
                models_on_disk.append(model)
            else:
                models_to_load_individual.append(model)

        existing_priority_usernames = ""
        config.default("priority_usernames", [])
        for item in config.priority_usernames:
            existing_priority_usernames += item
            existing_priority_usernames += ","
        if len(existing_priority_usernames) > 0 and existing_priority_usernames[-1] == ",":
            existing_priority_usernames = existing_priority_usernames[:-1]

        existing_blacklist = ""
        config.default("blacklist", [])
        for item in config.blacklist:
            existing_blacklist += item
            existing_blacklist += ","
        if len(existing_blacklist) > 0 and existing_blacklist[-1] == ",":
            existing_blacklist = existing_blacklist[:-1]

        existing_censorlist = ""
        config.default("censorlist", [])
        for item in config.censorlist:
            existing_censorlist += item
            existing_censorlist += ","
        if len(existing_censorlist) > 0 and existing_censorlist[-1] == ",":
            existing_censorlist = existing_censorlist[:-1]

        # Load css if it exists
        css = ""
        if os.path.exists("webui.css"):
            with open("webui.css", "rt", encoding="utf-8", errors="ignore") as cssfile:
                css = cssfile.read()

        with gr.Blocks(css=css) as self.app:
            gr.Markdown("# AI Horde Worker Configuration")

            with gr.Row():
                with gr.Tab("Basic Settings"), gr.Column():
                    worker_name = gr.Textbox(
                        label=self._label("worker_name"),
                        value=config.worker_name,
                        info=self._info("worker_name"),
                    )
                    config.default("dreamer_name", "An Awesome Dreamer")
                    dreamer_name = gr.Textbox(
                        label=self._label("dreamer_name"),
                        value=config.dreamer_name,
                        info=self._info("dreamer_name"),
                    )
                    config.default("alchemist_name", "An Awesome Alchemist")
                    alchemist_name = gr.Textbox(
                        label=self._label("alchemist_name"),
                        value=config.alchemist_name,
                        info=self._info("alchemist_name"),
                    )
                    api_key = gr.Textbox(
                        label=self._label("api_key"),
                        value=config.api_key,
                        type="password",
                        info=self._info("api_key"),
                    )
                    slider_desc = gr.Markdown("Maximum Image Size")
                    config.default("max_power", 8)
                    max_power = gr.Slider(
                        2,
                        128,
                        step=2,
                        label=self._label("max_power"),
                        show_label=False,
                        value=config.max_power,
                        info=self._info("max_power"),
                    )
                    # Hook the slider on change event to display image size
                    max_power.change(fn=self._imgsize, inputs=max_power, outputs=slider_desc)
                    priority_usernames = gr.Textbox(
                        label=self._label("priority_usernames"),
                        value=existing_priority_usernames,
                        info=self._info("priority_usernames"),
                    )

                with gr.Tab("Enable Features"), gr.Column():
                    config.default("allow_img2img", True)
                    allow_img2img = gr.Checkbox(
                        label=self._label("allow_img2img"),
                        value=config.allow_img2img,
                        info=self._info("allow_img2img"),
                    )
                    config.default("allow_painting", False)
                    allow_painting = gr.Checkbox(
                        label=self._label("allow_painting"),
                        value=config.allow_painting,
                        info=self._info("allow_painting"),
                    )
                    config.default("allow_post_processing", True)
                    allow_post_processing = gr.Checkbox(
                        label=self._label("allow_post_processing"),
                        value=config.allow_post_processing,
                        info=self._info("allow_post_processing"),
                    )
                    config.default("allow_controlnet", False)
                    allow_controlnet = gr.Checkbox(
                        label=self._label("allow_controlnet"),
                        value=config.allow_controlnet,
                        info=self._info("allow_controlnet"),
                    )
                    config.default("allow_lora", False)
                    allow_lora = gr.Checkbox(
                        label=self._label("allow_lora"),
                        value=config.allow_lora,
                        info=self._info("allow_lora"),
                    )
                    config.default("max_lora_cache_size", 10)
                    max_lora_cache_size = gr.Slider(
                        label=self._label("max_lora_cache_size"),
                        value=config.max_lora_cache_size,
                        info=self._info("max_lora_cache_size"),
                        minimum=10,
                        maximum=1024,
                        step=1,
                    )
                    config.default("forms", [])
                    forms = gr.CheckboxGroup(
                        label=self._label("forms"),
                        choices=["caption", "nsfw", "interrogation", "post-process"],
                        value=config.forms,
                        info=self._info("forms"),
                    )

                with gr.Tab("Models To Load"):
                    with gr.Row():
                        special_models_to_load = gr.CheckboxGroup(
                            choices=[
                                "All Models",
                                "All Realistic Models",
                                "All Anime Models",
                                "All Generalist Models",
                                "All Furry Models",
                                "All Artistic Models",
                                "All Other Models",
                            ],
                            label=self._label("special_models_to_load"),
                            value=models_to_load_all,
                            info=self._info("special_models_to_load"),
                        )
                    with gr.Row():
                        special_top_models_to_load = gr.Radio(
                            choices=[
                                "None",
                                "Top 1",
                                "Top 2",
                                "Top 3",
                                "Top 4",
                                "Top 5",
                                "Top 6",
                                "Top 7",
                                "Top 8",
                                "Top 9",
                                "Top 10",
                            ],
                            label=self._label("special_top_models_to_load"),
                            value=models_to_load_top,
                            info=self._info("special_top_models_to_load"),
                        )
                    with gr.Row(), gr.Column():
                        models_on_disk = gr.CheckboxGroup(
                            choices=self.models_found_on_disk,
                            label=self._label("models_on_disk"),
                            value=models_on_disk,
                            info=self._info("models_on_disk"),
                        )
                    with gr.Row(), gr.Column():
                        models_to_load = gr.CheckboxGroup(
                            choices=model_list,
                            label=self._label("models_to_load"),
                            value=models_to_load_individual,
                            info=self._info("models_to_load"),
                        )

                with gr.Tab("Models to Skip"), gr.Column():
                    config.default("models_to_skip", [])
                    models_to_skip = gr.CheckboxGroup(
                        choices=model_list,
                        label=self._label("models_to_skip"),
                        value=config.models_to_skip,
                        info=self._info("models_to_skip"),
                    )

                with gr.Tab("Model Management"), gr.Column():
                    config.default("always_download", True)
                    always_download = gr.Checkbox(
                        label=self._label("always_download"),
                        value=config.always_download,
                        info=self._info("always_download"),
                    )
                    config.default("max_models_to_download", 10)
                    max_models_to_download = gr.Number(
                        label=self._label("max_models_to_download"),
                        value=config.max_models_to_download,
                        precision=0,
                        info=self._info("max_models_to_download"),
                    )
                    config.default("dynamic_models", True)
                    dynamic_models = gr.Checkbox(
                        label=self._label("dynamic_models"),
                        value=config.dynamic_models,
                        info=self._info("dynamic_models"),
                    )
                    config.default("number_of_dynamic_models", 3)
                    number_of_dynamic_models = gr.Number(
                        label=self._label("number_of_dynamic_models"),
                        value=config.number_of_dynamic_models,
                        precision=0,
                        info=self._info("number_of_dynamic_models"),
                    )
                    config.default("cache_home", "./")
                    cache_home = gr.Textbox(
                        label=self._label("cache_home"),
                        value=config.cache_home,
                        info=self._info("cache_home"),
                    )
                    config.default("temp_dir", "./tmp")
                    temp_dir = gr.Textbox(
                        label=self._label("temp_dir"),
                        value=config.temp_dir,
                        info=self._info("temp_dir"),
                    )

                with gr.Tab("Security"), gr.Column():
                    config.default("allow_unsafe_ip", False)
                    allow_unsafe_ip = gr.Checkbox(
                        label=self._label("allow_unsafe_ip"),
                        value=config.allow_unsafe_ip,
                        info=self._info("allow_unsafe_ip"),
                    )
                    config.default("require_upfront_kudos", False)
                    require_upfront_kudos = gr.Checkbox(
                        label=self._label("require_upfront_kudos"),
                        value=config.require_upfront_kudos,
                        info=self._info("require_upfront_kudos"),
                    )
                    blacklist = gr.Textbox(
                        label=self._label("blacklist"),
                        value=existing_blacklist,
                        info=self._info("blacklist"),
                    )
                    censorlist = gr.Textbox(
                        label=self._label("censorlist"),
                        value=existing_censorlist,
                        info=self._info("censorlist"),
                    )
                    config.default("nsfw", True)
                    nsfw = gr.Checkbox(
                        label=self._label("nsfw"),
                        value=config.nsfw,
                        info=self._info("nsfw"),
                    )
                    config.default("censor_nsfw", False)
                    censor_nsfw = gr.Checkbox(
                        label=self._label("censor_nsfw"),
                        value=config.censor_nsfw,
                        info=self._info("censor_nsfw"),
                    )

                with gr.Tab("Performance"), gr.Column():
                    config.default("threads", 1)
                    max_threads = gr.Slider(
                        1,
                        8,
                        step=1,
                        label=self._label("threads"),
                        value=config.max_threads,
                        info=self._info("threads"),
                    )
                    config.default("queue_size", 1)
                    queue_size = gr.Slider(
                        0,
                        2,
                        step=1,
                        label=self._label("queue_size"),
                        value=config.queue_size,
                        info=self._info("queue_size"),
                    )
                    parsed_ram_from_config = 40
                    with contextlib.suppress(Exception):
                        parsed_ram_from_config = int(config.ram_to_leave_free.split("%")[0])
                    ram_to_leave_free = gr.Slider(
                        0,
                        100,
                        step=1,
                        label=self._label("ram_to_leave_free"),
                        value=parsed_ram_from_config,
                        info=self._info("ram_to_leave_free"),
                    )
                    parsed_vram_from_config = 40
                    with contextlib.suppress(Exception):
                        parsed_vram_from_config = int(config.vram_to_leave_free.split("%")[0])

                    vram_to_leave_free = gr.Slider(
                        0,
                        100,
                        step=1,
                        label=self._label("vram_to_leave_free"),
                        value=parsed_vram_from_config,
                        info=self._info("vram_to_leave_free"),
                    )

                with gr.Tab("Advanced"), gr.Column():
                    config.default("disable_terminal_ui", False)
                    disable_terminal_ui = gr.Checkbox(
                        label=self._label("disable_terminal_ui"),
                        value=config.disable_terminal_ui,
                        info=self._info("disable_terminal_ui"),
                    )
                    config.default("horde_url", "https://aihorde.net/")
                    horde_url = gr.Textbox(
                        label=self._label("horde_url"),
                        value=config.horde_url,
                        info=self._info("horde_url"),
                    )
                    config.default("stats_output_frequency", 30)
                    stats_output_frequency = gr.Number(
                        label=self._label("stats_output_frequency"),
                        value=config.stats_output_frequency,
                        precision=0,
                        info=self._info("stats_output_frequency"),
                    )

                with gr.Tab("Worker Control"), gr.Column():
                    gr.Markdown(
                        "Enable maintenance mode to prevent this worker fetching any more jobs to process. "
                        "Jobs that you submit yourself will still be picked up by your worker even if maintenance "
                        "mode is enabled.",
                    )
                    maint_button = gr.Button(value="Toggle Maintenance Mode", variant="secondary")
                    maint_message = gr.Markdown("")
                    worker_id = gr.Textbox(label="Worker ID")
                    maintenance_mode = gr.Textbox(label="Current Maintenance Mode Status")
                    self.app.load(self.load_workerID, inputs=worker_name, outputs=worker_id, every=15)
                    self.app.load(self.load_worker_mode, inputs=worker_name, outputs=maintenance_mode, every=15)

                    maint_button.click(
                        self.update_worker_mode,
                        inputs=[worker_name, worker_id, maintenance_mode, api_key],
                        outputs=[maint_message],
                    )
                with gr.Tab("Scribe Options"), gr.Column():
                    gr.Markdown(
                        "Options for the Scribes (text workers)",
                    )
                    config.default("scribe_name", "An Awesome Scribe")
                    scribe_name = gr.Textbox(
                        label=self._label("scribe_name"),
                        value=config.scribe_name,
                        info=self._info("scribe_name"),
                    )
                    config.default("kai_url", "http://localhost:5000")
                    kai_url = gr.Textbox(
                        label=self._label("kai_url"),
                        value=config.kai_url,
                        info=self._info("kai_url"),
                    )

                    config.default("max_length", 80)
                    max_length = gr.Slider(
                        0,
                        240,
                        step=10,
                        label=self._label("max_length"),
                        value=config.max_length,
                        info=self._info("max_length"),
                    )
                    config.default("max_context_length", 1024)
                    max_context_length = gr.Slider(
                        0,
                        8192,
                        step=128,
                        label=self._label("max_context_length"),
                        value=config.max_context_length,
                        info=self._info("max_context_length"),
                    )
                    config.default("branded_model", False)
                    branded_model = gr.Checkbox(
                        label=self._label("branded_model"),
                        value=config.branded_model,
                        info=self._info("branded_model"),
                    )

            with gr.Row():
                submit = gr.Button(value="Save Configuration", variant="primary")
            with gr.Row():
                message = gr.Markdown("")

            submit.click(
                self.save_config,
                inputs={
                    alchemist_name,
                    allow_controlnet,
                    allow_img2img,
                    allow_painting,
                    allow_post_processing,
                    allow_unsafe_ip,
                    always_download,
                    api_key,
                    blacklist,
                    censor_nsfw,
                    censorlist,
                    dreamer_name,
                    dynamic_models,
                    disable_terminal_ui,
                    forms,
                    horde_url,
                    max_models_to_download,
                    max_power,
                    max_threads,
                    models_on_disk,
                    models_to_load,
                    models_to_skip,
                    cache_home,
                    nsfw,
                    number_of_dynamic_models,
                    priority_usernames,
                    queue_size,
                    temp_dir,
                    require_upfront_kudos,
                    special_models_to_load,
                    special_top_models_to_load,
                    stats_output_frequency,
                    worker_name,
                    ram_to_leave_free,
                    vram_to_leave_free,
                    scribe_name,
                    kai_url,
                    max_length,
                    max_context_length,
                    branded_model,
                    allow_lora,
                    max_lora_cache_size,
                },
                outputs=[message],
            )

        self.app.queue()

    def run(self, share, nobrowser, lan, user, password):
        server_name = "0.0.0.0" if lan else None
        self.initialise()
        self.app.launch(
            quiet=True,
            share=share,
            auth=(user, password) if user and password else None,
            inbrowser=not nobrowser,
            server_name=server_name,
            prevent_thread_lock=True,
        )
        while True:
            time.sleep(0.1)


if __name__ == "__main__":
    # Check args
    parser = argparse.ArgumentParser(description="Horde Web Configuration")
    parser.add_argument("--share", action="store_true", help="Create a public URL")
    parser.add_argument("--no-browser", action="store_true", help="Don't open automatically in a web browser")
    parser.add_argument("--lan", action="store_true", help="Allow access on the local network")
    parser.add_argument("--user", action="store", nargs=1, help="Username for authentication")
    parser.add_argument("--password", action="store", nargs=1, help="Password for authentication")
    parser.add_argument("--no-auth", action="store_true", help="Disable authentication")
    args = parser.parse_args()

    if args.share or args.lan:
        if not args.no_auth and not args.user:
            print(
                (
                    "WARNING: You are running in public mode without authentication. This is not recommended.\n"
                    "To enable authentication, use the --user and --password arguments.\n"
                    "If you are running in a trusted environment, you can disable this warning with the "
                    "--no-auth argument.\n"
                    "Continue without authentication? (y/n)"
                ),
            )
            if input().lower() != "y":
                exit(0)

        if (args.user or args.password) and not (args.user and args.password):
            parser.error("--user and --password must both be specified")

    elif args.user or args.password:
        parser.error("--user and --password can only be used with --share or --lan")

    ui = WebUI()
    ui.run(
        args.share,
        args.no_browser,
        args.lan,
        args.user[0] if args.user else None,
        args.password[0] if args.password else None,
    )
