import argparse
import base64
import getpass
import json
import os
import random
import sys
import time
import importlib
from base64 import binascii
from io import BytesIO

import requests
from PIL import Image, UnidentifiedImageError

from nataili import disable_voodoo, disable_xformers
from nataili.util import logger, quiesce_logger, set_logger_verbosity
from nataili.util.cache import torch_gc

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-i",
    "--interval",
    action="store",
    required=False,
    type=int,
    default=1,
    help="The amount of seconds with which to check if there's new prompts to generate",
)
arg_parser.add_argument(
    "-a",
    "--api_key",
    action="store",
    required=False,
    type=str,
    help="The API key corresponding to the owner of this Horde instance",
)
arg_parser.add_argument(
    "-n",
    "--worker_name",
    action="store",
    required=False,
    type=str,
    help="The server name for the Horde. It will be shown to the world and there can be only one.",
)
arg_parser.add_argument(
    "-u",
    "--horde_url",
    action="store",
    required=False,
    type=str,
    help="The SH Horde URL. Where the bridge will pickup prompts and send the finished generations.",
)
arg_parser.add_argument(
    "--priority_usernames",
    type=str,
    action="append",
    required=False,
    help="Usernames which get priority use in this horde instance. The owner's username is always in this list.",
)
arg_parser.add_argument(
    "-p",
    "--max_power",
    type=int,
    required=False,
    help="How much power this instance has to generate pictures. Min: 2",
)
arg_parser.add_argument(
    "--sfw",
    action="store_true",
    required=False,
    help="Set to true if you do not want this worker generating NSFW images.",
)
arg_parser.add_argument(
    "--blacklist",
    nargs="+",
    required=False,
    help="List the words that you want to blacklist.",
)
arg_parser.add_argument(
    "--censorlist",
    nargs="+",
    required=False,
    help="List the words that you want to censor.",
)
arg_parser.add_argument(
    "--censor_nsfw",
    action="store_true",
    required=False,
    help="Set to true if you want this bridge worker to censor NSFW images.",
)
arg_parser.add_argument(
    "--allow_img2img",
    action="store_true",
    required=False,
    help="Set to true if you want this bridge worker to allow img2img request.",
)
arg_parser.add_argument(
    "--allow_painting",
    action="store_true",
    required=False,
    help="Set to true if you want this bridge worker to allow inpainting/outpainting requests.",
)
arg_parser.add_argument(
    "--allow_unsafe_ip",
    action="store_true",
    required=False,
    help="Set to true if you want this bridge worker to allow img2img requests from unsafe IPs.",
)
arg_parser.add_argument(
    "-m",
    "--model",
    action="store",
    required=False,
    help="Which model to run on this horde.",
)
arg_parser.add_argument("--debug", action="store_true", default=False, help="Show debugging messages.")
arg_parser.add_argument(
    "-v",
    "--verbosity",
    action="count",
    default=0,
    help=(
        "The default logging level is ERROR or higher. "
        "This value increases the amount of logging seen in your screen"
    ),
)
arg_parser.add_argument(
    "-q",
    "--quiet",
    action="count",
    default=0,
    help=(
        "The default logging level is ERROR or higher. "
        "This value decreases the amount of logging seen in your screen"
    ),
)
arg_parser.add_argument(
    "--log_file",
    action="store_true",
    default=False,
    help="If specified will dump the log to the specified file",
)
arg_parser.add_argument(
    "--skip_md5",
    action="store_true",
    default=False,
    help="If specified will not check the downloaded model md5sum.",
)
arg_parser.add_argument(
    "--disable_voodoo",
    action="store_true",
    default=False,
    help=(
        "If specified this bridge will not use voodooray to offload models into RAM and save VRAM"
        " (useful for cloud providers)."
    ),
)
arg_parser.add_argument(
    "--disable_xformers",
    action="store_true",
    default=False,
    help=(
        "If specified this bridge will not try use xformers to speed up generations."
        " This should normally be automatic, but in case you need to disable it manually, you can do so here."
    ),
)
args = arg_parser.parse_args()

disable_xformers.toggle(args.disable_xformers)
disable_voodoo.toggle(args.disable_voodoo)

# Note: for now we cannot put them at the top of the file because the imports
# will use the disable_voodoo and disable_xformers global variables
from nataili.inference.compvis.img2img import img2img  # noqa: E402
from nataili.inference.compvis.txt2img import txt2img  # noqa: E402
from nataili.inference.diffusers.inpainting import inpainting  # noqa: E402
from nataili.model_manager import ModelManager  # noqa: E402

model = ""
max_content_length = 1024
max_length = 80
current_softprompt = None
softprompts = {}


class BridgeData(object):
    def __init__(self):
        random.seed()
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
        self.max_power = int(os.environ.get("HORDE_MAX_POWER", 8))
        self.nsfw = os.environ.get("HORDE_NSFW", "true") == "true"
        self.censor_nsfw = os.environ.get("HORDE_CENSOR", "false") == "true"
        self.blacklist = list(filter(lambda a: a, os.environ.get("HORDE_BLACKLIST", "").split(",")))
        self.censorlist = list(filter(lambda a: a, os.environ.get("HORDE_CENSORLIST", "").split(",")))
        self.allow_img2img = os.environ.get("HORDE_IMG2IMG", "true") == "true"
        self.allow_painting = os.environ.get("HORDE_PAINTING", "true") == "true"
        self.allow_unsafe_ip = os.environ.get("HORDE_ALLOW_UNSAFE_IP", "true") == "true"
        self.model_names = os.environ.get("HORDE_MODELNAMES", "stable_diffusion").split(",")
        self.max_pixels = 64 * 64 * 8 * self.max_power
        self.initialized = False

    @logger.catch(reraise=True)
    def reload_data(self):
        previous_url = self.horde_url
        previous_api_key = self.api_key
        try:
            import bridgeData as bd
            importlib.reload(bd)
            self.api_key = bd.api_key
            self.worker_name = bd.worker_name
            self.horde_url = bd.horde_url
            self.priority_usernames = bd.priority_usernames
            self.max_power = bd.max_power
            self.model_names = bd.models_to_load
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
                self.allow_unsafe_ip = bd.allow_unsafe_ip
            except AttributeError:
                pass
        except (ImportError, AttributeError):
            logger.warning("bridgeData.py could not be loaded. Using defaults with anonymous account")
        if args.api_key:
            self.api_key = args.api_key
        if args.worker_name:
            self.worker_name = args.worker_name
        if args.horde_url:
            self.horde_url = args.horde_url
        if args.priority_usernames:
            self.priority_usernames = args.priority_usernames
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
        if args.allow_unsafe_ip:
            self.allow_unsafe_ip = args.allow_unsafe_ip
        if self.max_power < 2:
            self.max_power = 2
        self.max_pixels = 64 * 64 * 8 * self.max_power
        if self.censor_nsfw or len(self.censorlist):
            self.model_names.append("safety_checker")
        if not self.initialized or previous_api_key != self.api_key:
            try:
                user_req = requests.get(
                    self.horde_url + "/api/v2/find_user",
                    headers={"apikey": self.api_key},
                    timeout=10,
                )
                user_req = user_req.json()
                self.username = user_req["username"]
            except:
                logger.warning(f"Server {self.horde_url} error during find_user. Setting username 'N/A'")
                self.username = "N/A"
        if not self.initialized or previous_url != self.horde_url:
            logger.init(
                (
                    f"Username '{self.username}'. Server Name '{self.worker_name}'. "
                    f"Horde URL '{self.horde_url}'. Max Pixels {self.max_pixels}"
                ),
                status="Joining Horde",
            )

    @logger.catch(reraise=True)
    def check_models(self,mm):
        if not self.initialized:
            logger.init("Models", status="Checking")
        models_exist = True
        not_found_models = []
        for model in self.model_names:
            model_info = mm.get_model(model)
            if not model_info:
                logger.error(
                    f"Model name requested {model} in bridgeData is unknown to us. "
                    "Please check your configuration. Aborting!"
                )
                sys.exit(1)
            if model in mm.get_loaded_models_names():
                continue
            if not args.skip_md5 and not mm.validate_model(model):
                models_exist = False
                not_found_models.append(model)
            # Diffusers library uses its own internal download mechanism
            if model_info["type"] == "diffusers" and model_info["hf_auth"]:
                check_mm_auth(mm)
        if not models_exist:
            choice = input(
                "You do not appear to have downloaded the models needed yet.\nYou need at least a main model to proceed. "
                f"Would you like to download your prespecified models?\n\
            y: Download {not_found_models} (default).\n\
            n: Abort and exit\n\
            all: Download all models (This can take a significant amount of time and bandwidth)?\n\
            Please select an option: "
            )
            if choice not in ["y", "Y", "", "yes", "all", "a"]:
                sys.exit(1)
            needs_hf = False
            for model in not_found_models:
                dl = mm.get_model_download(model)
                for m in dl:
                    if m.get("hf_auth", False):
                        needs_hf = True
            if choice in ["all", "a"]:
                needs_hf = True
            if needs_hf:
                check_mm_auth(mm)
            mm.init()
            mm.taint_models(not_found_models)
            if choice in ["all", "a"]:
                mm.download_all()
            elif choice in ["y", "Y", "", "yes"]:
                for model in not_found_models:
                    logger.init(f"Model: {model}", status="Downloading")
                    if not mm.download_model(model):
                        logger.message(
                            "Something went wrong when downloading the model and it does not fit the expected checksum. "
                            "Please check that your HuggingFace authentication is correct and that you've accepted the "
                            "model license from the browser."
                        )
                        sys.exit(1)
            mm.init()
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
                "bridgeData.py created. Bridge will exit. Please edit bridgeData.py with your setup and restart the bridge"
            )
            sys.exit(2)

    def reload_models(self, mm):
        for model in mm.get_loaded_models_names():
            if model not in self.model_names:
                logger.init(f"{model}", status="Unloading")
                mm.unload_model(model)
        for model in self.model_names:
            if model not in mm.get_loaded_models_names():
                logger.init(f"{model}", status="Loading")
                success = mm.load_model(model)
                if success:
                    logger.init_ok(f"{model}", status="Loaded")
                else:
                    logger.init_err(f"{model}", status="Error")
        self.initialized = True


@logger.catch(reraise=True)
def bridge(interval, model_manager, bd):
    current_id = None
    current_payload = None
    loop_retry = 0
    while True:
        bd.reload_data()
        bd.check_models(model_manager)
        bd.reload_models(model_manager)
        # Pop new request from the Horde
        if loop_retry > 10 and current_id:
            logger.error(f"Exceeded retry count {loop_retry} for generation id {current_id}. Aborting generation!")
            current_id = None
            current_payload = None
            current_generation = None
            loop_retry = 0
        elif current_id:
            logger.debug(f"Retrying ({loop_retry}/10) for generation id {current_id}...")
        available_models = model_manager.get_loaded_models_names()
        if "LDSR" in available_models:
            logger.warning("LDSR is an upscaler and doesn't belond in the model list. Ignoring")
            available_models.remove("LDSR")
        if "safety_checker" in available_models:
            available_models.remove("safety_checker")
        gen_dict = {
            "name": bd.worker_name,
            "max_pixels": bd.max_pixels,
            "priority_usernames": bd.priority_usernames,
            "nsfw": bd.nsfw,
            "blacklist": bd.blacklist,
            "models": available_models,
            "allow_img2img": bd.allow_img2img,
            "allow_painting": bd.allow_painting,
            "allow_unsafe_ip": bd.allow_unsafe_ip,
            "bridge_version": 4,
        }
        # logger.debug(gen_dict)
        headers = {"apikey": bd.api_key}
        if current_id:
            loop_retry += 1
        else:
            try:
                pop_req = requests.post(
                    bd.horde_url + "/api/v2/generate/pop",
                    json=gen_dict,
                    headers=headers,
                    timeout=10,
                )
            except requests.exceptions.ConnectionError:
                logger.warning(f"Server {bd.horde_url} unavailable during pop. Waiting 10 seconds...")
                time.sleep(10)
                continue
            except TypeError:
                logger.warning(f"Server {bd.horde_url} unavailable during pop. Waiting 2 seconds...")
                time.sleep(2)
                continue
            except requests.exceptions.ReadTimeout:
                logger.warning(f"Server {bd.horde_url} timed out during pop. Waiting 2 seconds...")
                time.sleep(2)
                continue
            try:
                pop = pop_req.json()
            except json.decoder.JSONDecodeError:
                logger.error(f"Could not decode response from {bd.horde_url} as json. Please inform its administrator!")
                time.sleep(interval)
                continue
            if pop is None:
                logger.error(f"Something has gone wrong with {bd.horde_url}. Please inform its administrator!")
                time.sleep(interval)
                continue
            if not pop_req.ok:
                logger.warning(
                    f"During gen pop, server {bd.horde_url} responded with status code {pop_req.status_code}: "
                    f"{pop['message']}. Waiting for 10 seconds..."
                )
                if "errors" in pop:
                    logger.warning(f"Detailed Request Errors: {pop['errors']}")
                time.sleep(10)
                continue
            if not pop.get("id"):
                skipped_info = pop.get("skipped")
                if skipped_info and len(skipped_info):
                    skipped_info = f" Skipped Info: {skipped_info}."
                else:
                    skipped_info = ""
                logger.debug(f"Server {bd.horde_url} has no valid generations to do for us.{skipped_info}")
                time.sleep(interval)
                continue
            current_id = pop["id"]
            current_payload = pop["payload"]
        # Generate Image
        model = pop.get("model", available_models[0])
        # logger.info([current_id,current_payload])
        use_nsfw_censor = current_payload.get("use_nsfw_censor", False)
        if bd.censor_nsfw and not bd.nsfw:
            use_nsfw_censor = True
        elif any(word in current_payload["prompt"] for word in bd.censorlist):
            use_nsfw_censor = True
        # use_gfpgan = current_payload.get("use_gfpgan", True)
        # use_real_esrgan = current_payload.get("use_real_esrgan", False)
        source_processing = pop.get("source_processing")
        source_image = pop.get("source_image")
        source_mask = pop.get("source_mask")
        # These params will always exist in the payload from the horde
        gen_payload = {
            "prompt": current_payload["prompt"],
            "height": current_payload["height"],
            "width": current_payload["width"],
            "seed": current_payload["seed"],
            "n_iter": 1,
            "batch_size": 1,
            "save_individual_images": False,
            "save_grid": False,
        }
        # These params might not always exist in the horde payload
        if "ddim_steps" in current_payload:
            gen_payload["ddim_steps"] = current_payload["ddim_steps"]
        if "sampler_name" in current_payload:
            gen_payload["sampler_name"] = current_payload["sampler_name"]
        if "cfg_scale" in current_payload:
            gen_payload["cfg_scale"] = current_payload["cfg_scale"]
        if "ddim_eta" in current_payload:
            gen_payload["ddim_eta"] = current_payload["ddim_eta"]
        if "denoising_strength" in current_payload and source_image:
            gen_payload["denoising_strength"] = current_payload["denoising_strength"]
        # logger.debug(gen_payload)
        req_type = "txt2img"
        if source_image:
            img_source = None
            img_mask = None
            if source_processing == "img2img":
                req_type = "img2img"
            elif source_processing == "inpainting":
                req_type = "inpainting"
            if source_processing == "outpainting":
                req_type = "outpainting"
        # Prevent inpainting from picking text2img and img2img gens (as those go via compvis pipelines)
        if model == "stable_diffusion_inpainting" and req_type not in [
            "inpainting",
            "outpainting",
        ]:
            # Try to find any other model to do text2img or img2img
            for m in available_models:
                if m != "stable_diffusion_inpainting":
                    model = m
            # if the model persists as inpainting for text2img or img2img, we abort.
            if model == "stable_diffusion_inpainting":
                # We remove the base64 from the prompt to avoid flooding the output on the error
                if len(pop.get("source_image", "")) > 10:
                    pop["source_image"] = len(pop.get("source_image", ""))
                if len(pop.get("source_mask", "")) > 10:
                    pop["source_mask"] = len(pop.get("source_mask", ""))
                logger.error(
                    "Received an non-inpainting request for inpainting model. This shouldn't happen. "
                    f"Inform the developer. Current payload {pop}"
                )
                current_id = None
                current_payload = None
                current_generation = None
                loop_retry = 0
                continue
                # TODO: Send faulted
        logger.debug(f"{req_type} ({model}) request with id {current_id} picked up. Initiating work...")
        try:
            safety_checker = (
                model_manager.loaded_models["safety_checker"]["model"]
                if "safety_checker" in model_manager.loaded_models
                else None
            )
            if source_image:
                base64_bytes = source_image.encode("utf-8")
                img_bytes = base64.b64decode(base64_bytes)
                img_source = Image.open(BytesIO(img_bytes))
            if source_mask:
                base64_bytes = source_mask.encode("utf-8")
                img_bytes = base64.b64decode(base64_bytes)
                img_mask = Image.open(BytesIO(img_bytes))
                if img_mask.size != img_source.size:
                    logger.warning(
                        f"Source image/mask mismatch. Resizing mask from {img_mask.size} to {img_source.size}"
                    )
                    img_mask = img_mask.resize(img_source.size)
            if req_type == "img2img":
                gen_payload["init_img"] = img_source
                generator = img2img(
                    model_manager.loaded_models[model]["model"],
                    model_manager.loaded_models[model]["device"],
                    "bridge_generations",
                    load_concepts=True,
                    concepts_dir="models/custom/sd-concepts-library",
                    safety_checker=safety_checker,
                    filter_nsfw=use_nsfw_censor,
                    disable_voodoo=disable_voodoo.active,
                )
            elif req_type == "inpainting" or req_type == "outpainting":
                # These variables do not exist in the outpainting implementation
                if "save_grid" in gen_payload:
                    del gen_payload["save_grid"]
                if "sampler_name" in gen_payload:
                    del gen_payload["sampler_name"]
                if "denoising_strength" in gen_payload:
                    del gen_payload["denoising_strength"]
                # We prevent sending an inpainting without mask or transparency, as it will crash us.
                if img_mask is None:
                    try:
                        red, green, blue, alpha = img_source.split()
                    except ValueError:
                        logger.warning("inpainting image doesn't have an alpha channel. Aborting gen")
                        current_id = None
                        current_payload = None
                        current_generation = None
                        loop_retry = 0
                        continue
                        # TODO: Send faulted

                gen_payload["inpaint_img"] = img_source

                if img_mask:
                    gen_payload["inpaint_mask"] = img_mask
                generator = inpainting(
                    model_manager.loaded_models[model]["model"],
                    model_manager.loaded_models[model]["device"],
                    "bridge_generations",
                    filter_nsfw=use_nsfw_censor,
                )
            else:
                generator = txt2img(
                    model_manager.loaded_models[model]["model"],
                    model_manager.loaded_models[model]["device"],
                    "bridge_generations",
                    load_concepts=True,
                    concepts_dir="models/custom/sd-concepts-library",
                    safety_checker=safety_checker,
                    filter_nsfw=use_nsfw_censor,
                    disable_voodoo=disable_voodoo.active,
                )
        except KeyError:
            continue
        # If the received image is unreadable, we continue
        except UnidentifiedImageError:
            logger.error("Source image received for img2img is unreadable. Falling back to text2img!")
            if "denoising_strength" in gen_payload:
                del gen_payload["denoising_strength"]
            generator = txt2img(
                model_manager.loaded_models[model]["model"],
                model_manager.loaded_models[model]["device"],
                "bridge_generations",
                load_concepts=True,
                concepts_dir="models/custom/sd-concepts-library",
            )
        except binascii.Error:
            logger.error(
                "Source image received for img2img is cannot be base64 decoded (binascii.Error). "
                "Falling back to text2img!"
            )
            if "denoising_strength" in gen_payload:
                del gen_payload["denoising_strength"]
            generator = txt2img(
                model_manager.loaded_models[model]["model"],
                model_manager.loaded_models[model]["device"],
                "bridge_generations",
                load_concepts=True,
                concepts_dir="models/custom/sd-concepts-library",
            )
        generator.generate(**gen_payload)
        torch_gc()
        # Submit back to horde
        # images, seed, info, stats = txt2img(**current_payload)
        buffer = BytesIO()
        # We send as WebP to avoid using all the horde bandwidth
        image = generator.images[0]["image"]
        seed = generator.images[0]["seed"]
        image.save(buffer, format="WebP", quality=90)
        # logger.info(info)
        submit_dict = {
            "id": current_id,
            "generation": base64.b64encode(buffer.getvalue()).decode("utf8"),
            "api_key": bd.api_key,
            "seed": seed,
            "max_pixels": bd.max_pixels,
        }
        current_generation = seed
        while current_id and current_generation is not None:
            try:
                submit_req = requests.post(
                    bd.horde_url + "/api/v2/generate/submit",
                    json=submit_dict,
                    headers=headers,
                    timeout=20,
                )
                try:
                    submit = submit_req.json()
                except json.decoder.JSONDecodeError:
                    logger.error(
                        f"Something has gone wrong with {bd.horde_url} during submit. "
                        f"Please inform its administrator!  (Retry {loop_retry}/10)"
                    )
                    time.sleep(interval)
                    continue
                if submit_req.status_code == 404:
                    logger.warning("The generation we were working on got stale. Aborting!")
                elif not submit_req.ok:
                    logger.warning(
                        f"During gen submit, server {bd.horde_url} responded with status code {submit_req.status_code}: "
                        f"{submit['message']}. Waiting for 10 seconds...  (Retry {loop_retry}/10)"
                    )
                    if "errors" in submit:
                        logger.warning(f"Detailed Request Errors: {submit['errors']}")
                    time.sleep(10)
                    continue
                else:
                    logger.info(
                        f'Submitted generation with id {current_id} and contributed for {submit_req.json()["reward"]}'
                    )
                current_id = None
                current_payload = None
                current_generation = None
                loop_retry = 0
            except requests.exceptions.ConnectionError:
                logger.warning(
                    f"Server {bd.horde_url} unavailable during submit. Waiting 10 seconds...  (Retry {loop_retry}/10)"
                )
                time.sleep(10)
                continue
            except requests.exceptions.ReadTimeout:
                logger.warning(
                    f"Server {bd.horde_url} timed out during submit. Waiting 10 seconds...  (Retry {loop_retry}/10)"
                )
                time.sleep(10)
                continue
        time.sleep(interval)


def check_mm_auth(model_manager):
    if model_manager.has_authentication():
        return
    try:
        from creds import hf_password, hf_username
    except ImportError:
        hf_username = input("Please type your huggingface.co username: ")
        hf_password = getpass.getpass("Please type your huggingface.co Access Token or password: ")
    hf_auth = {"username": hf_username, "password": hf_password}
    model_manager.set_authentication(hf_auth=hf_auth)


if __name__ == "__main__":

    set_logger_verbosity(args.verbosity)
    if args.log_file:
        logger.add("koboldai_bridge_log.log", retention="7 days", level="warning")  # Automatically rotate too big file
    quiesce_logger(args.quiet)
    model_manager = ModelManager(disable_voodoo=disable_voodoo.active)
    model_manager.init()
    bridge_data = BridgeData()
    try:
        bridge(args.interval, model_manager, bridge_data)
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Received. Ending Process")
    logger.init(f"{bridge_data.worker_name} Instance", status="Stopped")
