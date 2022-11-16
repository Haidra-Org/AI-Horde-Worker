import hashlib
import json
import os
import shutil
import zipfile

import clip
import git
import open_clip
import requests
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from diffusers import StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from gfpgan import GFPGANer
from omegaconf import OmegaConf
from realesrgan import RealESRGANer
from tqdm import tqdm
from transformers import logging

from ldm.models.blip import blip_decoder
from ldm.util import instantiate_from_config
from nataili.inference.aitemplate.ait_pipeline import StableDiffusionAITPipeline

try:
    from nataili.util.voodoo import init_ait_module, push_model_to_plasma
except ModuleNotFoundError as e:
    from nataili import disable_voodoo

    if not disable_voodoo.active:
        raise e

from nataili.util import logger
from nataili.util.cache import torch_gc
from nataili.util.load_list import load_list

logging.set_verbosity_error()

models = json.load(open("./db.json"))
aitemplate = json.load(open("./aitemplate.json"))
dependencies = json.load(open("./db_dep.json"))
remote_models = "https://raw.githubusercontent.com/Sygil-Dev/nataili-model-reference/main/db.json"
remote_dependencies = "https://raw.githubusercontent.com/Sygil-Dev/nataili-model-reference/main/db_dep.json"


class ModelManager:
    def __init__(self, hf_auth=None, download=True, disable_voodoo=True):
        if download:
            try:
                logger.init("Model Reference", status="Downloading")
                r = requests.get(remote_models)
                self.models = r.json()
                r = requests.get(remote_dependencies)
                self.dependencies = json.load(open("./db_dep.json"))
                logger.init_ok("Model Reference", status="OK")
                self.aitemplates = json.load(open("./aitemplate.json"))
            except Exception:
                logger.init_err("Model Reference", status="Download Error")
                self.models = json.load(open("./db.json"))
                self.dependencies = json.load(open("./db_dep.json"))
                logger.init_warn("Model Reference", status="Local")

        else:
            self.models = json.load(open("./db.json"))
            self.dependencies = json.load(open("./db_dep.json"))
            self.aitemplates = json.load(open("./aitemplate.json"))
        self.available_models = []
        self.available_aitemplates = []
        self.tainted_models = []
        self.available_dependencies = []
        self.loaded_models = {}
        self.hf_auth = None
        self.set_authentication(hf_auth)
        self.disable_voodoo = disable_voodoo
        self.cuda_devices, self.recommended_gpu = self.detect_available_cuda_arch()
        self.ait_workdir = "./"

    def detect_available_cuda_arch(self):
        # get nvidia sm_xx version
        # get count of cuda devices
        # get compute capability of each device (sm_xx)
        if torch.cuda.is_available():
            number_of_cuda_devices = torch.cuda.device_count()
            cuda_arch = []
            for i in range(number_of_cuda_devices):
                cuda_device = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "sm": torch.cuda.get_device_capability(i)[0] * 10 + torch.cuda.get_device_capability(i)[1],
                }
                cuda_arch.append(cuda_device)
            # sort by sm desc
            cuda_arch = sorted(cuda_arch, key=lambda k: k["sm"], reverse=True)
            # recommended gpu = all gpu with highest sm
            recommended_gpu = [x for x in cuda_arch if x["sm"] == cuda_arch[0]["sm"]]
            return cuda_arch, recommended_gpu
        else:
            return None

    def get_ait_workdir(self, cuda_arch, model_name="stable_diffusion"):
        if cuda_arch == 89:
            return f"./{self.aitemplates[model_name]['config']['sm89']['download'][0]['file_path']}/"
        elif cuda_arch >= 80 and cuda_arch < 89:
            return f"./{self.aitemplates[model_name]['config']['sm80']['download'][0]['file_path']}/"
        elif cuda_arch == 75:
            return f"./{self.aitemplates[model_name]['config']['sm75']['download'][0]['file_path']}/"
        elif cuda_arch == 70:
            return f"./{self.aitemplates[model_name]['config']['sm70']['download'][0]['file_path']}/"
        else:
            raise ValueError("CUDA Compute Capability not supported")

    def init(self):
        dependencies_available = []
        for dependency in self.dependencies:
            if self.check_available(self.get_dependency_files(dependency)):
                dependencies_available.append(dependency)
        self.available_dependencies = dependencies_available

        models_available = []
        for model in self.models:
            if self.check_available(self.get_model_files(model)):
                models_available.append(model)
        self.available_models = models_available

        logger.info(f"Highest CUDA Compute Capability: {self.cuda_devices[0]['sm']}")
        logger.debug(f"Available CUDA Devices: {self.cuda_devices}")
        logger.info(f"Recommended GPU: {self.recommended_gpu}")
        sm = self.recommended_gpu[0]["sm"]
        logger.info(f"Using sm_{sm} for AITemplate")
        aitemplate_available = []
        for aitemplate in self.aitemplates:
            logger.info(f"{aitemplate}")
            if self.check_available(self.get_aitemplate_files(sm)):
                aitemplate_available.append(aitemplate)
        self.available_aitemplates = aitemplate_available
        if len(self.available_aitemplates) == 0:
            logger.debug("No AITemplate available")
        else:
            self.ait_workdir = self.get_ait_workdir(sm)

        if self.hf_auth is not None:
            if "username" not in self.hf_auth and "password" not in self.hf_auth:
                raise ValueError("hf_auth must contain username and password")
            else:
                if self.hf_auth["username"] == "" or self.hf_auth["password"] == "":
                    raise ValueError("hf_auth must contain username and password")
        return True

    def set_authentication(self, hf_auth=None):
        # We do not let No authentication override previously set auth
        if not hf_auth and self.hf_auth:
            return
        self.hf_auth = hf_auth
        if hf_auth:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_auth.get("password")

    def has_authentication(self):
        if self.hf_auth:
            return True
        return False

    def get_model(self, model_name):
        return self.models.get(model_name)

    def get_filtered_models(self, **kwargs):
        """Get all model names.
        Can filter based on metadata of the model reference db
        """
        filtered_models = self.models
        for keyword in kwargs:
            iterating_models = filtered_models.copy()
            filtered_models = {}
            for model in iterating_models:
                # logger.debug([keyword,iterating_models[model].get(keyword),kwargs[keyword]])
                if iterating_models[model].get(keyword) == kwargs[keyword]:
                    filtered_models[model] = iterating_models[model]
        return filtered_models

    def get_filtered_model_names(self, **kwargs):
        filtered_models = self.get_filtered_models(**kwargs)
        return list(filtered_models.keys())

    def get_dependency(self, dependency_name):
        return self.dependencies[dependency_name]

    def get_model_files(self, model_name):
        if self.models[model_name]["type"] == "diffusers":
            return []
        return self.models[model_name]["config"]["files"]

    def get_aitemplate_files(self, cuda_arch, model_name="stable_diffusion"):
        if cuda_arch == 89:
            return self.aitemplates[model_name]["config"]["sm89"]["files"]
        elif cuda_arch >= 80 and cuda_arch < 89:
            return self.aitemplates[model_name]["config"]["sm80"]["files"]
        elif cuda_arch == 75:
            return self.aitemplates[model_name]["config"]["sm75"]["files"]
        elif cuda_arch == 70:
            raise ValueError("CUDA Compute Capability not supported")
            # return self.aitemplates[model_name]['config']['sm70']['files']
        else:
            raise ValueError("CUDA Compute Capability not supported")

    def get_aitemplate_download(self, cuda_arch, model_name="stable_diffusion"):
        if cuda_arch == 89:
            return self.aitemplates[model_name]["config"]["sm89"]["download"]
        elif cuda_arch >= 80 and cuda_arch < 89:
            return self.aitemplates[model_name]["config"]["sm80"]["download"]
        elif cuda_arch == 75:
            return self.aitemplates[model_name]["config"]["sm75"]["download"]
        elif cuda_arch == 70:
            raise ValueError("CUDA Compute Capability not supported")
            # return self.aitemplates[model_name]['config']['sm70']['download']
        else:
            raise ValueError("CUDA Compute Capability not supported")

    def get_dependency_files(self, dependency_name):
        return self.dependencies[dependency_name]["config"]["files"]

    def get_model_download(self, model_name):
        return self.models[model_name]["config"]["download"]

    def get_dependency_download(self, dependency_name):
        return self.dependencies[dependency_name]["config"]["download"]

    def get_available_models(self):
        return self.available_models

    def get_available_dependencies(self):
        return self.available_dependencies

    def get_loaded_models(self):
        return self.loaded_models

    def get_loaded_models_names(self):
        return list(self.loaded_models.keys())

    def get_loaded_model(self, model_name):
        return self.loaded_models[model_name]

    def unload_model(self, model_name):
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            return True
        return False

    def unload_all_models(self):
        for model in self.loaded_models:
            del self.loaded_models[model]
        return True

    def taint_model(self, model_name):
        """Marks a model as not valid by remiving it from available_models"""
        if model_name in self.available_models:
            self.available_models.remove(model_name)
            self.tainted_models.append(model_name)

    def taint_models(self, models):
        for model in models:
            self.taint_model(model)

    def load_model_from_config(self, model_path="", config_path="", map_location="cpu"):
        config = OmegaConf.load(config_path)
        pl_sd = torch.load(model_path, map_location=map_location)
        if "global_step" in pl_sd:
            logger.info(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        model = model.eval()
        del pl_sd, sd, m, u
        return model

    def load_ckpt(self, model_name="", precision="half", gpu_id=0):
        ckpt_path = self.get_model_files(model_name)[0]["path"]
        config_path = self.get_model_files(model_name)[1]["path"]
        device = torch.device(f"cuda:{gpu_id}")
        if not self.disable_voodoo:
            model = self.load_model_from_config(model_path=ckpt_path, config_path=config_path)
            model = model if precision == "full" else model.half()
            logger.debug(f"Doing voodoo on {model_name}")
            model = push_model_to_plasma(model) if isinstance(model, torch.nn.Module) else model
        else:
            model = self.load_model_from_config(
                model_path=ckpt_path, config_path=config_path, map_location=f"cuda:{gpu_id}"
            )
            model = (model if precision == "full" else model.half()).to(device, memory_format=torch.channels_last)
        torch_gc()
        return {"model": model, "device": device}

    def load_realesrgan(self, model_name="", precision="half", gpu_id=0):

        RealESRGAN_models = {
            "RealESRGAN_x4plus": RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            ),
            "RealESRGAN_x4plus_anime_6B": RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=4,
            ),
        }

        model_path = self.get_model_files(model_name)[0]["path"]
        device = "cuda"
        model = RealESRGANer(
            scale=2,
            model_path=model_path,
            model=RealESRGAN_models[models[model_name]["name"]],
            pre_pad=0,
            half=True if precision == "half" else False,
            device=device,
            gpu_id=gpu_id,
        )
        return {"model": model, "device": device}

    def load_gfpgan(self, model_name="", gpu_id=0):

        model_path = self.get_model_files(model_name)[0]["path"]
        device = torch.device(f"cuda:{gpu_id}")
        model = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )
        return {"model": model, "device": device}

    def load_blip(
        self,
        model_name="",
        precision="half",
        gpu_id=0,
        blip_image_eval_size=512,
        vit="large",
    ):
        # vit = 'base' or 'large'
        vit = "base" if model_name == "BLIP" else "large"
        model_path = self.get_model_files(model_name)[0]["path"]
        device = torch.device(f"cuda:{gpu_id}")
        model = blip_decoder(
            pretrained=model_path,
            med_config="configs/blip/med_config.json",
            image_size=blip_image_eval_size,
            vit=vit,
        )
        model = model.eval()
        model = (model if precision == "full" else model.half()).to(device)
        return {"model": model, "device": device}

    def load_data_lists(self, data_path="data/img2txt"):
        data_lists = {}
        data_lists["artists"] = load_list(os.path.join(data_path, "artists.txt"))
        data_lists["flavors"] = load_list(os.path.join(data_path, "flavors.txt"))
        data_lists["mediums"] = load_list(os.path.join(data_path, "mediums.txt"))
        data_lists["movements"] = load_list(os.path.join(data_path, "movements.txt"))
        data_lists["sites"] = load_list(os.path.join(data_path, "sites.txt"))
        data_lists["techniques"] = load_list(os.path.join(data_path, "techniques.txt"))
        data_lists["tags"] = load_list(os.path.join(data_path, "tags.txt"))
        return data_lists

    def load_open_clip(self, model_name="", precision="half", gpu_id=0, data_path="data/img2txt"):
        pretrained = self.get_model(model_name)["pretrained_name"]
        device = torch.device(f"cuda:{gpu_id}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir="models/clip"
        )
        model = model.eval()
        model = (model if precision == "full" else model.half()).to(device)
        data_lists = self.load_data_lists(data_path=data_path)
        return {"model": model, "device": device, "preprocess": preprocess, "data_lists": data_lists}

    def load_clip(self, model_name="", precision="half", gpu_id=0, data_path="data/img2txt"):
        device = torch.device(f"cuda:{gpu_id}")
        model, preprocess = clip.load(model_name, device=device, download_root="models/clip")
        model = model.eval()
        model = (model if precision == "full" else model.half()).to(device)
        data_lists = self.load_data_lists(data_path=data_path)
        return {"model": model, "device": device, "preprocess": preprocess, "data_lists": data_lists}

    def load_diffuser(self, model_name=""):
        model_path = self.models[model_name]["hf_path"]
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=self.models[model_name]["hf_auth"],
        )

        pipe.enable_attention_slicing()
        pipe.to("cuda")
        return {"model": pipe, "device": "cuda"}

    def load_ait(self):
        self.loaded_models["ait"] = {}
        self.loaded_models["ait"]["unet"] = init_ait_module("unet.so", self.ait_workdir)
        self.loaded_models["ait"]["clip"] = init_ait_module("clip.so", self.ait_workdir)
        self.loaded_models["ait"]["vae"] = init_ait_module("vae.so", self.ait_workdir)
        self.loaded_models["ait"]["pipe"] = StableDiffusionAITPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=True if self.hf_auth is None else self.hf_auth["password"],
            clip_ait_exe=self.loaded_models["ait"]["clip"],
            unet_ait_exe=self.loaded_models["ait"]["unet"],
            vae_ait_exe=self.loaded_models["ait"]["vae"],
            filter_nsfw=False,
        ).to("cuda")
        return True

    def load_model(self, model_name="", precision="half", gpu_id=0, data_path="data/img2txt"):
        if model_name not in self.available_models:
            return False
        if self.models[model_name]["type"] == "ckpt":
            self.loaded_models[model_name] = self.load_ckpt(model_name, precision, gpu_id)
            return True
        elif self.models[model_name]["type"] == "realesrgan":
            self.loaded_models[model_name] = self.load_realesrgan(model_name, precision, gpu_id)
            return True
        elif self.models[model_name]["type"] == "gfpgan":
            self.loaded_models[model_name] = self.load_gfpgan(model_name, gpu_id)
            return True
        elif self.models[model_name]["type"] == "blip":
            self.loaded_models[model_name] = self.load_blip(model_name, precision, gpu_id, 512)
            return True
        elif self.models[model_name]["type"] == "open_clip":
            self.loaded_models[model_name] = self.load_open_clip(model_name, precision, gpu_id, data_path)
            return True
        elif self.models[model_name]["type"] == "clip":
            self.loaded_models[model_name] = self.load_clip(model_name, precision, gpu_id, data_path)
            return True
        elif self.models[model_name]["type"] == "diffusers":
            self.loaded_models[model_name] = self.load_diffuser(model_name)
            return True
        elif self.models[model_name]["type"] == "safety_checker":
            self.loaded_models[model_name] = self.load_safety_checker(model_name, gpu_id)
            return True
        else:
            return False

    def load_safety_checker(self, model_name="", gpu_id=0):
        model_path = os.path.dirname(self.get_model_files(model_name)[0]["path"])
        device = torch.device(f"cuda:{gpu_id}")
        model = StableDiffusionSafetyChecker.from_pretrained(model_path)
        model = model.eval().to(device)
        return {"model": model, "device": device}

    def validate_model(self, model_name, skip_checksum=False):
        files = self.get_model_files(model_name)
        for file_details in files:
            if not self.check_file_available(file_details["path"]):
                return False
            if not skip_checksum and not self.validate_file(file_details):
                return False
        return True

    def validate_file(self, file_details):
        if "md5sum" in file_details:
            file_name = file_details["path"]
            logger.debug(f"Getting md5sum of {file_name}")
            with open(file_name, "rb") as file_to_check:
                file_hash = hashlib.md5()
                while chunk := file_to_check.read(8192):
                    file_hash.update(chunk)
            if file_details["md5sum"] != file_hash.hexdigest():
                return False
        return True

    def check_file_available(self, file_path):
        return os.path.exists(file_path)

    def check_available(self, files):
        available = True
        for file in files:
            if not self.check_file_available(file["path"]):
                available = False
        return available

    def download_file(self, url, file_path):
        # make directory
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        pbar_desc = file_path.split("/")[-1]
        r = requests.get(url, stream=True, allow_redirects=True)
        with open(file_path, "wb") as f:
            with tqdm(
                # all optional kwargs
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=pbar_desc,
                total=int(r.headers.get("content-length", 0)),
            ) as pbar:
                for chunk in r.iter_content(chunk_size=16 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    def download_ait(self, cuda_arch):
        files = self.get_aitemplate_files(cuda_arch)
        download = self.get_aitemplate_download(cuda_arch)
        for i in range(len(download)):
            file_path = (
                f"{download[i]['file_path']}/{download[i]['file_name']}"
                if "file_path" in download[i]
                else files[i]["path"]
            )

            download_url = download[i]["file_url"]
            if not self.check_file_available(file_path):
                logger.debug(f"Downloading {download_url} to {file_path}")
                self.download_file(download_url, file_path)
        self.ait_workdir = self.get_ait_workdir(cuda_arch)

    def download_model(self, model_name):
        if model_name in self.available_models:
            logger.info(f"{model_name} is already available.")
            return True
        download = self.get_model_download(model_name)
        files = self.get_model_files(model_name)
        for i in range(len(download)):
            file_path = (
                f"{download[i]['file_path']}/{download[i]['file_name']}"
                if "file_path" in download[i]
                else files[i]["path"]
            )

            if "file_url" in download[i]:
                download_url = download[i]["file_url"]
                if "hf_auth" in download[i]:
                    username = self.hf_auth["username"]
                    password = self.hf_auth["password"]
                    download_url = download_url.format(username=username, password=password)
            if "file_name" in download[i]:
                download_name = download[i]["file_name"]
            if "file_path" in download[i]:
                download_path = download[i]["file_path"]

            if "manual" in download[i]:
                logger.warning(
                    f"The model {model_name} requires manual download from {download_url}. "
                    f"Please place it in {download_path}/{download_name} then press ENTER to continue..."
                )
                input("")
                continue
            # TODO: simplify
            if "file_content" in download[i]:
                file_content = download[i]["file_content"]
                logger.info(f"writing {file_content} to {file_path}")
                # make directory download_path
                os.makedirs(download_path, exist_ok=True)
                # write file_content to download_path/download_name
                with open(os.path.join(download_path, download_name), "w") as f:
                    f.write(file_content)
            elif "symlink" in download[i]:
                logger.info(f"symlink {file_path} to {download[i]['symlink']}")
                symlink = download[i]["symlink"]
                # make directory symlink
                os.makedirs(download_path, exist_ok=True)
                # make symlink from download_path/download_name to symlink
                os.symlink(symlink, os.path.join(download_path, download_name))
            elif "git" in download[i]:
                logger.info(f"git clone {download_url} to {file_path}")
                # make directory download_path
                os.makedirs(file_path, exist_ok=True)
                git.Git(file_path).clone(download_url)
                if "post_process" in download[i]:
                    for post_process in download[i]["post_process"]:
                        if "delete" in post_process:
                            # delete folder post_process['delete']
                            logger.info(f"delete {post_process['delete']}")
                            try:
                                shutil.rmtree(post_process["delete"])
                            except PermissionError as e:
                                logger.error(
                                    f"[!] Something went wrong while deleting the `{post_process['delete']}`. "
                                    "Please delete it manually."
                                )
                                logger.error("PermissionError: ", e)
            else:
                if not self.check_file_available(file_path) or model_name in self.tainted_models:
                    logger.debug(f"Downloading {download_url} to {file_path}")
                    self.download_file(download_url, file_path)
        if not self.validate_model(model_name):
            return False
        if model_name in self.tainted_models:
            self.tainted_models.remove(model_name)
        self.init()
        return True

    def download_dependency(self, dependency_name):
        if dependency_name in self.available_dependencies:
            logger.info(f"{dependency_name} is already installed.")
            return True
        download = self.get_dependency_download(dependency_name)
        files = self.get_dependency_files(dependency_name)
        for i in range(len(download)):
            if "git" in download[i]:
                logger.warning("git download not implemented yet")
                break

            file_path = files[i]["path"]
            if "file_url" in download[i]:
                download_url = download[i]["file_url"]
            if "file_name" in download[i]:
                download_name = download[i]["file_name"]
            if "file_path" in download[i]:
                download_path = download[i]["file_path"]
            logger.debug(download_name)
            if "unzip" in download[i]:
                zip_path = f"temp/{download_name}.zip"
                # os dirname zip_path
                # mkdir temp
                os.makedirs("temp", exist_ok=True)

                self.download_file(download_url, zip_path)
                logger.info(f"unzip {zip_path}")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall("temp/")
                # move temp/sd-concepts-library-main/sd-concepts-library to download_path
                logger.info(f"move temp/{download_name}-main/{download_name} to {download_path}")
                shutil.move(f"temp/{download_name}-main/{download_name}", download_path)
                logger.info(f"delete {zip_path}")
                os.remove(zip_path)
                logger.info(f"delete temp/{download_name}-main/")
                shutil.rmtree(f"temp/{download_name}-main")
            else:
                if not self.check_file_available(file_path):
                    logger.init(f"{file_path}", status="Downloading")
                    self.download_file(download_url, file_path)
        self.init()
        return True

    def download_all_models(self):
        for model in self.get_filtered_model_names(download_all=True):
            if not self.check_model_available(model):
                logger.init(f"{model}", status="Downloading")
                self.download_model(model)
            else:
                logger.info(f"{model} is already downloaded.")
        return True

    def download_all_dependencies(self):
        for dependency in self.dependencies:
            if not self.check_dependency_available(dependency):
                logger.init(f"{dependency}", status="Downloading")
                self.download_dependency(dependency)
            else:
                logger.info(f"{dependency} is already installed.")
        return True

    def download_all(self):
        self.download_all_dependencies()
        self.download_all_models()
        return True

    """
    FIXME: this method is present twice, commenting first one...

    def check_all_available(self):
        for model in self.models:
            if not self.check_available(self.get_model_files(model)):
                return False
        for dependency in self.dependencies:
            if not self.check_available(self.get_dependency_files(dependency)):
                return False
        return True
    """

    def check_model_available(self, model_name):
        if model_name not in self.models:
            return False
        return self.check_available(self.get_model_files(model_name))

    def check_dependency_available(self, dependency_name):
        if dependency_name not in self.dependencies:
            return False
        return self.check_available(self.get_dependency_files(dependency_name))

    def check_all_available(self):
        for model in self.models:
            if not self.check_model_available(model):
                return False
        for dependency in self.dependencies:
            if not self.check_dependency_available(dependency):
                return False
        return True
