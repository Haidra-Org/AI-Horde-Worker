
import time
from pathlib import Path

import torch
from diffusers import LMSDiscreteScheduler
from transformers import CLIPFeatureExtractor, CLIPTokenizer

# from nataili.aitemplate import StableDiffusionAITPipeline
from worker.cache import get_cache_directory
from worker.model_manager.base import BaseModelManager
from worker.logger import logger
# from nataili.util.voodoo import init_ait_module


class AITemplateModelManager(BaseModelManager):
    def __init__(self, download_reference=True):
        super().__init__()
        self.download_reference = download_reference
        self.path = f"{get_cache_directory()}/aitemplate"
        self.models_db_name = "aitemplate"
        self.models_path = self.pkg / f"{self.models_db_name}.json"
        self.remote_db = (
            f"https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/{self.models_db_name}.json"
        )
        self.ait_workdir = None
        self.init()

    def init(self):
        if self.cuda_available:
            logger.info(f"Highest CUDA Compute Capability: {self.cuda_devices[0]['sm']}")
            logger.debug(f"Available CUDA Devices: {self.cuda_devices}")
            logger.info(f"Recommended GPU: {self.recommended_gpu}")
            sm = self.recommended_gpu[0]["sm"]
            logger.info(f"Using sm_{sm} for AITemplate")
            for aitemplate in self.models:
                ait_files = self.get_aitemplate_files(sm)
                if len(ait_files) > 0 and self.check_available(ait_files):
                    self.available_models.append(aitemplate)
                    logger.info(f"Available AITemplate: {aitemplate}")
            if len(self.available_models) == 0:
                logger.warning("No AITemplate available")
            else:
                self.ait_workdir = self.get_ait_workdir(sm)

    def download_ait(self, cuda_arch):
        """
        :param cuda_arch: CUDA Compute Capability
        Download AITemplate modules for CUDA Compute Capability
        """
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

    def get_aitemplate_files(self, cuda_arch, model_name="stable_diffusion"):
        """
        :param cuda_arch: CUDA Compute Capability
        :param model_name: AITemplate model name
        :return: AITemplate files for CUDA Compute Capability
        """
        if cuda_arch == 89:
            return self.models[model_name]["config"]["sm89"]["files"]
        elif cuda_arch >= 80 and cuda_arch < 89:
            return self.models[model_name]["config"]["sm80"]["files"]
        elif cuda_arch == 75:
            return self.models[model_name]["config"]["sm75"]["files"]
        else:
            logger.warning("CUDA Compute Capability not supported")
            return []

    def get_aitemplate_download(self, cuda_arch, model_name="stable_diffusion"):
        """
        :param cuda_arch: CUDA Compute Capability
        :param model_name: AITemplate model name
        :return: AITemplate download details for CUDA Compute Capability
        """
        if cuda_arch == 89:
            return self.models[model_name]["config"]["sm89"]["download"]
        elif cuda_arch >= 80 and cuda_arch < 89:
            return self.models[model_name]["config"]["sm80"]["download"]
        elif cuda_arch == 75:
            return self.models[model_name]["config"]["sm75"]["download"]
        else:
            logger.warning("CUDA Compute Capability not supported")
            return []

    def get_ait_workdir(self, cuda_arch, model_name="stable_diffusion"):
        if cuda_arch == 89:
            return f"./{self.models[model_name]['config']['sm89']['download'][0]['file_path']}/"
        elif cuda_arch >= 80 and cuda_arch < 89:
            return f"./{self.models[model_name]['config']['sm80']['download'][0]['file_path']}/"
        elif cuda_arch == 75:
            return f"./{self.models[model_name]['config']['sm75']['download'][0]['file_path']}/"
        else:
            raise ValueError("CUDA Compute Capability not supported")

    def load(
        self,
        model_name: str = "stable_diffusion",
        gpu_id=0,
    ):
        """
        model_name: str. Name of the model to load. See available_models for a list of available models.
        gpu_id: int. The id of the cuda device to use.
        AITemplate requires CUDA.
        """
        if not self.cuda_available:
            logger.warning("AITemplate requires CUDA")
            return False
        if model_name not in self.models:
            logger.error(f"{model_name} not found")
            return False
        if len(self.available_models) == 0:
            logger.info("No available aitemplates")
            sm = self.recommended_gpu[0]["sm"]
            logger.init_ok(f"Downloading AITemplate for {sm}", status="Downloading")
            self.download_ait()
            logger.init_ok(f"AITemplate for {sm} downloaded", status="Downloading")
        if model_name not in self.loaded_models:
            tic = time.time()
            logger.init(f"{model_name}", status="Loading")
            self.loaded_models["ait"] = self.load_aitemplate(
                model_name,
                gpu_id=gpu_id,
            )
            logger.init_ok(f"Loading {model_name}", status="Success")
            toc = time.time()
            logger.init_ok(f"Loading {model_name}: Took {toc-tic} seconds", status="Success")
            return True

    def load_aitemplate(
        self,
        model_name,
        gpu_id=0,
    ):
        model_path = self.get_model_files(model_name)[0]["path"]
        model_path = f"{self.path}/{model_path}"
        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"Loading model {model_name} on {device}")
        logger.info(f"Model path: {model_path}")
        ait = {}
        ait["unet"] = init_ait_module("unet.so", self.ait_workdir)
        ait["clip"] = init_ait_module("clip.so", self.ait_workdir)
        ait["vae"] = init_ait_module("vae.so", self.ait_workdir)
        ait["pipe"] = StableDiffusionAITPipeline(
            vae=None,
            unet=None,
            text_encoder=None,
            tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
            scheduler=LMSDiscreteScheduler.from_pretrained("runwayml/stable-diffusion-v1-5"),
            safety_checker=None,
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14"),
            clip_ait_exe=self.loaded_models["ait"]["clip"],
            unet_ait_exe=self.loaded_models["ait"]["unet"],
            vae_ait_exe=self.loaded_models["ait"]["vae"],
            filter_nsfw=False,
        ).to("cuda")
        return ait
