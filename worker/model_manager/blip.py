import sys
from pathlib import Path

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

import time

import torch

from worker.cache import get_cache_directory
from worker.model_manager.base import BaseModelManager
# from nataili.util.blip import blip_decoder
from worker.logger import logger


class BlipModelManager(BaseModelManager):
    def __init__(self, download_reference=True):
        super().__init__()
        self.download_reference = download_reference
        self.path = f"{get_cache_directory()}/blip"
        self.models_db_name = "blip"
        self.models_path = self.pkg / f"{self.models_db_name}.json"
        self.remote_db = (
            f"https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/{self.models_db_name}.json"
        )
        self.init()

    def load(
        self,
        model_name: str,
        half_precision=True,
        gpu_id=0,
        cpu_only=False,
        blip_image_eval_size=512,
    ):
        """
        model_name: str. Name of the model to load. See available_models for a list of available models.
        half_precision: bool. If True, the model will be loaded in half precision.
        gpu_id: int. The id of the gpu to use. If the gpu is not available, the model will be loaded on the cpu.
        cpu_only: bool. If True, the model will be loaded on the cpu. If True, half_precision will be set to False.
        blip_image_eval_size: int. The size of the image to use for the blip model.
        """
        if model_name not in self.models:
            logger.error(f"{model_name} not found")
            return False
        if model_name not in self.available_models:
            logger.error(f"{model_name} not available")
            logger.init_ok(f"Downloading {model_name}", status="Downloading")
            self.download_model(model_name)
            logger.init_ok(f"{model_name} downloaded", status="Downloading")
        if model_name not in self.loaded_models:
            tic = time.time()
            logger.init(f"{model_name}", status="Loading")
            self.loaded_models[model_name] = self.load_blip(
                model_name,
                half_precision=half_precision,
                gpu_id=gpu_id,
                cpu_only=cpu_only,
                blip_image_eval_size=blip_image_eval_size,
            )
            logger.init_ok(f"Loading {model_name}", status="Success")
            toc = time.time()
            logger.init_ok(f"Loading {model_name}: Took {toc-tic} seconds", status="Success")
            return True

    def load_blip(
        self,
        model_name,
        half_precision=True,
        gpu_id=0,
        cpu_only=False,
        blip_image_eval_size=512,
    ):
        if not self.cuda_available:
            cpu_only = True
        vit = "base" if model_name == "BLIP" else "large"
        model_path = self.get_model_files(model_name)[0]["path"]
        model_path = f"{self.path}/{model_path}"
        if cpu_only:
            device = torch.device("cpu")
            half_precision = False
        else:
            device = torch.device(f"cuda:{gpu_id}" if self.cuda_available else "cpu")
        logger.info(f"Loading model {model_name} on {device}")
        logger.info(f"Model path: {model_path}")
        with importlib_resources.as_file(self.pkg / "med_config.json") as med_config:
            logger.info(f"Med config path: {med_config}")
            model = blip_decoder(
                pretrained=model_path,
                med_config=med_config,
                image_size=blip_image_eval_size,
                vit=vit,
            )
        model = model.eval()
        model.to(device)
        if half_precision:
            model = model.half()
        return {"model": model, "device": device, "half_precision": half_precision}
