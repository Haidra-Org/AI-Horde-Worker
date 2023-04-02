import time
from pathlib import Path

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from worker.cache import get_cache_directory
from worker.model_manager.base import BaseModelManager
from worker.logger import logger


class EsrganModelManager(BaseModelManager):
    def __init__(self, download_reference=True):
        super().__init__()
        self.download_reference = download_reference
        self.path = f"{get_cache_directory()}/esrgan"
        self.models_db_name = "esrgan"
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
    ):
        """
        model_name: str. Name of the model to load. See available_models for a list of available models.
        half_precision: bool. If True, the model will be loaded in half precision.
        gpu_id: int. The id of the gpu to use. If the gpu is not available, the model will be loaded on the cpu.
        cpu_only: bool. If True, the model will be loaded on the cpu. If True, half_precision will be set to False.
        """
        if not self.cuda_available:
            cpu_only = True
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
            self.loaded_models[model_name] = self.load_esrgan(
                model_name,
                half_precision=half_precision,
                gpu_id=gpu_id,
                cpu_only=cpu_only,
            )
            logger.init_ok(f"Loading {model_name}", status="Success")
            toc = time.time()
            logger.init_ok(f"Loading {model_name}: Took {toc-tic} seconds", status="Success")
            return True

    def load_esrgan(
        self,
        model_name,
        half_precision=True,
        gpu_id=0,
        cpu_only=False,
    ):
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
            "RealESRGAN_x2plus": RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            ),
            "NMKD_Siax": RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            ),
            "4x_AnimeSharp": RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            ),
        }
        model_path = self.get_model_files(model_name)[0]["path"]
        model_path = f"{self.path}/{model_path}"
        if cpu_only:
            device = torch.device("cpu")
            half_precision = False
        else:
            device = torch.device(f"cuda:{gpu_id}" if self.cuda_available else "cpu")
        logger.info(f"Loading model {model_name} on {device}")
        logger.info(f"Model path: {model_path}")
        if "Real" in model_name:
            model = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=RealESRGAN_models[self.models[model_name]["name"]],
                half=True if half_precision else False,
                device=device,
                gpu_id=gpu_id,
            )
        else:
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=4,
            )
            model.eval()
            model.to(device)
        return {"model": model, "device": device, "half_precision": half_precision}
