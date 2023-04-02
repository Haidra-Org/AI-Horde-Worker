import safetensors.torch
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from worker.cache import get_cache_directory
from worker.model_manager.base import BaseModelManager

# from nataili.model_manager.compvis import CompVisModelManager, DisableInitialization
from worker.logger import logger


class ControlNetModelManager(BaseModelManager):
    def __init__(self, download_reference=True, compvis=None):
        super().__init__()
        self.download_reference = download_reference
        self.path = f"{get_cache_directory()}/controlnet"
        self.models_db_name = "controlnet"
        self.models_path = self.pkg / f"{self.models_db_name}.json"
        self.remote_db = (
            f"https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/{self.models_db_name}.json"
        )
        self.control_nets = {}
        self.init()

    def load_control_ldm(
        self,
        model_name,
        target_name,
        input_state_dict,
        half_precision=True,
        gpu_id=0,
        cpu_only=False,
        _device=None,
    ):
        if not self.cuda_available:
            cpu_only = True
        if cpu_only:
            device = torch.device("cpu")
            half_precision = False
        else:
            device = torch.device(f"cuda:{gpu_id}" if self.cuda_available else "cpu")
        if model_name not in self.control_nets:
            logger.error(f"{model_name} not loaded")
            return False
        config_path = self.get_model_files(model_name)[1]["path"]
        config_path = f"{self.pkg}/{config_path}"
        logger.info(f"Loading controlLDM {model_name} for {target_name}")
        config = OmegaConf.load(config_path)
        try:
            with DisableInitialization(disable_clip=True):
                model = instantiate_from_config(config.model)
        except Exception as e:
            pass
        full_name = f"{model_name}_{target_name}"
        logger.info(f"Loaded {full_name} ControlLDM")
        sd15_with_control_state_dict = self.control_nets[model_name]["state_dict"]
        final_state_dict: dict = input_state_dict.copy()
        keys = sd15_with_control_state_dict.keys()
        logger.info("Merge control net state dict into target state dict")
        if "_sd2" not in model_name:
            for key in keys:
                if not key.startswith("control_"):
                    continue
                p = sd15_with_control_state_dict[key]
                key_name = f'model.diffusion_model.{key.replace("control_model.", "")}'
                if key_name in input_state_dict.keys():
                    # logger.info(f"merging {key_name} from input {key} from control")
                    p_new = p + input_state_dict[key_name].clone().cpu()
                else:
                    # logger.info(f"directly copying {key_name} from control")
                    p_new = p
                final_state_dict[key] = p_new
        else:
            for key in keys:
                p = sd15_with_control_state_dict[key]
                key_name = f'model.diffusion_model.{key.replace("control_model.", "")}'
                if key in input_state_dict.keys():
                    # logger.info(f"merging {key_name} from input {key} from control")
                    p_new = p + input_state_dict[key_name].clone().cpu()
                else:
                    # logger.info(f"directly copying {key_name} from control")
                    p_new = p
                final_state_dict[f"control_model.{key}"] = p_new
        # remove key "lvlb_weights"
        if "lvlb_weights" in final_state_dict.keys():
            final_state_dict.pop("lvlb_weights")
        logger.info("Finished merging control net state dict into target state dict")
        logger.info(f"Loading {full_name} state dict")
        model.load_state_dict(final_state_dict, strict=True)
        logger.info(f"Loaded {full_name} state dict")
        if half_precision:
            model.half()
        del final_state_dict, sd15_with_control_state_dict, input_state_dict
        self.loaded_models[full_name] = {"model": model, "device": device, "half_precision": half_precision}

    def load_controlnet(
        self,
        model_name,
    ):
        if model_name not in self.models:
            logger.error(f"{model_name} not found")
            return False
        if model_name not in self.available_models:
            logger.error(f"{model_name} not available")
            logger.init_ok(f"Downloading {model_name}", status="Downloading")
            self.download_model(model_name)
            logger.init_ok(f"{model_name} downloaded", status="Downloading")
        if model_name in self.control_nets:
            logger.info(f"{model_name} already loaded")
            return True
        model_path = self.get_model_files(model_name)[0]["path"]
        model_path = f"{self.path}/{model_path}"
        logger.info(f"Loading controlnet {model_name}")
        logger.info(f"Model path: {model_path}")
        state_dict = safetensors.torch.load_file(model_path)

        self.control_nets[model_name] = {"state_dict": state_dict}
        return True
