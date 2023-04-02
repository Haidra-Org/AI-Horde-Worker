import os
import sys
import time
from pathlib import Path

import open_clip
import torch
import transformers.utils.hub
from omegaconf import OmegaConf
from torch import nn

# import ldm.modules.encoders.modules
# from ldm.util import instantiate_from_config
# from nataili import enable_ray_alternative
from worker.cache import get_cache_directory
from worker.model_manager.base import BaseModelManager

from worker.logger import logger
# from nataili.util.voodoo import get_model_cache_filename, have_model_cache, push_model_to_plasma


class CompVisModelManager(BaseModelManager):
    def __init__(self, download_reference=True, custom_path="models/custom"):
        super().__init__()
        self.download_reference = download_reference
        self.path = f"{get_cache_directory()}/compvis"
        self.custom_path = custom_path
        self.models_db_name = "stable_diffusion"
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
        voodoo=False,
    ):
        """
        model_name: str. Name of the model to load. See available_models for a list of available models.
        half_precision: bool. If True, the model will be loaded in half precision.
        gpu_id: int. The id of the gpu to use. If the gpu is not available, the model will be loaded on the cpu.
        cpu_only: bool. If True, the model will be loaded on the cpu. If True, half_precision will be set to False.
        voodoo: bool. Voodoo (Ray)
        """
        if model_name not in self.models:
            logger.error(f"{model_name} not found")
            return False
        if model_name not in self.available_models:
            logger.error(f"{model_name} not available")
            self.download_model(model_name)
            logger.init_ok(f"{model_name}", status="Downloaded")
        if model_name not in self.loaded_models:
            if not self.cuda_available:
                cpu_only = True
                voodoo = False
            tic = time.time()
            logger.init(f"{model_name}", status="Loading")
            self.loaded_models[model_name] = self.load_compvis(
                model_name,
                half_precision=half_precision,
                gpu_id=gpu_id,
                cpu_only=cpu_only,
                voodoo=voodoo,
            )
            toc = time.time()
            logger.init_ok(f"{model_name}: {round(toc-tic,2)} seconds", status="Loaded")
            return True

    def load_model_from_config(self, model_path="", config_path="", map_location="cpu"):
        config = OmegaConf.load(config_path)
        pl_sd = torch.load(model_path, map_location=map_location)
        if "global_step" in pl_sd:
            logger.info(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd

        sd1_clip_weight = "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
        sd2_clip_weight = "cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight"
        clip_is_included_into_sd = sd1_clip_weight in sd or sd2_clip_weight in sd

        model = None
        try:
            with DisableInitialization(disable_clip=clip_is_included_into_sd):
                model = instantiate_from_config(config.model)
        except Exception as e:
            pass

        if model is None:
            logger.info("Failed to create model quickly; will retry using slow method.")
            model = instantiate_from_config(config.model)

        m, u = model.load_state_dict(sd, strict=False)
        model = model.eval()
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m._orig_padding_mode = m.padding_mode
        del pl_sd, sd, m, u
        return model

    def load_custom(self, ckpt_path, config_path, model_name=None, replace=False):
        if not os.path.isfile(ckpt_path):
            logger.error(f"{ckpt_path} not found")
            return
        if not os.path.isfile(config_path):
            logger.error(f"{config_path} not found")
            return
        if not ckpt_path.endswith(".ckpt"):
            logger.error(f"{ckpt_path} is not a valid checkpoint file")
            return
        if not config_path.endswith(".yaml"):
            logger.error(f"{config_path} is not a valid config file")
            return
        if model_name is None:
            model_name = os.path.basename(ckpt_path).replace(".ckpt", "")
        if model_name not in self.models or replace:
            self.models[model_name] = {
                "name": model_name,
                "type": "ckpt",
                "description": f"custom model {model_name}",
                "config": {
                    "files": [
                        {"path": f"{self.custom_path}/{model_name}.ckpt"},
                        {"path": f"{self.custom_path}/{model_name}.yaml"},
                    ]
                },
                "available": True,
            }
            self.available_models.append(model_name)

    def load_available_models_from_custom(self, replace=False):
        # ckpt files and matching config yaml files
        for file in os.listdir(self.custom_path):
            if file.endswith(".ckpt"):
                ckpt_path = f"{self.custom_path}/{file}"
                config_path = ckpt_path.replace(".ckpt", ".yaml")
                self.load_custom(
                    ckpt_path,
                    config_path,
                    replace=replace,
                )

    def load_compvis(
        self,
        model_name,
        half_precision=True,
        gpu_id=0,
        cpu_only=False,
        voodoo=False,
    ):
        ckpt_path = self.get_model_files(model_name)[0]["path"]
        ckpt_path = f"{self.path}/{ckpt_path}"
        config_path = self.get_model_files(model_name)[1]["path"]
        config_path = f"{self.pkg}/{config_path}"
        if cpu_only:
            device = torch.device("cpu")
            half_precision = False
        else:
            device = torch.device(f"cuda:{gpu_id}" if self.cuda_available else "cpu")
        logger.debug(f"Loading model {model_name} on {device}")
        logger.debug(f"Model path: {ckpt_path}")

        if voodoo and enable_ray_alternative.active and have_model_cache(ckpt_path):
            logger.debug("Have up to date model cache, using that instead of ckpt model")
            model = get_model_cache_filename(ckpt_path)
        else:
            logger.debug("Loading model from checkpoint")
            model = self.load_model_from_config(model_path=ckpt_path, config_path=config_path)
            if half_precision:
                logger.debug("Converting model to half precision")
                model = model.half()
                logger.debug("Converting model.cond_stage_model.transformer to half precision")
                if "stable diffusion 2" in self.models[model_name]["baseline"]:
                    model.cond_stage_model.model.transformer = model.cond_stage_model.model.transformer.half()
                else:
                    model.cond_stage_model.transformer = model.cond_stage_model.transformer.half()

        if voodoo and isinstance(model, torch.nn.Module):
            logger.debug(f"Doing voodoo on {model_name}")
            model = push_model_to_plasma(model, ckpt_path)
        elif isinstance(model, torch.nn.Module):
            logger.debug(f"Moving model data directly to device {device}")
            model = model.to(device)
            logger.debug(f"Sending model.cond_stage_model.transformer to {device}")
            if "stable diffusion 2" in self.models[model_name]["baseline"]:
                model.cond_stage_model.model.transformer = model.cond_stage_model.model.transformer.to(device)
            else:
                model.cond_stage_model.transformer = model.cond_stage_model.transformer.to(device)
            logger.debug(f"Setting model.cond_stage_model.device to {device}")
            model.cond_stage_model.device = device

        return {"model": model, "device": device, "half_precision": half_precision}

    def check_model_available(self, model_name):
        if model_name not in self.models:
            return False
        return self.check_file_available(self.get_model_files(model_name)[0]["path"])


class DisableInitialization:
    """
    When an object of this class enters a `with` block, it starts:
    - preventing torch's layer initialization functions from working
    - changes CLIP and OpenCLIP to not download model weights
    - changes CLIP to not make requests to check if there is a new version of a file you already have
    When it leaves the block, it reverts everything to how it was before.
    Use it like this:
    ```
    with DisableInitialization():
        do_things()
    ```
    """

    def __init__(self, disable_clip=True):
        self.replaced = []
        self.disable_clip = disable_clip

    def replace(self, obj, field, func):
        original = getattr(obj, field, None)
        if original is None:
            return None

        self.replaced.append((obj, field, original))
        setattr(obj, field, func)

        return original

    def __enter__(self):
        def do_nothing(*args, **kwargs):
            pass

        def create_model_and_transforms_without_pretrained(*args, pretrained=None, **kwargs):
            return self.create_model_and_transforms(*args, pretrained=None, **kwargs)

        def CLIPTextModel_from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs):
            if sys.version_info != (3, 8, 10):
                try:
                    res = self.CLIPTextModel_from_pretrained(
                        None, *model_args, config=pretrained_model_name_or_path, state_dict={}, **kwargs
                    )
                except Exception as e:
                    res = self.CLIPTextModel_from_pretrained(None, *model_args, **kwargs)
            else:
                res = self.CLIPTextModel_from_pretrained(None, *model_args, **kwargs)

            res.name_or_path = pretrained_model_name_or_path
            return res

        def transformers_modeling_utils_load_pretrained_model(*args, **kwargs):
            args = (
                args[0:3] + ("/",) + args[4:]
            )  # resolved_archive_file; must set it to something to prevent what seems to be a bug
            return self.transformers_modeling_utils_load_pretrained_model(*args, **kwargs)

        def transformers_utils_hub_get_file_from_cache(original, url, *args, **kwargs):
            # this file is always 404, prevent making request
            if (
                url == "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/added_tokens.json"
                or url == "openai/clip-vit-large-patch14"
                and args[0] == "added_tokens.json"
            ):
                return None

            try:
                res = original(url, *args, local_files_only=True, **kwargs)
                if res is None:
                    res = original(url, *args, local_files_only=False, **kwargs)
                return res
            except Exception as e:
                return original(url, *args, local_files_only=False, **kwargs)

        def transformers_utils_hub_get_from_cache(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(
                self.transformers_utils_hub_get_from_cache, url, *args, **kwargs
            )

        def transformers_tokenization_utils_base_cached_file(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(
                self.transformers_tokenization_utils_base_cached_file, url, *args, **kwargs
            )

        def transformers_configuration_utils_cached_file(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(
                self.transformers_configuration_utils_cached_file, url, *args, **kwargs
            )

        self.replace(torch.nn.init, "kaiming_uniform_", do_nothing)
        self.replace(torch.nn.init, "_no_grad_normal_", do_nothing)
        self.replace(torch.nn.init, "_no_grad_uniform_", do_nothing)

        if self.disable_clip:
            self.create_model_and_transforms = self.replace(
                open_clip, "create_model_and_transforms", create_model_and_transforms_without_pretrained
            )
            self.CLIPTextModel_from_pretrained = self.replace(
                ldm.modules.encoders.modules.CLIPTextModel, "from_pretrained", CLIPTextModel_from_pretrained
            )
            self.transformers_modeling_utils_load_pretrained_model = self.replace(
                transformers.modeling_utils.PreTrainedModel,
                "_load_pretrained_model",
                transformers_modeling_utils_load_pretrained_model,
            )
            self.transformers_tokenization_utils_base_cached_file = self.replace(
                transformers.tokenization_utils_base, "cached_file", transformers_tokenization_utils_base_cached_file
            )
            self.transformers_configuration_utils_cached_file = self.replace(
                transformers.configuration_utils, "cached_file", transformers_configuration_utils_cached_file
            )
            self.transformers_utils_hub_get_from_cache = self.replace(
                transformers.utils.hub, "get_from_cache", transformers_utils_hub_get_from_cache
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        for obj, field, original in self.replaced:
            setattr(obj, field, original)

        self.replaced.clear()
