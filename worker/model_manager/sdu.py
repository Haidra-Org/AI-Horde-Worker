import time
from pathlib import Path

import k_diffusion as K
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from ldm.util import instantiate_from_config
from worker.model_manager.base import BaseModelManager
from worker.logger import logger


class NoiseLevelAndTextConditionedUpscaler(nn.Module):
    def __init__(self, inner_model, sigma_data=1.0, embed_dim=256):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        self.low_res_noise_embed = K.layers.FourierFeatures(1, embed_dim, std=2)

    def forward(self, input, sigma, low_res, low_res_sigma, c, **kwargs):
        cross_cond, cross_cond_padding, pooler = c
        c_in = 1 / (low_res_sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = low_res_sigma.log1p()[:, None]
        c_in = K.utils.append_dims(c_in, low_res.ndim)
        low_res_noise_embed = self.low_res_noise_embed(c_noise)
        low_res_in = F.interpolate(low_res, scale_factor=2, mode="nearest") * c_in
        mapping_cond = torch.cat([low_res_noise_embed, pooler], dim=1)
        return self.inner_model(
            input,
            sigma,
            unet_cond=low_res_in,
            mapping_cond=mapping_cond,
            cross_cond=cross_cond,
            cross_cond_padding=cross_cond_padding,
            **kwargs,
        )


class CLIPTokenizerTransform:
    def __init__(self, version="openai/clip-vit-large-patch14", max_length=77):
        from transformers import CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.max_length = max_length

    def __call__(self, text):
        indexer = 0 if isinstance(text, str) else ...
        tok_out = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tok_out["input_ids"][indexer]
        attention_mask = 1 - tok_out["attention_mask"][indexer]
        return input_ids, attention_mask


class CLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()
        from transformers import CLIPTextModel, logging

        logging.set_verbosity_error()
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.transformer = self.transformer.eval().requires_grad_(False).to(device)

    @property
    def device(self):
        return self.transformer.device

    def forward(self, tok_out):
        input_ids, cross_cond_padding = tok_out
        clip_out = self.transformer(input_ids=input_ids.to(self.device), output_hidden_states=True)
        return clip_out.hidden_states[-1], cross_cond_padding.to(self.device), clip_out.pooler_output


class SDUModelManager(BaseModelManager):
    def __init__(self, download_reference=False):
        super().__init__()
        self.download_reference = download_reference
        self.path = f"{Path.home()}/.cache/nataili/sdu"
        self.models_db_name = "sdu"
        self.models_path = self.pkg / f"{self.models_db_name}.json"
        self.remote_db = (
            f"https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/{self.models_db_name}.json"
        )
        self.init()

    def load(
        self,
        model_name,
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
        if model_name not in self.models:
            logger.error(f"{model_name} not found")
            return
        if model_name not in self.available_models:
            logger.error(f"{model_name} not available")
            logger.init_ok(f"Downloading {model_name}", status="Downloading")
            self.download_model(model_name)
            logger.init_ok(f"{model_name} downloaded", status="Downloading")
        if model_name not in self.loaded_models:
            tic = time.time()
            logger.init(f"{model_name}", status="Loading")
            self.loaded_models[model_name] = self.load_sdu(
                model_name,
                half_precision=half_precision,
                gpu_id=gpu_id,
                cpu_only=cpu_only,
            )
            logger.init_ok(f"Loading {model_name}", status="Success")
            toc = time.time()
            logger.init_ok(f"Loading {model_name}: Took {toc-tic} seconds", status="Success")

    def load_model_from_config(self, model_path="", config_path="", map_location="cpu", device="cpu"):
        config = OmegaConf.load(config_path)
        pl_sd = torch.load(model_path, map_location=map_location)
        if "global_step" in pl_sd:
            logger.info(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        model = model.eval().requires_grad_(False)
        del pl_sd, sd, m, u
        return model

    def load_sdu(
        self,
        model_name,
        half_precision=True,
        gpu_id=0,
        cpu_only=False,
        train=False,
        pooler_dim=768,
    ):
        config_path = self.get_model_files(model_name)[0]["path"]
        config_path = f"{self.path}/{config_path}"
        model_path = self.get_model_files(model_name)[1]["path"]
        model_path = f"{self.path}/{model_path}"
        vae_config_path = self.get_model_files(model_name)[2]["path"]
        vae_config_path = f"{self.path}/{vae_config_path}"
        vae_840k_path = self.get_model_files(model_name)[3]["path"]
        vae_840k_path = f"{self.path}/{vae_840k_path}"
        vae_560k_path = self.get_model_files(model_name)[4]["path"]
        vae_560k_path = f"{self.path}/{vae_560k_path}"
        if cpu_only:
            device = torch.device("cpu")
            half_precision = False
        else:
            device = torch.device(f"cuda:{gpu_id}" if self.cuda_available else "cpu")
        logger.info(f"Loading model {model_name} on {device}")
        logger.info(f"Model path: {model_path}")
        config = K.config.load_config(open(config_path))
        model = K.config.make_model(config)
        model = NoiseLevelAndTextConditionedUpscaler(
            model,
            sigma_data=config["model"]["sigma_data"],
            embed_dim=config["model"]["mapping_cond_dim"] - pooler_dim,
        )
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model_ema"])
        model = K.config.make_denoiser_wrapper(config)(model)
        if not train:
            model = model.eval().requires_grad_(False)
        model = model.eval()
        model.to(device)
        vae_model_840k = self.load_model_from_config(model_path=vae_840k_path, config_path=vae_config_path)
        vae_model_560k = self.load_model_from_config(model_path=vae_560k_path, config_path=vae_config_path)
        vae_model_840k = vae_model_840k.to(device)
        vae_model_560k = vae_model_560k.to(device)
        tokenizer = CLIPTokenizerTransform()
        text_encoder = CLIPEmbedder(device=device)
        return {
            "model": model,
            "device": device,
            "half_precision": half_precision,
            "vae_model_840k": vae_model_840k,
            "vae_model_560k": vae_model_560k,
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
        }
