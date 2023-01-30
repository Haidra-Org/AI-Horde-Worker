import os
import re

import einops
import k_diffusion as K
import numpy as np
import PIL
import skimage
import torch
from einops import rearrange
from slugify import slugify
from torch import nn
from transformers import CLIPFeatureExtractor

from ldm2.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.kdiffusion import CFGMaskedDenoiser, KDiffusionSampler
from ldm.models.diffusion.plms import PLMSSampler
from nataili.util import logger
from nataili.util.cache import torch_gc
from nataili.util.create_random_tensors import create_random_tensors
from nataili.util.get_next_sequence_number import get_next_sequence_number
from nataili.util.img2img import find_noise_for_image, get_matched_noise, process_init_mask, resize_image
from nataili.util.performance import performance
from nataili.util.process_prompt_tokens import process_prompt_tokens
from nataili.util.save_sample import save_sample
from nataili.util.seed_to_int import seed_to_int
from PIL import Image, ImageOps
from torch import autocast

try:
    from nataili.util.voodoo import load_from_plasma
except ModuleNotFoundError as e:
    from nataili import disable_voodoo

    if not disable_voodoo.active:
        raise e

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


class CompVisPix2Pix:
    def __init__(
        self,
        model,
        device,
        output_dir,
        model_name=None,
        save_extension="jpg",
        output_file_path=False,
        load_concepts=False,
        concepts_dir=None,
        verify_input=True,
        auto_cast=True,
        filter_nsfw=False,
        safety_checker=None,
        disable_voodoo=False,
    ):
        self.model = model
        self.model_name = model_name
        self.output_dir = output_dir
        self.output_file_path = output_file_path
        self.save_extension = save_extension
        self.load_concepts = load_concepts
        self.concepts_dir = concepts_dir
        self.verify_input = verify_input
        self.auto_cast = auto_cast
        self.device = device
        self.comments = []
        self.output_images = []
        self.info = ""
        self.stats = ""
        self.images = []
        self.filter_nsfw = filter_nsfw
        self.safety_checker = safety_checker
        self.feature_extractor = CLIPFeatureExtractor()
        self.disable_voodoo = disable_voodoo

    @performance
    def generate(
        self,
        prompt: str,
        init_img=None,
        resize_mode="resize",
        noise_mode="seed",
        find_noise_steps=50,
        denoising_strength: float = 0.8,
        ddim_steps=50,
        sampler_name="k_lms",
        n_iter=1,
        batch_size=1,
        cfg_scale=7.5,
        seed=None,
        height=512,
        width=512,
        save_individual_images: bool = True,
        save_grid: bool = True,
        ddim_eta: float = 0.0,
        sigma_override: dict = None,
        tiling: bool = False,
    ):
        if init_img:
            init_img = resize_image(resize_mode, init_img, width, height)
        
        assert 0.0 <= denoising_strength <= 1.0, "can only work with strength in [0.0, 1.0]"
        t_enc = int(denoising_strength * ddim_steps)

        def sample_pix2pix(
            init_data,
            x,
            conditioning,
            unconditional_conditioning,
            sampler_name,
            batch_size=1,
            shape=None,
            karras=False,
            sigma_override: dict = None,
        ):
            if sampler_name == "dpmsolver":
                samples_ddim, _ = sampler.sample(
                    S=ddim_steps,
                    conditioning=conditioning,
                    unconditional_guidance_scale=cfg_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    x_T=x,
                    karras=karras,
                    batch_size=batch_size,
                    shape=shape,
                    sigma_override=sigma_override,
                )
            else:
                samples_ddim, _ = sampler.sample(
                    S=ddim_steps,
                    conditioning=conditioning,
                    unconditional_guidance_scale=cfg_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    x_T=x,
                    karras=karras,
                    sigma_override=sigma_override,
                )
            return samples_ddim

        seed = seed_to_int(seed)

        image_dict = {"seed": seed}
        init_image = init_img
        init_image = ImageOps.fit(init_image, (width, height), method=Image.Resampling.LANCZOS)
        os.makedirs(self.output_dir, exist_ok=True)

        sample_path = os.path.join(self.output_dir, "samples")
        os.makedirs(sample_path, exist_ok=True)

        karras = False
        if "karras" in sampler_name:
            karras = True
            sampler_name = sampler_name.replace("_karras", "")

        if not self.disable_voodoo:
            with load_from_plasma(self.model, self.device) as model:
                for m in model.modules():
                    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                        m.padding_mode = "circular" if tiling else m._orig_padding_mode
                if self.model_name.startswith("stable_diffusion_2"):
                    sampler = DPMSolverSampler(model)
                    sampler_name = "dpmsolver"
                elif sampler_name == "PLMS":
                    sampler = PLMSSampler(model)
                elif sampler_name == "DDIM":
                    sampler = DDIMSampler(model)
                elif sampler_name == "k_dpm_2_a":
                    sampler = KDiffusionSampler(model, "dpm_2_ancestral")
                elif sampler_name == "k_dpm_2":
                    sampler = KDiffusionSampler(model, "dpm_2")
                elif sampler_name == "k_euler_a":
                    sampler = KDiffusionSampler(model, "euler_ancestral")
                elif sampler_name == "k_euler":
                    sampler = KDiffusionSampler(model, "euler")
                elif sampler_name == "k_heun":
                    sampler = KDiffusionSampler(model, "heun")
                elif sampler_name == "k_lms":
                    sampler = KDiffusionSampler(model, "lms")
                elif sampler_name == "k_dpm_fast":
                    sampler = KDiffusionSampler(model, "dpm_fast")
                elif sampler_name == "k_dpm_adaptive":
                    sampler = KDiffusionSampler(model, "dpm_adaptive")
                elif sampler_name == "k_dpmpp_2s_a":
                    sampler = KDiffusionSampler(model, "dpmpp_2s_ancestral")
                elif sampler_name == "k_dpmpp_2m":
                    sampler = KDiffusionSampler(model, "dpmpp_2m")
                elif sampler_name == "dpmsolver":
                    sampler = DPMSolverSampler(model)
                else:
                    logger.info("Unknown sampler: " + sampler_name)
                if self.load_concepts and self.concepts_dir is not None:
                    prompt_tokens = re.findall("<([a-zA-Z0-9-]+)>", prompt)
                    if prompt_tokens:
                        process_prompt_tokens(prompt_tokens, model, self.concepts_dir)

                all_prompts = batch_size * n_iter * [prompt]
                all_seeds = [seed + x for x in range(len(all_prompts))]

                model_wrap = K.external.CompVisDenoiser(model)
                model_wrap_cfg = CFGDenoiser(model_wrap)
                null_token = model.get_learned_conditioning([""])

                with torch.no_grad():
                    for n in range(n_iter):
                        print(f"Iteration: {n+1}/{n_iter}")
                        prompts = all_prompts[n * batch_size : (n + 1) * batch_size]
                        seeds = all_seeds[n * batch_size : (n + 1) * batch_size]

                        cond = {}
                        cond["c_crossattn"] = [model.get_learned_conditioning(prompts)]
                        init_image = 2 * torch.tensor(np.array(init_image)).float() / 255 - 1
                        init_image = rearrange(init_image, "h w c -> 1 c h w").to(model.device)
                        cond["c_concat"] = [model.encode_first_stage(init_image).mode()]

                        uncond = {}
                        uncond["c_crossattn"] = [null_token]
                        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

                        sigmas = model_wrap.get_sigmas(ddim_steps)

                        extra_args = {
                            "cond": cond,
                            "uncond": uncond,
                            "text_cfg_scale": cfg_scale,
                            "image_cfg_scale": denoising_strength,
                        }
                        torch.manual_seed(seed)
                        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                        z = sampler.samplePix2(model_wrap_cfg, z, sigmas, extra_args=extra_args)
                        x = model.decode_first_stage(z)
                        x_samples_ddim = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)

        else:
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    m.padding_mode = "circular" if tiling else m._orig_padding_mode
            if sampler_name == "PLMS":
                sampler = PLMSSampler(self.model)
            elif sampler_name == "DDIM":
                sampler = DDIMSampler(self.model)
            elif sampler_name == "k_dpm_2_a":
                sampler = KDiffusionSampler(self.model, "dpm_2_ancestral")
            elif sampler_name == "k_dpm_2":
                sampler = KDiffusionSampler(self.model, "dpm_2")
            elif sampler_name == "k_euler_a":
                sampler = KDiffusionSampler(self.model, "euler_ancestral")
            elif sampler_name == "k_euler":
                sampler = KDiffusionSampler(self.model, "euler")
            elif sampler_name == "k_heun":
                sampler = KDiffusionSampler(self.model, "heun")
            elif sampler_name == "k_lms":
                sampler = KDiffusionSampler(self.model, "lms")
            elif sampler_name == "k_dpm_fast":
                sampler = KDiffusionSampler(self.model, "dpm_fast")
            elif sampler_name == "k_dpm_adaptive":
                sampler = KDiffusionSampler(self.model, "dpm_adaptive")
            elif sampler_name == "k_dpmpp_2s_a":
                sampler = KDiffusionSampler(self.model, "dpmpp_2s_ancestral")
            elif sampler_name == "k_dpmpp_2m":
                sampler = KDiffusionSampler(self.model, "dpmpp_2m")
            elif sampler_name == "dpmsolver":
                sampler = DPMSolverSampler(self.model)
            else:
                logger.info("Unknown sampler: " + sampler_name)
            if self.model_name == "stable_diffusion_2.0":
                sampler = DPMSolverSampler(self.model)
                sampler_name = "dpmsolver"
            if self.load_concepts and self.concepts_dir is not None:
                prompt_tokens = re.findall("<([a-zA-Z0-9-]+)>", prompt)
                if prompt_tokens:
                    process_prompt_tokens(prompt_tokens, self.model, self.concepts_dir)

            all_prompts = batch_size * n_iter * [prompt]
            all_seeds = [seed + x for x in range(len(all_prompts))]

            model_wrap = K.external.CompVisDenoiser(self.model)
            model_wrap_cfg = CFGDenoiser(model_wrap)
            null_token = self.model.get_learned_conditioning([""])

            with torch.no_grad():
                for n in range(n_iter):
                    print(f"Iteration: {n+1}/{n_iter}")
                    prompts = all_prompts[n * batch_size : (n + 1) * batch_size]
                    seeds = all_seeds[n * batch_size : (n + 1) * batch_size]
                    print (f"Prompt = {prompts} - Seed = {seeds}")
                    cond = {}
                    cond["c_crossattn"] = [self.model.get_learned_conditioning(prompts)]
                    init_image = 2 * torch.tensor(np.array(init_image)).float() / 255 - 1
                    init_image = rearrange(init_image, "h w c -> 1 c h w").to(self.model.device)
                    cond["c_concat"] = [self.model.encode_first_stage(init_image).mode()]

                    uncond = {}
                    uncond["c_crossattn"] = [null_token]
                    uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

                    sigmas = model_wrap.get_sigmas(ddim_steps)

                    extra_args = {
                        "cond": cond,
                        "uncond": uncond,
                        "text_cfg_scale": cfg_scale,
                        "image_cfg_scale": denoising_strength,
                    }
                    torch.manual_seed(seed)
                    z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                    z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
                    x = self.model.decode_first_stage(z)
                    x_samples_ddim = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)

        for i, x_sample in enumerate(x_samples_ddim):
            sanitized_prompt = slugify(prompts[i])
            full_path = os.path.join(os.getcwd(), sample_path)
            sample_path_i = sample_path
            base_count = get_next_sequence_number(sample_path_i)
            if karras:
                sampler_name += "_karras"
            filename = f"{base_count:05}-{ddim_steps}_{sampler_name}_{seeds[i]}_{sanitized_prompt}"[
                : 200 - len(full_path)
            ]

            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
            x_sample = x_sample.astype(np.uint8)
            image = PIL.Image.fromarray(x_sample)
            if self.safety_checker is not None and self.filter_nsfw:
                image_features = self.feature_extractor(image, return_tensors="pt").to("cpu")
                output_images, has_nsfw_concept = self.safety_checker(
                    clip_input=image_features.pixel_values, images=x_sample
                )
                if has_nsfw_concept and True in has_nsfw_concept:
                    logger.info(f"Image {filename} has NSFW concept")
                    image = PIL.Image.new("RGB", (512, 512))
                    image_dict["censored"] = True
            image_dict["image"] = image
            self.images.append(image_dict)

            if save_individual_images:
                path = os.path.join(sample_path, filename + "." + self.save_extension)
                success = save_sample(image, filename, sample_path_i, self.save_extension)
                if success:
                    if self.output_file_path:
                        self.output_images.append(path)
                    else:
                        self.output_images.append(image)
                else:
                    return

        self.info = f"""
                {prompt}
                Steps: {ddim_steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}
                """.strip()
        self.stats = """
                """

        for comment in self.comments:
            self.info += "\n\n" + comment

        torch_gc()

        del sampler

        return
