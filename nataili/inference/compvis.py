import os
import re
import sys
from contextlib import nullcontext

import k_diffusion as K
import numpy as np
import PIL
import skimage
import torch
import tqdm
from einops import rearrange
from slugify import slugify
from transformers import CLIPFeatureExtractor

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.kdiffusion import CFGMaskedDenoiser, KDiffusionSampler
from ldm.models.diffusion.plms import PLMSSampler
from nataili.util import logger
from nataili.util.cache import torch_gc
from nataili.util.check_prompt_length import check_prompt_length
from nataili.util.get_next_sequence_number import get_next_sequence_number

from nataili.util.save_sample import save_sample
from nataili.util.seed_to_int import seed_to_int
from nataili.util.img2img import *
from nataili.util.create_random_tensors import create_random_tensors

try:
    from nataili.util.voodoo import load_from_plasma, performance
except ModuleNotFoundError as e:
    from nataili import disable_voodoo

    if not disable_voodoo.active:
        raise e

class CompVis:
    def __init__(
        self,
        model,
        device,
        output_dir,
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
        init_mask=None,
        mask_mode="mask",
        resize_mode="resize",
        noise_mode="seed",
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
    ):
        if mask_mode == "mask":
            if init_mask:
                init_mask = process_init_mask(init_mask)
        elif mask_mode == "invert":
            if init_mask:
                init_mask = process_init_mask(init_mask)
                init_mask = PIL.ImageOps.invert(init_mask)
        elif mask_mode == "alpha":
            init_img_transparency = init_img.split()[-1].convert(
                "L"
            )  # .point(lambda x: 255 if x > 0 else 0, mode='1')
            init_mask = init_img_transparency
            init_mask = init_mask.convert("RGB")
            init_mask = resize_image(resize_mode, init_mask, width, height)
            init_mask = init_mask.convert("RGB")

        assert 0.0 <= denoising_strength <= 1.0, "can only work with strength in [0.0, 1.0]"
        t_enc = int(denoising_strength * ddim_steps)

        if (
            init_mask is not None
            and (noise_mode == "matched" or noise_mode == "find_and_matched")
            and init_img is not None
        ):
            noise_q = 0.99
            color_variation = 0.0
            mask_blend_factor = 1.0

            np_init = (np.asarray(init_img.convert("RGB")) / 255.0).astype(
                np.float64
            )  # annoyingly complex mask fixing
            np_mask_rgb = 1.0 - (np.asarray(PIL.ImageOps.invert(init_mask).convert("RGB")) / 255.0).astype(np.float64)
            np_mask_rgb -= np.min(np_mask_rgb)
            np_mask_rgb /= np.max(np_mask_rgb)
            np_mask_rgb = 1.0 - np_mask_rgb
            np_mask_rgb_hardened = 1.0 - (np_mask_rgb < 0.99).astype(np.float64)
            blurred = skimage.filters.gaussian(np_mask_rgb_hardened[:], sigma=16.0, channel_axis=2, truncate=32.0)
            blurred2 = skimage.filters.gaussian(np_mask_rgb_hardened[:], sigma=16.0, channel_axis=2, truncate=32.0)
            # np_mask_rgb_dilated = np_mask_rgb + blurred  # fixup mask todo: derive magic constants
            # np_mask_rgb = np_mask_rgb + blurred
            np_mask_rgb_dilated = np.clip((np_mask_rgb + blurred2) * 0.7071, 0.0, 1.0)
            np_mask_rgb = np.clip((np_mask_rgb + blurred) * 0.7071, 0.0, 1.0)

            noise_rgb = get_matched_noise(np_init, np_mask_rgb, noise_q, color_variation)
            blend_mask_rgb = np.clip(np_mask_rgb_dilated, 0.0, 1.0) ** (mask_blend_factor)
            noised = noise_rgb[:]
            blend_mask_rgb **= 2.0
            noised = np_init[:] * (1.0 - blend_mask_rgb) + noised * blend_mask_rgb

            np_mask_grey = np.sum(np_mask_rgb, axis=2) / 3.0
            ref_mask = np_mask_grey < 1e-3

            all_mask = np.ones((height, width), dtype=bool)
            noised[all_mask, :] = skimage.exposure.match_histograms(
                noised[all_mask, :] ** 1.0, noised[ref_mask, :], channel_axis=1
            )

            init_img = PIL.Image.fromarray(np.clip(noised * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")

        def init():
            image = init_img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = image[None].transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)

            mask_channel = None
            if init_mask:
                alpha = resize_image(resize_mode, init_mask, width // 8, height // 8)
                mask_channel = alpha.split()[-1]

            mask = None
            if mask_channel is not None:
                mask = np.array(mask_channel).astype(np.float32) / 255.0
                mask = 1 - mask
                mask = np.tile(mask, (4, 1, 1))
                mask = mask[None].transpose(0, 1, 2, 3)
                mask = torch.from_numpy(mask).to(model.device)

            init_image = 2.0 * image - 1.0
            init_image = init_image.to(model.device)
            init_latent = model.get_first_stage_encoding(
                model.encode_first_stage(init_image)
            )  # move to latent space

            return (
                init_latent,
                mask,
            )

        def sample_img2img(init_data, x, conditioning, unconditional_conditioning, sampler_name):
            t_enc_steps = t_enc
            obliterate = False
            if ddim_steps == t_enc_steps:
                t_enc_steps = t_enc_steps - 1
                obliterate = True

            if sampler_name != "DDIM":
                x0, z_mask = init_data

                sigmas = sampler.model_wrap.get_sigmas(ddim_steps)
                noise = x * sigmas[ddim_steps - t_enc_steps - 1]

                xi = x0 + noise

                # Obliterate masked image
                if z_mask is not None and obliterate:
                    random = torch.randn(z_mask.shape, device=xi.device)
                    xi = (z_mask * noise) + ((1 - z_mask) * xi)

                sigma_sched = sigmas[ddim_steps - t_enc_steps - 1 :]
                model_wrap_cfg = CFGMaskedDenoiser(sampler.model_wrap)
                samples_ddim = K.sampling.__dict__[f"sample_{sampler.get_sampler_name()}"](
                    model_wrap_cfg,
                    xi,
                    sigma_sched,
                    extra_args={
                        "cond": conditioning,
                        "uncond": unconditional_conditioning,
                        "cond_scale": cfg_scale,
                        "mask": z_mask,
                        "x0": x0,
                        "xi": xi,
                    },
                    disable=False,
                )
            else:

                x0, z_mask = init_data

                sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
                z_enc = sampler.stochastic_encode(
                    x0,
                    torch.tensor([t_enc_steps] * batch_size).to(self.model.device),
                )

                # Obliterate masked image
                if z_mask is not None and obliterate:
                    random = torch.randn(z_mask.shape, device=z_enc.device)
                    z_enc = (z_mask * random) + ((1 - z_mask) * z_enc)

                    # decode it
                samples_ddim = sampler.decode(
                    z_enc,
                    conditioning,
                    t_enc_steps,
                    unconditional_guidance_scale=cfg_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    z_mask=z_mask,
                    x0=x0,
                )
            return samples_ddim

        def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
            samples_ddim, _ = sampler.sample(
                S=ddim_steps,
                conditioning=conditioning,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=unconditional_conditioning,
                x_T=x,
            )
            return samples_ddim

        seed = seed_to_int(seed)

        image_dict = {"seed": seed}
        negprompt = ""
        if "###" in prompt:
            prompt, negprompt = prompt.split("###", 1)
            prompt = prompt.strip()
            negprompt = negprompt.strip()
        
        if self.load_concepts and self.concepts_dir is not None:
            prompt_tokens = re.findall("<([a-zA-Z0-9-]+)>", prompt)
            if prompt_tokens:
                self.process_prompt_tokens(prompt_tokens, model)

        os.makedirs(self.output_dir, exist_ok=True)

        sample_path = os.path.join(self.output_dir, "samples")
        os.makedirs(sample_path, exist_ok=True)

        if not self.disable_voodoo:
            with load_from_plasma(self.model, self.device) as model:
                if sampler_name == "PLMS":
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
                else:
                    raise Exception("Unknown sampler: " + sampler_name)

                if self.verify_input:
                    try:
                        check_prompt_length(model, prompt, self.comments)
                    except Exception:
                        import traceback

                        print("Error verifying input:", file=sys.stderr)
                        print(traceback.format_exc(), file=sys.stderr)

                    all_prompts = batch_size * n_iter * [prompt]
                    all_seeds = [seed + x for x in range(len(all_prompts))]

                with torch.no_grad():
                    for n in range(n_iter):
                        print(f"Iteration: {n+1}/{n_iter}")
                        prompts = all_prompts[n * batch_size : (n + 1) * batch_size]
                        seeds = all_seeds[n * batch_size : (n + 1) * batch_size]

                        uc = model.get_learned_conditioning(len(prompts) * [negprompt])

                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        c = model.get_learned_conditioning(prompts)

                        opt_C = 4
                        opt_f = 8
                        shape = [opt_C, height // opt_f, width // opt_f]

                        x = create_random_tensors(shape, seeds=seeds, device=self.device)
                        init_data = init() if init_img else None
                        samples_ddim = sample_img2img(
                            init_data=init_data,
                            x=x,
                            conditioning=c,
                            unconditional_conditioning=uc,
                            sampler_name=sampler_name,
                        ) if init_img else sample(
                            init_data=init_data,
                            x=x,
                            conditioning=c,
                            unconditional_conditioning=uc,
                            sampler_name=sampler_name,
                        )

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        else:
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
            else:
                raise Exception("Unknown sampler: " + sampler_name)

            if self.verify_input:
                try:
                    check_prompt_length(self.model, prompt, self.comments)
                except Exception:
                    import traceback

                    print("Error verifying input:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

                all_prompts = batch_size * n_iter * [prompt]
                all_seeds = [seed + x for x in range(len(all_prompts))]

            with torch.no_grad():
                for n in range(n_iter):
                    print(f"Iteration: {n+1}/{n_iter}")
                    prompts = all_prompts[n * batch_size : (n + 1) * batch_size]
                    seeds = all_seeds[n * batch_size : (n + 1) * batch_size]

                    uc = self.model.get_learned_conditioning(len(prompts) * [negprompt])

                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    c = self.model.get_learned_conditioning(prompts)

                    opt_C = 4
                    opt_f = 8
                    shape = [opt_C, height // opt_f, width // opt_f]

                    x = create_random_tensors(shape, seeds=seeds, device=self.device)

                    samples_ddim = sample(
                        init_data=None,
                        x=x,
                        conditioning=c,
                        unconditional_conditioning=uc,
                        sampler_name=sampler_name,
                    )

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        for i, x_sample in enumerate(x_samples_ddim):
            sanitized_prompt = slugify(prompts[i])
            full_path = os.path.join(os.getcwd(), sample_path)
            sample_path_i = sample_path
            base_count = get_next_sequence_number(sample_path_i)
            filename = f"{base_count:05}-{ddim_steps}_{sampler_name}_{seeds[i]}_{sanitized_prompt}"[
                : 200 - len(full_path)
            ]

            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
            x_sample = x_sample.astype(np.uint8)
            image = PIL.Image.fromarray(x_sample)
            if self.safety_checker is not None and self.filter_nsfw:
                image_features = self.feature_extractor(image, return_tensors="pt").to(self.device)
                output_images, has_nsfw_concept = self.safety_checker(
                    clip_input=image_features.pixel_values, images=x_sample
                )
                if has_nsfw_concept:
                    logger.info(f"Image {filename} has NSFW concept")
                    image = output_images[0]
                    image = PIL.Image.fromarray(image)
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
