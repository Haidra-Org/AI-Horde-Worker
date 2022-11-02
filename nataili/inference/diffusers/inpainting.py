import os
import re
import sys
import PIL
import PIL.ImageOps
import torch
from contextlib import nullcontext
from slugify import slugify
from diffusers import StableDiffusionInpaintPipeline

from nataili.util.cache import torch_gc
from nataili.util.check_prompt_length import check_prompt_length
from nataili.util.get_next_sequence_number import get_next_sequence_number
from nataili.util.save_sample import save_sample
from nataili.util.seed_to_int import seed_to_int
from nataili.util import logger

class inpainting:
    def __init__(self, pipe, device, output_dir, save_extension='jpg', output_file_path=False, load_concepts=False,
      concepts_dir=None, verify_input=True, auto_cast=True, filter_nsfw = False):
        self.output_dir = output_dir
        self.output_file_path = output_file_path
        self.save_extension = save_extension
        self.load_concepts = load_concepts
        self.concepts_dir = concepts_dir
        self.verify_input = verify_input
        self.auto_cast = auto_cast
        self.pipe = pipe
        self.device = device
        self.comments = []
        self.output_images = []
        self.info = ''
        self.stats = ''
        self.images = []
        self.filter_nsfw = filter_nsfw

    def resize_image(self, resize_mode, im, width, height):
        LANCZOS = (PIL.Image.Resampling.LANCZOS if hasattr(PIL.Image, 'Resampling') else PIL.Image.LANCZOS)
        if resize_mode == "resize":
            res = im.resize((width, height), resample=LANCZOS)
        elif resize_mode == "crop":
            ratio = width / height
            src_ratio = im.width / im.height

            src_w = width if ratio > src_ratio else im.width * height // im.height
            src_h = height if ratio <= src_ratio else im.height * width // im.width

            resized = im.resize((src_w, src_h), resample=LANCZOS)
            res = PIL.Image.new("RGBA", (width, height))
            res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        else:
            ratio = width / height
            src_ratio = im.width / im.height

            src_w = width if ratio < src_ratio else im.width * height // im.height
            src_h = height if ratio >= src_ratio else im.height * width // im.width

            resized = im.resize((src_w, src_h), resample=LANCZOS)
            res = PIL.Image.new("RGBA", (width, height))
            res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

            if ratio < src_ratio:
                fill_height = height // 2 - src_h // 2
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
            elif ratio > src_ratio:
                fill_width = width // 2 - src_w // 2
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

        return res

    def generate(self, prompt: str, inpaint_img=None, inpaint_mask=None, ddim_steps=50, n_iter=1, batch_size=1,
      cfg_scale=7.5, seed=None, height=512, width=512, save_individual_images: bool = True):

        safety_checker = None
        if not self.filter_nsfw:
            safety_checker = self.pipe.safety_checker
            self.pipe.safety_checker = None
        seed = seed_to_int(seed)
        inpaint_img = self.resize_image('resize', inpaint_img, width, height)

        # mask information has been transferred in the Alpha channel of the inpaint image
        logger.debug(inpaint_mask)
        if inpaint_mask is None:
           try:
               red, green, blue, alpha = inpaint_img.split()
           except ValueError:
              raise Exception("inpainting image doesn't have an alpha channel.")              
           
           inpaint_mask = alpha
           inpaint_mask = PIL.ImageOps.invert(inpaint_mask)
        else:
           inpaint_mask = self.resize_image('resize', inpaint_mask, width, height)

        torch_gc()

        if self.load_concepts and self.concepts_dir is not None:
            prompt_tokens = re.findall('<([a-zA-Z0-9-]+)>', prompt)
            if prompt_tokens:
                self.process_prompt_tokens(prompt_tokens)

        os.makedirs(self.output_dir, exist_ok=True)

        sample_path = os.path.join(self.output_dir, "samples")
        os.makedirs(sample_path, exist_ok=True)

        all_prompts = batch_size * [prompt]
        all_seeds = [seed + x for x in range(len(all_prompts))]

        precision_scope = torch.autocast if self.auto_cast else nullcontext

        with torch.no_grad(), precision_scope("cuda"):
           for n in range(batch_size):
              print(f"Iteration: {n+1}/{batch_size}")

              prompt = all_prompts[n]
              seed = all_seeds[n]
              print("prompt: " + prompt + ", seed: " + str(seed))

              generator = torch.Generator(device=self.device).manual_seed(seed)

              x_samples = self.pipe(
                 prompt=prompt,
                 image=inpaint_img,
                 mask_image=inpaint_mask,
                 guidance_scale=cfg_scale,
                 num_inference_steps=ddim_steps,
                 generator=generator,
                 num_images_per_prompt=n_iter,
                 width=width,
                 height=height                 
              ).images

              for i, x_sample in enumerate(x_samples):
                 image_dict = {
                   "seed": seed,
                   "image": x_sample
                 }
                 
                 self.images.append(image_dict)
                 if safety_checker:
                    self.pipe.safety_checker = safety_checker

                 if save_individual_images:
                    sanitized_prompt = slugify(prompt)
                    sample_path_i = sample_path
                    base_count = get_next_sequence_number(sample_path_i)
                    full_path = os.path.join(os.getcwd(), sample_path)
                    filename = f"{base_count:05}-{ddim_steps}_{seed}_{sanitized_prompt}"[:200-len(full_path)]

                    path = os.path.join(sample_path, filename + '.' + self.save_extension)
                    success = save_sample(x_sample, filename, sample_path_i, self.save_extension)

                    if success:
                       if self.output_file_path:
                          self.output_images.append(path)
                       else:
                          self.output_images.append(x_sample)
                    else:
                       return

        self.info = f"""
                {prompt}
                Steps: {ddim_steps}, CFG scale: {cfg_scale}, Seed: {seed}
                """.strip()
        self.stats = f'''
                '''

        for comment in self.comments:
            self.info += "\n\n" + comment

        torch_gc()

        del generator

        return
