import time
import uuid

import PIL

from nataili.inference.compvis import CompVis
from nataili.model_manager import ModelManager
from nataili.util.cache import torch_gc
from nataili.util.logger import logger

init_image = PIL.Image.open("./01.png").convert("RGB")

mm = ModelManager()

mm.init()
logger.debug("Available dependencies:")
for dependency in mm.available_dependencies:
    logger.debug(dependency)

logger.debug("Available models:")
for model in mm.available_models:
    logger.debug(model)

models_to_load = [
    "stable_diffusion",
    # 'waifu_diffusion',
    # "trinart",
    # 'GFPGAN', 'RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B',
    # 'BLIP', 'ViT-L/14', 'ViT-g-14', 'ViT-H-14'
]
logger.init(f"{models_to_load}", status="Loading")


def test_compvis(
    model,
    prompt,
    sampler,
    steps=30,
    output_dir=None,
    seed=None,
    init_img=None,
    denoising_strength=0.75,
    sigma_override=None,
    filter_nsfw=False,
    safety_checker=None,
):
    log_message = f"sampler: {sampler} steps: {steps} model: {model}"
    if seed:
        log_message += f" seed: {seed}"
    if sigma_override:
        log_message += f" sigma_override: {sigma_override}"
    logger.info(log_message)
    if filter_nsfw:
        logger.info("Filtering NSFW")
    if init_img:
        logger.info("Using init image")
        logger.info(f"Denoising strength: {denoising_strength}")
    compvis = CompVis(
        mm.loaded_models[model]["model"],
        mm.loaded_models[model]["device"],
        output_dir,
        disable_voodoo=True,
        filter_nsfw=filter_nsfw,
        safety_checker=safety_checker,
    )
    compvis.generate(
        prompt,
        sampler_name=sampler,
        ddim_steps=steps,
        seed=seed,
        init_img=init_img,
        sigma_override=sigma_override,
        denoising_strength=denoising_strength,
    )


samplers = [
    "k_dpm_fast",
    "k_dpmpp_2s_a",
    "k_dpm_adaptive",
    "k_dpmpp_2m",
    "k_dpm_2_a",
    "k_dpm_2",
    "k_euler_a",
    "k_euler",
    "k_heun",
    "k_lms",
]

step_counts = [5, 7, 8, 15]

sigma_overrides = [
    {"min": 0.6958, "max": 9.9172, "rho": 7.0},
]

denoising_strengths = [0.22, 0.44, 0.88]

output_dir = f"./test_output/{str(uuid.uuid4())}"


@logger.catch
def test():
    tic = time.time()
    model = "safety_checker"
    logger.init(f"Model: {model}", status="Loading")
    success = mm.load_model(model)
    toc = time.time()
    logger.init_ok(f"Loading {model}: Took {toc-tic} seconds", status=success)
    for model in models_to_load:
        torch_gc()
        tic = time.time()
        logger.init(f"Model: {model}", status="Loading")

        success = mm.load_model(model)

        toc = time.time()
        logger.init_ok(f"Loading {model}: Took {toc-tic} seconds", status=success)
        torch_gc()

        if model in ["stable_diffusion", "waifu_diffusion", "trinart"]:
            logger.info(f"Output dir: {output_dir}")
            logger.debug(f"Running inference on {model}")
            logger.info(f"Testing {len(samplers)} samplers")
            prompt = (
                "Headshot of cybernetic female character, cybernetic implants, solid background color,"
                "digital art, illustration, smooth color, cinematic moody lighting, cyberpunk, body modification,"
                "wenjun lin, studio ghibli, pixiv, artgerm, greg rutkowski, ilya kuvshinov"
            )
            logger.info(f"Prompt: {prompt}")
            for sampler in samplers:
                test_compvis(model, prompt, sampler, output_dir=output_dir)

            samplers.remove("k_dpm_adaptive")  # This sampler doesn't work with step counts
            logger.info(f"Testing {len(samplers)} samplers with Karras on step counts {step_counts}")
            logger.info(f"Prompt: {prompt}")
            for sampler in samplers:
                sampler = f"{sampler}_karras"
                for steps in step_counts:
                    test_compvis(model, prompt, sampler, steps=steps, output_dir=output_dir)

            logger.info(
                f"Testing {len(samplers)} samplers with Karras on step counts {step_counts} and sigma overrides"
            )
            logger.info(f"Prompt: {prompt}")
            for sampler in samplers:
                sampler = f"{sampler}_karras"
                for steps in step_counts:
                    for sigma_override in sigma_overrides:
                        test_compvis(
                            model, prompt, sampler, steps=steps, sigma_override=sigma_override, output_dir=output_dir
                        )

            prompt = "boobs"
            logger.info(f"Prompt: {prompt}")
            test_compvis(
                model,
                prompt,
                "k_lms",
                filter_nsfw=True,
                safety_checker=mm.loaded_models["safety_checker"]["model"],
                output_dir=output_dir,
            )

            prompt = "cute anime girl"
            logger.info(f"Prompt: {prompt}")
            for denoising_strength in denoising_strengths:
                test_compvis(
                    model,
                    prompt,
                    "k_lms",
                    init_img=init_image,
                    denoising_strength=denoising_strength,
                    output_dir=output_dir,
                )

        logger.init_ok(f"Model {model}", status="Unloading")
        mm.unload_model(model)
        torch_gc()

    while True:
        print("Enter model name to load:")
        print(mm.available_models)
        model = input()
        if model == "exit":
            break
        print(f"Loading {model}")
        success = mm.load_model(model)
        print(f"Loading {model} successful: {success}")
        print("")


if __name__ == "__main__":
    test()
