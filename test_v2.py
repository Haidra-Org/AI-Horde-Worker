import argparse
import time
import uuid

import PIL

from nataili import disable_xformers
from nataili.inference.compvis import CompVis
from nataili.model_manager import ModelManager
from nataili.util.cache import torch_gc
from nataili.util.logger import logger

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--disable_xformers",
    action="store_true",
    default=False,
    help=(
        "If specified this test will not try use xformers to speed up generations."
        " This should normally be automatic, but in case you need to disable it manually, you can do so here."
    ),
)

args = arg_parser.parse_args()

disable_xformers.toggle(args.disable_xformers)

init_image = PIL.Image.open("./01.png").convert("RGB")

mm = ModelManager(download=False) # TODO: update db reference

mm.init()
logger.debug("Available dependencies:")
for dependency in mm.available_dependencies:
    logger.debug(dependency)

logger.debug("Available models:")
for model in mm.available_models:
    logger.debug(model)

models_to_load = [
    "stable_diffusion_2.0",
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

        if model in ["stable_diffusion_2.0"]:
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
