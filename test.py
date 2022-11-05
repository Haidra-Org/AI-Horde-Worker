import time

from nataili.inference.compvis import CompVis

from nataili.model_manager import ModelManager
from nataili.util.cache import torch_gc
from nataili.util.logger import logger

init_image = "./01.png"

mm = ModelManager()

mm.init()
logger.debug("Available dependencies:")
for dependency in mm.available_dependencies:
    logger.debug(dependency)

logger.debug("Available models:")
for model in mm.available_models:
    logger.debug(model)

models_to_load = [
    # 'stable_diffusion',
    # 'waifu_diffusion',
    "trinart",
    # 'GFPGAN', 'RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B',
    # 'BLIP', 'ViT-L/14', 'ViT-g-14', 'ViT-H-14'
]
logger.init(f"{models_to_load}", status="Loading")


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
            logger.debug(f"Running inference on {model}")
            logger.info('Testing txt2img with prompt "collosal corgi"')

            t2i = CompVis(
                mm.loaded_models[model]["model"],
                mm.loaded_models[model]["device"],
                "test_output",
                disable_voodoo=True,
            )
            t2i.generate("collosal corgi")

            torch_gc()

            logger.info('Testing nsfw filter with prompt "boobs"')

            t2i = CompVis(
                mm.loaded_models[model]["model"],
                mm.loaded_models[model]["device"],
                "test_output",
                filter_nsfw=True,
                safety_checker=mm.loaded_models["safety_checker"]["model"],
                disable_voodoo=True,
            )
            t2i.generate("boobs")

            torch_gc()

            logger.info('Testing img2img with prompt "cute anime girl"')

            i2i = CompVis(
                mm.loaded_models[model]["model"],
                mm.loaded_models[model]["device"],
                "test_output",
                disable_voodoo=True,
            )
            # init_img = PIL.Image.open(init_img)
            i2i.generate("cute anime girl", init_img=init_image)
            torch_gc()

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
