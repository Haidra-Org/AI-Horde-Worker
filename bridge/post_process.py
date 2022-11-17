import PIL

from nataili.model_manager import ModelManager
from nataili.postprocess.gfpgan import gfpgan
from nataili.upscalers.realesrgan import realesrgan
from nataili.util.logger import logger

KNOWN_POST_PROCESSORS = {
    "GFPGAN": gfpgan,
    "RealESRGAN_x4plus": realesrgan,
}

def post_process(model, image, model_manager):
    if model not in KNOWN_POST_PROCESSORS:
        logger.warning(f"Post processor {model} is unknown. Returning original image")
        return(image)
    if model not in model_manager.available_models:
        logger.warning(f"{model} not available")
        logger.init(f"{model}", status="Downloading")
        model_manager.download_model(model)
        logger.init_ok(f"{model}", status="Downloaded")

    if not model_manager.is_model_loaded(model):
        logger.init(f"{model}", status="Loading")
        success = model_manager.load_model(model)
        logger.init_ok(f"{model}", status="Success")

    pprocessor = KNOWN_POST_PROCESSORS[model]
    pp = pprocessor(
        model_manager.loaded_models[model]["model"],
        model_manager.loaded_models[model]["device"],
        save_individual_images = False,
    )

    results = pp(input_image=image, strength=1.0)
    return pp.output_images[0]
