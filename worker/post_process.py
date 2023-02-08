"""Post process images"""
from nataili.codeformers import codeformers
from nataili.esrgan import esrgan
from nataili.gfpgan import gfpgan
from nataili.util.logger import logger

KNOWN_POST_PROCESSORS = {"GFPGAN": gfpgan, "RealESRGAN_x4plus": esrgan, "CodeFormers": codeformers}


def post_process(model, image, model_manager):
    """This is the post-processing function,
    it takes the model name, and the image, and returns the post processed image"""
    if model not in KNOWN_POST_PROCESSORS:
        logger.warning(f"Post processor {model} is unknown. Returning original image")
        return image

    if model not in model_manager.loaded_models:
        logger.init(f"{model}", status="Loading")
        model_manager.load(model)
        logger.init_ok(f"{model}", status="Success")

    pprocessor = KNOWN_POST_PROCESSORS[model]
    post_processor = pprocessor(
        model_manager.loaded_models[model],
        save_individual_images=False,
    )

    post_processor(input_image=image, strength=1.0)
    return post_processor.output_images[0]
