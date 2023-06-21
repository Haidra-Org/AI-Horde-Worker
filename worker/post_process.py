"""Post process images"""
import rembg
from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager
from loguru import logger

hordelib = HordeLib()


def post_process(model, image, strength):  # noqa: ARG001
    """This is the post-processing function,
    it takes the model name, and the image, and returns the post processed image"""
    if model not in KNOWN_POST_PROCESSORS:
        logger.warning(f"Post processor {model} is unknown. Returning original image")
        return image

    if model != "strip_background" and not SharedModelManager.manager.is_model_loaded(model):
        logger.init(f"{model}", status="Loading")
        load_result = SharedModelManager.manager.load(model)
        if not load_result:
            logger.init_err(f"{model}", status="Error")
            return image
        logger.init_ok(f"{model}", status="Success")

    pprocessor = KNOWN_POST_PROCESSORS[model]
    payload = {
        "model": model,
        "source_image": image,
    }
    return pprocessor(payload)


# TODO: move to hordelib or ComfyUI
def strip_background(payload):
    session = rembg.new_session("u2net")
    image = rembg.remove(
        payload["source_image"],
        session=session,
        only_mask=False,
        alpha_matting=10,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    )
    del session
    return image


# At the bottom, as we need to define the method first
KNOWN_POST_PROCESSORS = {
    "RealESRGAN_x4plus": hordelib.image_upscale,
    "RealESRGAN_x2plus": hordelib.image_upscale,
    "RealESRGAN_x4plus_anime_6B": hordelib.image_upscale,
    "NMKD_Siax": hordelib.image_upscale,
    "4x_AnimeSharp": hordelib.image_upscale,
    "strip_background": strip_background,
    "GFPGAN": hordelib.image_facefix,
    "CodeFormers": hordelib.image_facefix,
}
