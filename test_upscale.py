import PIL

from nataili.model_manager import ModelManager
from nataili.upscalers.realesrgan import realesrgan
from nataili.util.logger import logger

image = PIL.Image.open("./01.png").convert("RGB")

mm = ModelManager()

mm.init()

model = "RealESRGAN_x4plus"

if model not in mm.available_models:
    logger.error(f"Model {model} not available", status=False)
    logger.init(f"Downloading {model}", status="Downloading")
    mm.download_model(model)
    logger.init_ok(f"Downloaded {model}", status=True)

logger.init(f"Model: {model}", status="Loading")
success = mm.load_model(model)
logger.init_ok(f"Loading {model}", status=success)

upscaler = realesrgan(
    mm.loaded_models[model]["model"],
    mm.loaded_models[model]["device"],
    "./",
)

results = upscaler(input_image=image)
images = upscaler.output_images
