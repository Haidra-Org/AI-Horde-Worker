import time
import PIL

from nataili.clip.interrogate import Interrogator
from nataili.model_manager import ModelManager
from nataili.util.logger import logger

image = PIL.Image.open("./01.png").convert("RGB")

mm = ModelManager()

mm.init()

model = "ViT-L/14"

if model not in mm.available_models:
    logger.error(f"Model {model} not available", status=False)
    logger.init(f"Downloading {model}", status="Downloading")
    mm.download_model(model)
    logger.init_ok(f"Downloaded {model}", status=True)

logger.init(f"Model: {model}", status="Loading")
success = mm.load_model(model)
logger.init_ok(f"Loading {model}", status=success)

interrogator = Interrogator(
    mm.loaded_models[model]["model"],
    mm.loaded_models[model]["preprocess"],
    mm.loaded_models[model]["data_lists"],
    mm.loaded_models[model]["device"],
    batch_size=100,
)

for _ in range(100):
    results = interrogator(image)
    logger.info(results)
