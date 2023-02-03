from nataili.model_manager import ModelManager
from nataili.util.logger import logger
from nataili.clip.coca import CoCa
from PIL import Image

image = Image.open("01.png").convert("RGB")

mm = ModelManager()

model = "coca_ViT-L-14"

if model not in mm.available_models:
    logger.error(f"Model {model} not available", status=False)
    logger.init(f"Downloading {model}", status="Downloading")
    mm.download_model(model)
    logger.init_ok(f"Downloaded {model}", status=True)


logger.init(f"Model: {model}", status="Loading")
success = mm.load_model(model)
logger.init_ok(f"Loading {model}", status=success)

coca = CoCa(
    mm.loaded_models[model]["model"],
    mm.loaded_models[model]["transform"],
    mm.loaded_models[model]["device"],
    mm.loaded_models[model]["half_precision"],
)

logger.generation(coca(image))
