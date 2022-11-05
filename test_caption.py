import time
import PIL
from nataili.model_manager import ModelManager
from nataili.blip.caption import Caption
from nataili.util import logger

image = PIL.Image.open("01.png").convert("RGB")

mm = ModelManager()
mm.init()

model = "BLIP"

if model not in mm.available_models:
    logger.error("BLIP not available")
    logger.info("Downloading BLIP")
    mm.download_model(model)

tic = time.time()
logger.info("Loading BLIP")
success = mm.load_model(model)
logger.init_ok("Loading BLIP", status=success)
toc = time.time()
logger.init_ok(f"Loading {model}: Took {toc-tic} seconds", status=success)
start = time.time()

for i in range(100):
    caption = Caption(mm.loaded_models[model]["model"], mm.loaded_models[model]["device"])(image)
    logger.info(caption)