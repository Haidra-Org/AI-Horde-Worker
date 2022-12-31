import requests
from PIL import Image

from nataili.inference.diffusers.depth2img import Depth2Img
from nataili.model_manager import ModelManager
from nataili.util.logger import logger

mm = ModelManager(download=False)

mm.init()

model = "Stable Diffusion 2 Depth"

if model not in mm.available_models:
    logger.error(f"Model {model} not available", status=False)
    logger.init(f"Downloading {model}", status="Downloading")
    mm.download_model(model)
    logger.init_ok(f"Downloaded {model}", status=True)


logger.init(f"Model: {model}", status="Loading")
success = mm.load_model(model)
logger.init_ok(f"Loading {model}", status=success)


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
init_image = Image.open(requests.get(url, stream=True).raw)

generator = Depth2Img(
    pipe=mm.loaded_models[model]["model"],
    device=mm.loaded_models[model]["device"],
    output_dir="bridge_generations",
    filter_nsfw=False,
    disable_voodoo=True,
)

prompt = "two tigers ### bad, deformed, ugly, bad anatomy"
for iter in range(5):
    output = generator.generate(prompt=prompt, input_img=init_image, height=480, width=640)
