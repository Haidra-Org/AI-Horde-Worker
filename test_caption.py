import time

import PIL

from nataili.blip.caption import Caption
from nataili.model_manager import ModelManager
from nataili.util import logger

image = PIL.Image.open("01.png").convert("RGB")

mm = ModelManager(download=False)
mm.init()


def test_caption(model, fast_test=True):
    if model not in mm.available_models:
        logger.error(f"{model} not available")
        logger.init(f"{model}", status="Downloading")
        mm.download_model(model)

    if model not in mm.loaded_models:
        tic = time.time()
        logger.init(f"{model}", status="Loading")
        mm.load_model(model)
        logger.init_ok(f"Loading {model}", status="Success")
        toc = time.time()
        logger.init_ok(f"Loading {model}: Took {toc-tic} seconds", status="Success")

    start = time.time()

    blip = Caption(mm.loaded_models[model]["model"], mm.loaded_models[model]["device"])

    if fast_test:
        logger.message(f"Fast test for {model}")
        logger.generation(f"caption: {blip(image, sample=False)} - sample: False")
        logger.generation(f"caption: {blip(image, sample=True)} - sample: True")
    else:
        logger.message(f"Slow test for {model}")
        num_beams = [3, 7]
        min_length = [10, 20, 30]
        top_p = [0.9, 0.95]
        repetition_penalty = [1.0, 1.4]
        for n in num_beams:
            for m in min_length:
                for t in top_p:
                    for r in repetition_penalty:
                        caption = blip(
                            image, num_beams=n, min_length=m, max_length=m + 20, top_p=t, repetition_penalty=r
                        )
                        logger.generation(f"Beams: {n}, Min: {m}, TopP: {t}, RepP: {r}: {caption}")
        caption = blip(image, sample=False)
        logger.generation(f"caption: {caption} sample: False")
        caption = blip(image, sample=False, num_beams=5)
        logger.generation(f"caption: {caption} sample: False, num_beams=5")
        caption = blip(image, sample=False, num_beams=7, min_length=60, max_length=90)
        logger.generation(f"caption: {caption} sample: False, num_beams=7, min_length=60, max_length=90")

    end = time.time()

    logger.info(f"Total time: {end-start}")


test_caption("BLIP", fast_test=True)
test_caption("BLIP_Large", fast_test=True)
test_caption("BLIP", fast_test=False)
test_caption("BLIP_Large", fast_test=False)
