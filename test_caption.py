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
        logger.info(f"Downloading {model}")
        mm.download_model(model)

    if model not in mm.loaded_models:
        tic = time.time()
        logger.info(f"Loading {model}")
        success = mm.load_model(model)
        logger.init_ok(f"Loading {model}", status=success)
        toc = time.time()
        logger.init_ok(f"Loading {model}: Took {toc-tic} seconds", status=success)

    start = time.time()

    blip = Caption(mm.loaded_models[model]["model"], mm.loaded_models[model]["device"])

    if fast_test:
        logger.info(f"Fast test for {model}")
        logger.info(f"caption: {blip(image, sample=False)} sample: False")
        logger.info(f"caption: {blip(image, sample=True)} sample: True")
    else:
        logger.info(f"Slow test for {model}")
        num_beams = [3, 5]
        max_length = [30, 50, 70]
        top_p = [0.9, 0.95]
        repetition_penalty = [1.0, 1.2]
        for n in num_beams:
            for m in max_length:
                for t in top_p:
                    for r in repetition_penalty:
                        caption = blip(image, num_beams=n, max_length=m, top_p=t, repetition_penalty=r)
                        logger.info(f"Num Beams: {n}, Max Length: {m}, Top P: {t}, Repetition Penalty: {r}")
                        logger.info(caption)
        caption = blip(image, sample=False)
        logger.info(f"caption: {caption} sample: False")
        caption = blip(image, sample=False, num_beams=5)
        logger.info(f"caption: {caption} sample: False, num_beams=5")
        caption = blip(image, sample=False, num_beams=7, max_length=50)
        logger.info(f"caption: {caption} sample: False, num_beams=7, max_length=50")

    end = time.time()

    logger.info(f"Total time: {end-start}")


test_caption("BLIP", fast_test=True)
test_caption("BLIP_Large", fast_test=True)
test_caption("BLIP", fast_test=False)
test_caption("BLIP_Large", fast_test=False)
