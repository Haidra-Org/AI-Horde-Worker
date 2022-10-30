from nataili.inference.compvis.img2img import img2img
from nataili.model_manager import ModelManager
from nataili.inference.compvis.txt2img import txt2img
from nataili.util.cache import torch_gc
from nataili.util.logger import logger
import time
import PIL


init_image = './01.png'

mm = ModelManager()

mm.init()
logger.debug(f'Available dependencies:')
for dependency in mm.available_dependencies:
    logger.debug(dependency)

logger.debug(f'Available models:')
for model in mm.available_models:
    logger.debug(model)

models_to_load = [#'stable_diffusion',
                  #'waifu_diffusion',
                  'trinart',
                  #'GFPGAN', 'RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B',
                  #'BLIP', 'ViT-L/14', 'ViT-g-14', 'ViT-H-14'
                  ]
logger.init(f'{models_to_load}', status="Loading")

@logger.catch
def test():
    tic = time.time()
    model = 'safety_checker'
    logger.init(f'Model: {model}', status="Loading")
    success = mm.load_model(model)
    toc = time.time()
    logger.init_ok(f'Loading {model}: Took {toc-tic} seconds', status=success)
    for model in models_to_load:
        torch_gc()
        tic = time.time()
        logger.init(f'Model: {model}', status="Loading")
        
        success = mm.load_model(model, use_voodoo=True)

        toc = time.time()
        logger.init_ok(f'Loading {model}: Took {toc-tic} seconds', status=success)
        torch_gc()

        if model in ['stable_diffusion', 'waifu_diffusion', 'trinart']:
            logger.debug(f'Running inference on {model}')
            logger.info(f'Testing txt2img with prompt "collosal corgi"')

            t2i = txt2img(mm.loaded_models[model]["model"], mm.loaded_models[model]["device"], 'test_output', use_voodoo=True)
            t2i.generate('collosal corgi')

            torch_gc()

            logger.info(f'Testing nsfw filter with prompt "boobs"')

            t2i = txt2img(mm.loaded_models[model]["model"], mm.loaded_models[model]["device"], 'test_output', filter_nsfw=True, safety_checker=mm.loaded_models['safety_checker']['model'], use_voodoo=True)
            t2i.generate('boobs')

            torch_gc()

            logger.info(f'Testing img2img with prompt "cute anime girl"')

            i2i = img2img(mm.loaded_models[model]["model"], mm.loaded_models[model]["device"], 'test_output', use_voodoo=True)
            init_img = PIL.Image.open(init_image)
            i2i.generate('cute anime girl', init_img)
            torch_gc()

        logger.init_ok(f'Model {model}', status="Unloading")
        mm.unload_model(model)
        torch_gc()

    while True:
        print('Enter model name to load:')
        print(mm.available_models)
        model = input()
        if model == 'exit':
            break
        print(f'Loading {model}')
        success = mm.load_model(model)
        print(f'Loading {model} successful: {success}')
        print('')


if __name__ == "__main__":
    test()
