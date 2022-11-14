from nataili.model_manager import ModelManager
import torch
from nataili.inference.aitemplate.ait_pipeline import StableDiffusionAITPipeline
from nataili.inference.aitemplate.AITemplate import AITemplate
import creds
from nataili.util.logger import logger

hf_auth = {"username": creds.hf_username, "password": creds.hf_password}
mm = ModelManager(hf_auth=hf_auth)
mm.init()
if len(mm.available_aitemplates) == 0:
    logger.info('No available aitemplates')
    logger.info('Downloading aitemplate')
    mm.download_ait(mm.recommended_gpu[0]["sm"])
mm.load_ait()

def run():
    while True:
        logger.info('init')
        ait = AITemplate(mm.loaded_models['ait']['pipe'], 'ait_output')
        logger.info('start')
        ait.generate('corgi', ddim_steps = 10)
        logger.info('end')

if __name__ == "__main__":
    run()
