"""Post process images"""
import time
from nataili.clip.interrogate import Interrogator
from nataili.util.logger import logger


def check_for_csam(clip_model, image):
    """This is the post-processing function,
    it takes the model name, and the image, and returns the post processed image"""
    poc_start = time.time()
    interrogator = Interrogator(
        clip_model
    )
    underage_context = {
        "loli": 0.2,
        "child": 0.195,
        "teen": 0.2,
        "baby": 0.2,
    }
    lewd_context = {
        "porn": 0.2,
        "nude": 0.2,
        "hentai": 0.27,
    }
    word_list = list(underage_context.keys()) + list(lewd_context.keys())
    similarity_result = interrogator(image=image, text_array=word_list, similarity=True)
    poc_elapsed_time = time.time() - poc_start
    is_csam = False
    found_uc = 0
    for u_c in underage_context:
        if similarity_result['default'][u_c] > underage_context[u_c]:
            found_uc += 1
    found_lewd = 0
    for l_c in lewd_context:
        if similarity_result['default'][l_c] > lewd_context[l_c]:
            found_lewd += 1
    if found_uc >= 3 and found_lewd >= 1:
        is_csam = True
    logger.info(f"Similarity Result after {poc_elapsed_time} seconds - Result = {is_csam} - Details = {similarity_result['default']}")
    return is_csam
        
