"""Post process images"""
import time

from nataili.clip.interrogate import Interrogator
from nataili.util.logger import logger


def check_for_csam(clip_model, image):
    """This is the post-processing function,
    it takes the model name, and the image, and returns the post processed image"""
    poc_start = time.time()
    interrogator = Interrogator(clip_model)
    underage_context = {
        "lolicon": 0.2,
        "child": 0.188,
        "children": 0.188,
        "teen": 0.21,
        "teens": 0.21,
        "infant": 0.19,
        "infants": 0.19,
        "toddler": 0.19,
        "toddlers": 0.19,
        "tween":0.188,
        "tweens":0.188,
    }
    underage_critical = {
        "lolicon": 0.25,
        "child": 0.225,
        "children": 0.225,
        "toddler": 0.22,
        "toddlers": 0.22,
        "infant": 0.22,
        "infants": 0.22,
        "teen": 0.26,
        "teens": 0.26,
    }
    lewd_context = {
        "porn": 0.2,
        "naked": 0.195,
        "hentai": 0.25,
        "orgy": 0.21,
        "nudity":0.195,
        "lesbian scene": 0.2,
        "gay scene": 0.2,
    }
    control_words = [
        "pregnant",
        "anime",
    ]
    test_words = [
        "family",
    ]
    word_list = list(underage_context.keys()) + list(lewd_context.keys()) + control_words + test_words
    similarity_result = interrogator(image=image, text_array=word_list, similarity=True)
    poc_elapsed_time = time.time() - poc_start
    is_csam = False
    found_uc = 0
    # For some reason clip associates infant with pregnant women a lot. 
    # So to avoid censoring pregnant women, when they're drawn we reduce 
    # the weight of "infant"
    if similarity_result["default"]["pregnant"] > 0.21:
        similarity_result["default"]["infant"] -= 0.03
        similarity_result["default"]["infants"] -= 0.03
    if similarity_result["default"]["anime"] > 0.23:
        similarity_result["default"]["teen"] -= 0.03
        similarity_result["default"]["teen"] -= 0.03
    for u_c in underage_context:
        if similarity_result["default"][u_c] > underage_context[u_c]:
            found_uc += 1
    # When the value for some underage context is too high, it goes critical and we triple the suspicion
    for u_c in underage_critical:
        if similarity_result["default"][u_c] > underage_critical[u_c]:
            found_uc += 2
    found_lewd = 0
    for l_c in lewd_context:
        if similarity_result["default"][l_c] > lewd_context[l_c]:
            found_lewd += 1
    if found_uc >= 3 and found_lewd >= 1:
        is_csam = True
    logger.info(f"Similarity Result after {poc_elapsed_time} seconds - Result = {is_csam}")
    return is_csam, similarity_result['default']
        
