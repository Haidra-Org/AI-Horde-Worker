"""Post process images"""
import re
import time

from nataili.clip.interrogate import Interrogator
from nataili.util.logger import logger
from unidecode import unidecode

UNDERAGE_CONTEXT = {
    "lolicon": 0.2,
    "child": 0.188,
    "children": 0.188,
    "teen": 0.21,
    "teens": 0.21,
    "infant": 0.19,
    "infants": 0.19,
    "toddler": 0.19,
    "toddlers": 0.19,
    "tween": 0.188,
    "tweens": 0.188,
}
UNDERAGE_CRITICAL = {
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
LEWD_CONTEXT = {
    "porn": 0.2,
    "naked": 0.195,
    "hentai": 0.25,
    "orgy": 0.21,
    "nudity": 0.195,
    "lesbian scene": 0.2,
    "gay scene": 0.2,
}
CONTROL_WORDS = [
    "pregnant",
    "anime",
]
TEST_WORDS = []

PROMPT_BOOSTS = [
    {
        "regex": re.compile(r"\bgirl|\bboy\b|nina", re.IGNORECASE),
        "adjustments": {
            "teen": 0.015,
            "teens": 0.015,
            "tween": 0.005,
            "tweens": 0.005,
        },
    },
    {
        "regex": re.compile(r"flat chest", re.IGNORECASE),
        "adjustments": {
            "tween": 0.015,
            "tweens": 0.015,
        },
    },
    {
        "regex": re.compile(r"baby|toddler|infant", re.IGNORECASE),
        "adjustments": {
            "infant": 0.02,
            "infants": 0.02,
            "toddler": 0.02,
            "toddlers": 0.02,
            "child": 0.01,
            "children": 0.01,
        },
    },
    {
        "regex": re.compile(r"child|kin?d|angel", re.IGNORECASE),
        "adjustments": {
            "infant": 0.01,
            "infants": 0.01,
            "toddler": 0.01,
            "toddlers": 0.01,
            "child": 0.02,
            "children": 0.02,
        },
    },
    {
        "regex": re.compile(r"sister|brother|\bbro\b|\bsis\b|daughter|tochter|\bson\b|twin", re.IGNORECASE),
        "adjustments": {
            "child": 0.005,
            "children": 0.005,
            "teen": 0.008,
            "teens": 0.008,
        },
    },
    {
        "regex": re.compile(r"small|little|\btiny\b|petite", re.IGNORECASE),
        "adjustments": {
            "child": 0.01,
            "children": 0.01,
            "toddler": 0.01,
            "toddlers": 0.01,
            "tween": 0.005,
            "tweens": 0.005,
        },
    },
    {
        "regex": re.compile(r"woman|adult|old", re.IGNORECASE),
        "adjustments": {
            "child": -0.02,
            "children": -0.02,
            "toddler": -0.02,
            "toddlers": -0.02,
            "infant": -0.02,
            "infants": -0.02,
            "teen": -0.01,
            "teens": -0.01,
            "tween": -0.01,
            "tweens": -0.01,
        },
    },
]
NEGPROMPT_BOOSTS = {"mature" "old" "adult" "elderly", "middle aged"}
NEGPROMPT_DEBUFFS = {"young" "little" "child"}

PAIRS = {
    "tween": "tweens",
    "tweens": "tween",
    "teen": "teens",
    "teens": "teen",
    "infant": "infants",
    "infants": "infant",
    "toddler": "toddlers",
    "toddlers": "toddler",
    "child": "children",
    "children": "child",
}

weight_remover = re.compile(r"\((.*?):\d+\.\d+\)")
whitespace_remover = re.compile(r"(\s(\w)){3,}\b")
whitespace_converter = re.compile(r"[^\w\s]")


def check_for_csam(clip_model, image, prompt):
    """This is the post-processing function,
    it takes the model name, and the image, and returns the post processed image"""
    poc_start = time.time()
    interrogator = Interrogator(clip_model)

    word_list = list(UNDERAGE_CONTEXT.keys()) + list(LEWD_CONTEXT.keys()) + CONTROL_WORDS + TEST_WORDS
    similarity_result = interrogator(image=image, text_array=word_list, similarity=True)["default"]
    prompt, negprompt = normalize_prompt(prompt)
    for entry in NEGPROMPT_BOOSTS:
        if negprompt and entry in negprompt:
            for weight in UNDERAGE_CONTEXT:
                similarity_result[weight] += 0.005
    for entry in NEGPROMPT_DEBUFFS:
        if negprompt and entry in negprompt:
            for weight in UNDERAGE_CONTEXT:
                similarity_result[weight] -= 0.005
    for entry in PROMPT_BOOSTS:
        if entry["regex"].search(prompt):
            for weight in entry["adjustments"]:
                #  The below prevents us from increasing the plural and the singlar above the threshold
                # due to the boost. This prevents us from hitting the threshold with something like
                # teen + teens due to boosts
                if weight in PAIRS and similarity_result[PAIRS[weight]] > UNDERAGE_CONTEXT[weight]:
                    continue
                similarity_result[weight] += entry["adjustments"][weight]
    poc_elapsed_time = time.time() - poc_start
    is_csam = False
    found_uc = 0
    # For some reason clip associates infant with pregnant women a lot.
    # So to avoid censoring pregnant women, when they're drawn we reduce
    # the weight of "infant"
    if similarity_result["pregnant"] > 0.21:
        similarity_result["infant"] -= 0.03
        similarity_result["infants"] -= 0.03
        similarity_result["toddler"] -= 0.02
        similarity_result["toddlers"] -= 0.02
        similarity_result["child"] -= 0.01
        similarity_result["children"] -= 0.01
    if similarity_result["anime"] > 0.23:
        similarity_result["teen"] -= 0.03
        similarity_result["teens"] -= 0.03
    for u_c in UNDERAGE_CONTEXT:
        if similarity_result[u_c] > UNDERAGE_CONTEXT[u_c]:
            found_uc += 1
    # When the value for some underage context is too high, it goes critical and we triple the suspicion
    for u_c in UNDERAGE_CRITICAL:
        if similarity_result[u_c] > UNDERAGE_CRITICAL[u_c]:
            found_uc += 2
    found_lewd = 0
    for l_c in LEWD_CONTEXT:
        if similarity_result[l_c] > LEWD_CONTEXT[l_c]:
            found_lewd += 1
    if found_uc >= 3 and found_lewd >= 1:
        is_csam = True
    logger.info(f"Similarity Result after {poc_elapsed_time} seconds - Result = {is_csam}")
    return is_csam, similarity_result


def normalize_prompt(prompt):
    """Prepares the prompt to be scanned by the regex, by removing tricks one might use to avoid the filters"""
    negprompt = None
    if "###" in prompt:
        prompt, negprompt = prompt.split("###", 1)
    prompt = weight_remover.sub(r"\1", prompt)
    prompt = whitespace_converter.sub(" ", prompt)
    for match in re.finditer(whitespace_remover, prompt):
        trim_match = match.group(0).strip()
        replacement = re.sub(r"\s+", "", trim_match)
        prompt = prompt.replace(trim_match, replacement)
    prompt = re.sub(r"\s+", " ", prompt)
    # Remove all accents
    prompt = unidecode(prompt)
    if negprompt:
        negprompt = weight_remover.sub(r"\1", negprompt)
        negprompt = whitespace_converter.sub(" ", negprompt)
        for match in re.finditer(whitespace_remover, negprompt):
            trim_match = match.group(0).strip()
            replacement = re.sub(r"\s+", "", trim_match)
            negprompt = negprompt.replace(trim_match, replacement)
        negprompt = re.sub(r"\s+", " ", negprompt)
        # Remove all accents
        negprompt = unidecode(negprompt)
    return prompt, negprompt
