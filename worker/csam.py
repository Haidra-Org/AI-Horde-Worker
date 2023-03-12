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
    "tween": 0.25,
    "tweens": 0.25,
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
    "shota",
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
            "child": 0.015,
            "children": 0.015,
            "lolicon": 0.01,
        },
    },
    {
        "regex": re.compile(r"pig ?tails", re.IGNORECASE),
        "adjustments": {
            "tween": 0.007,
            "tweens": 0.007,
            "child": 0.01,
            "children": 0.01,
            "lolicon": 0.005,
            "toddler": 0.01,
            "toddlers": 0.01,
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
            "child": -0.01,
            "children": -0.01,
            "toddler": -0.02,
            "toddlers": -0.02,
            "infant": -0.02,
            "infants": -0.02,
            "teen": -0.005,
            "teens": -0.005,
            "tween": -0.005,
            "tweens": -0.005,
        },
    },
    {
        "regex": re.compile(r"school|grade|classroom", re.IGNORECASE),
        "adjustments": {
            "child": 0.02,
            "children": 0.02,
            "toddler": 0.005,
            "toddlers": 0.005,
            "teen": 0.02,
            "teens": 0.02,
            "tween": 0.02,
            "tweens": 0.02,
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


CONTROL_WORD_ADJUSTMENTS = [
    {
        "control": ("pregnant", 0.21),
        "adjustments": [
            ("infant", -0.03),
            ("infants", -0.03),
            ("toddler", -0.02),
            ("toddlers", -0.02),
            ("child", -0.01),
            ("children", -0.01),
        ],
    },
    {
        "control": ("anime", 0.23),
        "adjustments": [
            ("teen", -0.03),
            ("teens", -0.03),
        ],
    },
    {
        "control": ("shota", 0.2),
        "adjustments": [
            ("teen", 0.005),
            ("teens", 0.005),
            ("tween", 0.01),
            ("tweens", 0.01),
            ("child", 0.015),
            ("children", 0.015),
        ],
    },
]
weight_remover = re.compile(r"\((.*?):\d+\.\d+\)")
whitespace_remover = re.compile(r"(\s(\w)){3,}\b")
whitespace_converter = re.compile(r"([^\w\s]|_)")


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
    prompt_tweaks = {}
    for entry in PROMPT_BOOSTS:
        prompt_re = entry["regex"].search(prompt)
        if prompt_re:
            for adjust_word in entry["adjustments"]:
                #  The below prevents us from increasing the plural and the singlar above the threshold
                # due to the boost. This prevents us from hitting the threshold with something like
                # teen + teens due to boosts
                if adjust_word in PAIRS and similarity_result[PAIRS[adjust_word]] > UNDERAGE_CONTEXT[adjust_word]:
                    continue
                if adjust_word not in prompt_tweaks:
                    prompt_tweaks[adjust_word] = []
                prompt_tweaks[adjust_word].append(prompt_re.group())
                similarity_result[adjust_word] += entry["adjustments"][adjust_word]
    poc_elapsed_time = time.time() - poc_start
    is_csam = False
    found_uc = []
    # For some reason clip associates infant with pregnant women a lot.
    # So to avoid censoring pregnant women, when they're drawn we reduce
    # the weight of "infant"
    adjustments = {}
    for control in CONTROL_WORD_ADJUSTMENTS:
        control_word, threshold = control["control"]
        if similarity_result[control_word] > threshold:
            for adjust_word, weight_adjustment in control["adjustments"]:
                if adjust_word in PAIRS and similarity_result[PAIRS[adjust_word]] > UNDERAGE_CONTEXT[adjust_word]:
                    continue
                similarity_result[adjust_word] += weight_adjustment
                if adjust_word not in adjustments:
                    adjustments[adjust_word] = []
                adjustments[adjust_word].append(control_word)
    for u_c in UNDERAGE_CONTEXT:
        if similarity_result[u_c] > UNDERAGE_CONTEXT[u_c]:
            found_uc.append(
                {
                    "word": u_c,
                    "similarity": similarity_result[u_c],
                    "threshold": UNDERAGE_CONTEXT[u_c],
                    "prompt_tweaks": prompt_tweaks.get(u_c),
                    "adjustments": adjustments.get(u_c),
                }
            )
    # When the value for some underage context is too high, it goes critical and we triple the suspicion
    for u_c in UNDERAGE_CRITICAL:
        if similarity_result[u_c] > UNDERAGE_CRITICAL[u_c]:
            found_uc.append(
                {
                    "word": u_c,
                    "similarity": similarity_result[u_c],
                    "threshold": UNDERAGE_CRITICAL[u_c],
                    "prompt_tweaks": prompt_tweaks.get(u_c),
                    "adjustments": adjustments.get(u_c),
                    "critical": True,
                }
            )
            found_uc.append(
                {
                    "word": u_c,
                    "similarity": similarity_result[u_c],
                    "threshold": UNDERAGE_CRITICAL[u_c],
                    "prompt_tweaks": prompt_tweaks.get(u_c),
                    "adjustments": adjustments.get(u_c),
                    "critical": True,
                }
            )
    found_lewd = []
    for l_c in LEWD_CONTEXT:
        if similarity_result[l_c] > LEWD_CONTEXT[l_c]:
            found_lewd.append(
                {
                    "word": l_c,
                    "similarity": similarity_result[l_c],
                    "threshold": LEWD_CONTEXT[l_c],
                    "prompt_tweaks": prompt_tweaks.get(l_c),
                    "adjustments": adjustments.get(l_c),
                }
            )
    if len(found_uc) >= 3 and len(found_lewd) >= 1:
        is_csam = True
    logger.info(f"Similarity Result after {poc_elapsed_time} seconds - Result = {is_csam}")
    return is_csam, similarity_result, {"found_uc": found_uc, "found_lewd": found_lewd}


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
