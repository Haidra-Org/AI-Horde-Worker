"""Post process images"""
import time

import regex as re
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
    "lesbian scene": 0.22,
    "gay scene": 0.22,
}
CONTROL_WORDS = [
    "pregnant",
    "anime",
    "shota",
    "east asian",
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
            "child": 0.005,
            "children": 0.005,
            "toddler": 0.005,
            "toddlers": 0.005,
            "tween": 0.003,
            "tweens": 0.003,
        },
    },
    {
        "regex": re.compile(r"woman|adult|(?<!years? )old", re.IGNORECASE),
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
        "regex": re.compile(r"school|grade|\b(?<!high )class\b", re.IGNORECASE),
        "adjustments": {
            "child": 0.01,
            "children": 0.01,
            "toddler": 0.002,
            "toddlers": 0.002,
            "teen": 0.02,
            "teens": 0.02,
            "tween": 0.015,
            "tweens": 0.015,
        },
    },
    {
        "regex": re.compile(r"kitten", re.IGNORECASE),
        "adjustments": {
            "child": 0.025,
            "children": 0.02,
            "toddler": 0.025,
            "toddlers": 0.02,
            "teen": 0.01,
            "teens": 0.01,
            "tween": 0.01,
            "tweens": 0.01,
            "infant": 0.01,
            "infants": 0.01,
        },
    },
    {
        "regex": re.compile(r"realistic", re.IGNORECASE),
        "adjustments": {
            "lolicon": -0.015,
        },
    },
]
NEGPROMPT_BOOSTS = {"mature", " old", "adult", "elderly", "middle aged"}
NEGPROMPT_DEBUFFS = {"young", "little", "child"}

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
            ("infant", -0.04),
            ("infants", -0.04),
            ("toddler", -0.03),
            ("toddlers", -0.03),
            ("child", -0.02),
            ("children", -0.02),
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
        "control": ("shota", 0.21),
        "adjustments": [
            ("teen", 0.005),
            ("teens", 0.005),
            ("tween", 0.01),
            ("tweens", 0.01),
            ("child", 0.013),
            ("children", 0.013),
        ],
    },
    {
        # For some reason, clip thinks all east asian woman are very child-like
        "control": ("east asian", 0.24),
        "adjustments": [
            ("teen", -0.025),
            ("child", -0.025),
            ("tween", -0.025),
        ],
    },
]
MODEL_TAG_ADJUSTMENTS = {
    "anime": [
        ("teen", -0.015),
        ("teens", -0.015),
        ("anime", 0.02),
        ("tween", -0.015),
        ("tweens", -0.015),
        ("child", -0.01),
        ("children", -0.01),
    ],
    "hentai": [
        ("hentai", 0.02),
    ],
}
NSFW_MODEL_ADJUSTMENTS = [
    ("nudity", 0.02),
    ("naked", 0.02),
    ("porn", 0.015),
    ("orgy", 0.01),
]
weight_remover = re.compile(r"\((.*?):\d+\.\d+\)")
whitespace_remover = re.compile(r"(\s(\w)){3,}\b")
whitespace_converter = re.compile(r"([^\w\s]|_)")


def check_for_csam(clip_model, image, prompt, model_info=None):
    """This is the post-processing function,
    it takes the model name, and the image, and returns the post processed image"""
    # return False, [], {}
    if not model_info:
        model_info = {}
    model_nsfw = model_info.get("nsfw", False)
    model_tags = model_info.get("tags")
    if not model_tags:
        model_tags = []
    poc_start = time.time()
    interrogator = Interrogator(clip_model)

    word_list = list(UNDERAGE_CONTEXT.keys()) + list(LEWD_CONTEXT.keys()) + CONTROL_WORDS + TEST_WORDS
    similarity_result = interrogator(image=image, text_array=word_list, similarity=True)["default"]
    poc_elapsed_time = time.time() - poc_start
    prompt, negprompt = normalize_prompt(prompt)
    prompt_tweaks = {}
    for entry in NEGPROMPT_BOOSTS:
        if negprompt and entry in negprompt:
            for adjust_word in UNDERAGE_CONTEXT:
                add_value_to_dict_array(prompt_tweaks, adjust_word, entry)
                similarity_result[adjust_word] += 0.005
    for entry in NEGPROMPT_DEBUFFS:
        if negprompt and entry in negprompt:
            for adjust_word in UNDERAGE_CONTEXT:
                add_value_to_dict_array(prompt_tweaks, adjust_word, entry)
                similarity_result[adjust_word] -= 0.005
    for entry in PROMPT_BOOSTS:
        if prompt_re := entry["regex"].search(prompt):
            for adjust_word in entry["adjustments"]:
                #  The below prevents us from increasing the plural and the singlar above the threshold
                # due to the boost. This prevents us from hitting the threshold with something like
                # teen + teens due to boosts
                if adjust_word in PAIRS and similarity_result[PAIRS[adjust_word]] > UNDERAGE_CONTEXT[adjust_word]:
                    continue
                add_value_to_dict_array(prompt_tweaks, adjust_word, prompt_re.group())
                similarity_result[adjust_word] += entry["adjustments"][adjust_word]
    # For some reason clip associates infant with pregnant women a lot.
    # So to avoid censoring pregnant women, when they're drawn we reduce
    # the weight of "infant"
    model_tweaks = {}
    if model_nsfw:
        for adjust_word, similarity_adjustment in NSFW_MODEL_ADJUSTMENTS:
            add_value_to_dict_array(model_tweaks, adjust_word, "nsfw")
            similarity_result[adjust_word] += similarity_adjustment
    for tag in [tag for tag in MODEL_TAG_ADJUSTMENTS if tag in model_tags]:
        for adjust_word, similarity_adjustment in MODEL_TAG_ADJUSTMENTS[tag]:
            add_value_to_dict_array(model_tweaks, adjust_word, tag)
            similarity_result[adjust_word] += similarity_adjustment
    adjustments = {}
    for control in CONTROL_WORD_ADJUSTMENTS:
        control_word, threshold = control["control"]
        # logger.info([similarity_result[control_word],control_word,threshold])
        if similarity_result[control_word] > threshold:
            for adjust_word, similarity_adjustment in control["adjustments"]:
                if adjust_word in PAIRS and similarity_result[PAIRS[adjust_word]] > UNDERAGE_CONTEXT[adjust_word]:
                    continue
                similarity_result[adjust_word] += similarity_adjustment
                add_value_to_dict_array(adjustments, adjust_word, control_word)
    found_uc = [
        {
            "word": u_c,
            "similarity": similarity_result[u_c],
            "threshold": UNDERAGE_CONTEXT[u_c],
            "prompt_tweaks": prompt_tweaks.get(u_c),
            "model_tweaks": model_tweaks.get(u_c),
            "adjustments": adjustments.get(u_c),
        }
        for u_c in UNDERAGE_CONTEXT
        if similarity_result[u_c] > UNDERAGE_CONTEXT[u_c]
    ]
    # When the value for some underage context is too high, it goes critical and we triple the suspicion
    for u_c in UNDERAGE_CRITICAL:
        if similarity_result[u_c] > UNDERAGE_CRITICAL[u_c]:
            found_uc.extend(
                (
                    {
                        "word": u_c,
                        "similarity": similarity_result[u_c],
                        "threshold": UNDERAGE_CRITICAL[u_c],
                        "prompt_tweaks": prompt_tweaks.get(u_c),
                        "adjustments": adjustments.get(u_c),
                        "model_tweaks": model_tweaks.get(u_c),
                        "critical": True,
                    },
                    {
                        "word": u_c,
                        "similarity": similarity_result[u_c],
                        "threshold": UNDERAGE_CRITICAL[u_c],
                        "prompt_tweaks": prompt_tweaks.get(u_c),
                        "adjustments": adjustments.get(u_c),
                        "model_tweaks": model_tweaks.get(u_c),
                        "critical": True,
                    },
                ),
            )
    found_lewd = [
        {
            "word": l_c,
            "similarity": similarity_result[l_c],
            "threshold": LEWD_CONTEXT[l_c],
            "prompt_tweaks": prompt_tweaks.get(l_c),
            "adjustments": adjustments.get(l_c),
            "model_tweaks": model_tweaks.get(l_c),
        }
        for l_c in LEWD_CONTEXT
        if similarity_result[l_c] > LEWD_CONTEXT[l_c]
    ]
    is_csam = bool(len(found_uc) >= 3 and found_lewd)
    logger.debug(f"Similarity Result after {poc_elapsed_time} seconds - Result = {is_csam}")
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


def add_value_to_dict_array(dict_to_modify, array_key, value):
    """Adds a value to an array stored in a dict key
    If the key does not exist, it is created
    """
    if array_key not in dict_to_modify:
        dict_to_modify[array_key] = []
    dict_to_modify[array_key].append(value)
