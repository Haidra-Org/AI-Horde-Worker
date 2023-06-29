BRIDGE_VERSION = 23
RELEASE = f"{BRIDGE_VERSION}.1.0"
BRIDGE_CONFIG_FILE = "bridgeData.yaml"
KNOWN_UPSCALERS = {
    "RealESRGAN_x4plus",
    "RealESRGAN_x2plus",
    "RealESRGAN_x4plus_anime_6B",
    "NMKD_Siax",
    "4x_AnimeSharp",
}

# The the order of this list is the order the face fixers will be applied if more than one is specified in a job
KNOWN_FACE_FIXERS = {
    "GFPGAN",
    "CodeFormers",
}
POST_PROCESSORS_HORDELIB_MODELS = KNOWN_UPSCALERS | KNOWN_FACE_FIXERS
KNOWN_POST_PROCESSORS = POST_PROCESSORS_HORDELIB_MODELS | {
    "strip_background",
}
KNOWN_INTERROGATORS = {
    "ViT-L/14",
}
