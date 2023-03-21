BRIDGE_VERSION = 19
BRIDGE_CONFIG_FILE = "bridgeData.yaml"
KNOWN_UPSCALERS = {
    "RealESRGAN_x4plus",
    "RealESRGAN_x4plus_anime_6B",
    "NMKD_Siax",
    "4x_AnimeSharp",
}
KNOWN_FACE_FIXERS = {
    "GFPGAN",
    "CodeFormers",
}
POST_PROCESSORS_NATAILI_MODELS = KNOWN_UPSCALERS | KNOWN_FACE_FIXERS
KNOWN_POST_PROCESSORS = POST_PROCESSORS_NATAILI_MODELS | {
    "strip_background",
}
KNOWN_INTERROGATORS = {
    "ViT-L/14",
}
