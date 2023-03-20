BRIDGE_VERSION = 19
BRIDGE_CONFIG_FILE = "bridgeData.yaml"
POST_PROCESSORS_NATAILI_MODELS = {
    "GFPGAN",
    "RealESRGAN_x4plus",
    "RealESRGAN_x4plus_anime_6B",
    "NMKD_Siax",
    "4x_AnimeSharp",
    "CodeFormers",
}
KNOWN_POST_PROCESSORS = POST_PROCESSORS_NATAILI_MODELS | {
    "strip_background",
}
KNOWN_INTERROGATORS = {
    "ViT-L/14",
}
