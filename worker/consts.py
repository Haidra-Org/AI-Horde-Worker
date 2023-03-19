BRIDGE_VERSION = 18
BRIDGE_CONFIG_FILE = "bridgeData.yaml"
POST_PROCESSORS_NATAILI_MODELS = {
    "GFPGAN", 
    "RealESRGAN_x4plus", 
    "RealESRGAN_x4plus_anime_6B", 
    "CodeFormers", 
}
KNOWN_POST_PROCESSORS = POST_PROCESSORS_NATAILI_MODELS | {
    "strip_background",
}
KNOWN_INTERROGATORS = {
    "ViT-L/14", 
}
