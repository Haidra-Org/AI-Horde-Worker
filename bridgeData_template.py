# The horde url
horde_url = "https://stablehorde.net"
# Give a cool name to your instance
worker_name = "My Awesome Instance"
# The api_key identifies a unique user in the horde
# Visit https://stablehorde.net/register to create one before you can join
api_key = "0000000000"
# Put other users whose prompts you want to prioritize.
# The owner's username is always included so you don't need to add it here, unless you want it to have lower priority than another user
priority_usernames = []
# The amount of power your system can handle
# 8 means 512*512. Each increase increases the possible resoluion by 64 pixes
# So if you put this to 2 (the minimum, your SD can only generate 64x64 pixels
# If you put this to 32, it is equivalent to 1024x1024 pixels
max_power = 8
# Set this to false, if you do not want your worker to receive requests for NSFW generations
nsfw = True
# Set this to True if you want your worker to censor NSFW generations. This will only be active is horde_nsfw == False
censor_nsfw = False
# A list of words which you do not want to your worker to accept
blacklist = []
# A list of words for which you always want to allow the NSFW censor filter, even when this worker is in NSFW mode
censorlist = []
# If set to False, this worker will no longer pick img2img jobs
allow_img2img = True
# If set to True, this worker will can pick inpainting jobs
allow_painting = True
# If set to False, this worker will no longer pick img2img jobs from unsafe IPs
allow_unsafe_ip = True
# The models to use. You can select a different main model, or select more than one if you have enough VRAM
# The last model in this list takes priority when the client accepts more than 1
# if you do not know which models you can add here, use the below command
# python show_available_models.py
models_to_load = [
    "stable_diffusion", # This is the standard compvis model. It is not using Diffusers (yet)
    ## Specialized Style models
    # "trinart",
    # "Furry Epoch",
    # "Yiffy",
    # "waifu_diffusion",
    ## Dreambooth Models
    # "Arcane Diffusion",
    # "Spier-Verse Diffusion",
    # "Elden Ring Diffusion",
    # "Robo-Diffusion",
    # "Modern Disney Diffusion",

    # "stable_diffusion_inpainting", # Enable this to allow inpainting/outpainting. Careful of trying to enable this in tandem with other models if you have 8G or less VRAM!
]