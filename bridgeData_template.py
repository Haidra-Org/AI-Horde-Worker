# The horde url
horde_url = "https://stablehorde.net"
# Give a cool name to your instance
worker_name = "My Awesome Instance"
# The api_key identifies a unique user in the horde
# Visit https://stablehorde.net/register to create one before you can join
api_key = "0000000000"
# Put other users whose prompts you want to prioritize.
# The owner's username is always included so you don't need to add it here,
priority_usernames = []
# The amount of power your system can handle
# 8 means 512*512. Each increase increases the possible resoluion by 64 pixes
# So if you put this to 2 (the minimum, your SD can only generate 64x64 pixels
# If you put this to 32, it is equivalent to 1024x1024 pixels
max_power = 8
# The amount of parallel jobs to pick up for the horde. Each running job will consume the amount of RAM needed to run each model, and will also affect the speed of other running jobs
# so make sure you have enough VRAM to load models in parallel, and that the speed of fulfilling requests is not too slow
# Expected limit per VRAM size: <6 VMAM: 1, <=8 VRAM: 2, <=12 VRAM:3, <=14 VRAM: 4
# But remember that the speed of your gens will also be affected for each parallel job
max_threads = 1
# We will keep this many requests in the queue so we can start working as soon as a thread is available
# Recommended to keep no higher than 1
queue_size = 0
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
# If you set this to True, the worker will detect the most popular models and load them automatically ( Defaults to True if missing )
# Note this ultimately overrides the models_to_load list
dynamic_models = True

# Adjust how many models to load into memory. In future this will likely be an argument for memory size or may disappear, but for right now, I'm lazy
number_of_dynamic_models = 3
# The maximum amount of models to download dynamically for this worker. Increase this amount of you have plenty of space. Keep it low if you do not
# When the amount of models downloaded reaches this amount, the dynamic list will only use dynamic models already downloaded
# Therefore make sure you put some generalist and popular models in your models_to_load list if this number is small!
max_models_to_download = 10
# The models to use. You can select a different main model, or select more than one.
# With you can easily load 20 of these models with 32Gb RAM and 6G VRAM.
# Adjust how many models you load based on how much RAM (not VRAM) you have available.
# The last model in this list takes priority when the client accepts more than 1
# if you do not know which models you can add here, use the below command
# python show_available_models.py
## WARNING: In case you have dynamic models this list is instead specifying models to always load along with the dynamic models!
# So your total list will be your specific models + the dynamic models
# in that case, keep this list short, preferrably to only a few more obscure models you'd like to always see available
models_to_load = [
    "stable_diffusion_2.1", # This is the standard compvis model. It is not using Diffusers (yet)
    "stable_diffusion",  # This is the standard compvis model. It is not using Diffusers (yet)
    ## Specialized Style models:
    # "Anything Diffusion",
    # "Yiffy",
    # "waifu_diffusion",
    ## Dreambooth Models:
    # "Arcane Diffusion",
    # "Spider-Verse Diffusion",
    # "Elden Ring Diffusion",
    # "Robo-Diffusion",
    # "mo-di-diffusion",
    # "Knollingcase",
    ## Enable this to allow inpainting/outpainting.
    # "stable_diffusion_inpainting",
]
# This is used when dynamic_models == True
# The models in this list will not be loaded when they exist in the top models
# This is to avoid loading models which you do not want either due to VRAM constraints, or due to NSFW content
models_to_skip = [
    "stable_diffusion_inpainting",  # Inpainting is generally quite heavy along with other models for smaller GPUs.
    # "stable_diffusion_2.1",  # Stable diffusion 2.1 has bigger memory requirements than 1.5, so if your card cannot lift, it, disable it
    # "stable_diffusion_2.0",  # Same as Stable diffusion 2.1
    ## Popular NSFW models:
    # "Zeipher Female Model",
    # "Hentai Diffusion",
]
