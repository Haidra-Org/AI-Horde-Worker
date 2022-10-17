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
# The models to use. You can select a different main model, or select more than one if you have enough VRAM
# The last model in this list takes priority when the client accepts more than 1
models_to_load = [
    "stable_diffusion"
    # "waifu_diffusion"
]