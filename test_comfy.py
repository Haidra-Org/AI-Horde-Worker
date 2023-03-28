import json
import sys
sys.path.append("ComfyUI")
from ComfyUI import execution

prompt_text = """
{
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 8,
            "denoise": 1,
            "latent_image": [
                "5",
                0
            ],
            "model": [
                "4",
                0
            ],
            "negative": [
                "7",
                0
            ],
            "positive": [
                "6",
                0
            ],
            "sampler_name": "euler",
            "scheduler": "normal",
            "seed": 8566257,
            "steps": 20
        }
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "model_1_5.ckpt"
        }
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "batch_size": 1,
            "height": 512,
            "width": 512
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [
                "4",
                1
            ],
            "text": "masterpiece best quality girl"
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [
                "4",
                1
            ],
            "text": "bad hands"
        }
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": [
                "3",
                0
            ],
            "vae": [
                "4",
                2
            ]
        }
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": [
                "8",
                0
            ]
        }
    }
}
"""

# Stub for suppressing server
class Callback():
    def __init__(self):
        pass
server = Callback()

def generate(text_prompt, neg_prompt, seed, model_name, sampler_name, scheduler, cfg, steps):

    prompt = json.loads(prompt_text)
    prompt["6"]["inputs"]["text"] = text_prompt
    prompt["7"]["inputs"]["text"] = neg_prompt
    prompt["3"]["inputs"]["seed"] = seed
    prompt["3"]["inputs"]["cfg"] = cfg
    prompt["3"]["inputs"]["seed"] = steps
    prompt["3"]["inputs"]["scheduler"] = scheduler
    prompt["3"]["inputs"]["sampler_name"] = sampler_name
    # Model - needs to be in ./models/checkpoints/
    prompt["4"]["inputs"]["ckpt_name"] = model_name 

    # Is the pipeline valid?
    valid = execution.validate_prompt(prompt)
    if not valid[0]:
        raise Exception("Pipeline is not valid")

    # Execute the pipeline
    exec = execution.PromptExecutor(server)
    exec.execute(prompt)


# Possible samplers are: "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", 
# "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"

# Possible schedulers are:
# "karras", "normal", "simple", "ddim_uniform"

generate(
    text_prompt="Closeup photo of a confused dog",
    neg_prompt="cat",
    seed=123456,
    model_name="Deliberate.ckpt",
    sampler_name="dpmpp_2m",
    scheduler="karras",
    cfg=7.5,
    steps=25
)
