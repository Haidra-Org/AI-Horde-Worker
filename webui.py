import gradio as gr
import regex as re
import requests
from nataili.util.logger import logger

from worker.bridge_data.interrogation import InterrogationBridgeData
from worker.bridge_data.stable_diffusion import StableDiffusionBridgeData


def Process_Input_List(list):
    output = []
    if list != "":
        temp = list.split(",")
        for item in temp:
            trimmed_item = item.strip()
            output.append(trimmed_item)
    return output


def Update_Bridge(
    horde_url="https://stablehorde.net",
    worker_name=None,
    api_key="000000000000",
    priority_usernames=None,
    max_power=8,
    queue_size=0,
    max_threads=1,
    nsfw=True,
    censor_nsfw=False,
    blacklist="",
    censorlist="",
    allow_img2img=True,
    allow_painting=True,
    allow_unsafe_ip=True,
    allow_post_processing=True,
    allow_controlnet=True,
    require_upfront_kudos=False,
    dynamic_models=True,
    number_of_dynamic_models=3,
    max_models_to_download=10,
    models_to_load=None,
    models_to_skip=None,
    forms=None,
):
    priority_usernames_list = Process_Input_List(priority_usernames)
    blacklist_list = Process_Input_List(blacklist)
    censorlist_list = Process_Input_List(censorlist)

    data = f"""horde_url = "{horde_url}"
worker_name = "{worker_name}"
api_key = "{api_key}"
priority_usernames = {priority_usernames_list}
max_power = {max_power}
queue_size = {queue_size}
max_threads = {max_threads}
nsfw = {nsfw}
censor_nsfw = {censor_nsfw}
blacklist = {blacklist_list}
censorlist = {censorlist_list}
allow_img2img = {allow_img2img}
allow_painting = {allow_painting}
allow_unsafe_ip = {allow_unsafe_ip}
allow_post_processing = {allow_post_processing}
allow_controlnet = {allow_controlnet}
require_upfront_kudos = {require_upfront_kudos}
dynamic_models = {dynamic_models}
number_of_dynamic_models = {number_of_dynamic_models}
max_models_to_download = {max_models_to_download}
models_to_load = {models_to_load}
models_to_skip = {models_to_skip}
forms = {forms}"""
    toExec = """text_file = open("bridgeData.py", "w+"); text_file.write(data); text_file.close()"""
    try:
        exec(toExec)
        output = "Bridge Data Updated Successfully\n"
    except Exception as e:
        output = f"Failed to update: {e}"
    data = re.sub(r"api_key.*", 'api_key = "**********"', data)
    output += "\n" + data
    return output


def download_models(model_location):
    models = None
    try:
        r = requests.get(model_location)
        models = r.json()
        print("Models downloaded successfully")
    except Exception:
        print("Failed to load models")
    return models


def load_models():
    remote_models = "https://raw.githubusercontent.com/Sygil-Dev/nataili-model-reference/main/stable_diffusion.json"
    latest_models = download_models(remote_models)

    available_models = []
    for model in latest_models:
        if model not in [
            "RealESRGAN_x4plus",
            "RealESRGAN_x4plus_anime_6B",
            "GFPGAN",
            "CodeFormers",
            "LDSR",
            "BLIP",
            "BLIP_Large",
            "ViT-L/14",
            "ViT-g-14",
            "ViT-H-14",
            "diffusers_stable_diffusion",
            "safety_checker",
        ]:
            available_models.append(model)
    model_list = sorted(available_models, key=str.casefold)
    return model_list


def load_workerID(worker_name):
    workerID = ""
    workers_URL = "https://stablehorde.net/api/v2/workers"
    r = requests.get(workers_URL)
    worker_json = r.json()
    for item in worker_json:
        if item["name"] == worker_name:
            workerID = item["id"]
    return workerID


def load_worker_mode(worker_name):
    worker_mode = False
    workers_URL = "https://stablehorde.net/api/v2/workers"
    r = requests.get(workers_URL)
    worker_json = r.json()
    for item in worker_json:
        if item["name"] == worker_name:
            worker_mode = item["maintenance_mode"]
    return worker_mode


def load_worker_stats(worker_name):
    worker_stats = ""
    workers_URL = "https://stablehorde.net/api/v2/workers"
    r = requests.get(workers_URL)
    worker_json = r.json()
    for item in worker_json:
        if item["name"] == worker_name:
            worker_stats += "Current MPS:  " + str(item["performance"]).split()[0] + " MPS\n"
            worker_stats += "Total Kudos Earned:  " + str(item["kudos_rewards"]) + "\n"
            worker_stats += "Total Jobs Completed:  " + str(item["requests_fulfilled"])
    return worker_stats


def update_worker_mode(worker_name, worker_id, current_mode, apikey):
    header = {"apikey": apikey}
    payload = {"maintenance": False, "name": worker_name}
    if current_mode == "False":
        payload = {"maintenance": True, "name": worker_name}
    worker_URL = f"https://stablehorde.net/api/v2/workers/{worker_id}"
    r = requests.put(worker_URL, json=payload, headers=header)
    return r.json()


def Start_WebUI(stable_diffusion_bridge_data, interrogation_bridge_data):
    stable_diffusion_bridge_data.reload_data()
    interrogation_bridge_data.reload_data()

    model_list = load_models()
    current_model_list = stable_diffusion_bridge_data.model_names
    if stable_diffusion_bridge_data.dynamic_models:
        current_model_list = stable_diffusion_bridge_data.predefined_models
    current_model_list = list(set(current_model_list))
    existing_priority_usernames = ""
    for item in stable_diffusion_bridge_data.priority_usernames:
        existing_priority_usernames += item
        existing_priority_usernames += ","
    if len(existing_priority_usernames) > 0:
        if existing_priority_usernames[-1] == ",":
            existing_priority_usernames = existing_priority_usernames[:-1]

    existing_blacklist = ""
    for item in stable_diffusion_bridge_data.blacklist:
        existing_blacklist += item
        existing_blacklist += ","
    if len(existing_blacklist) > 0:
        if existing_blacklist[-1] == ",":
            existing_blacklist = existing_blacklist[:-1]

    existing_censorlist = ""
    for item in stable_diffusion_bridge_data.censorlist:
        existing_censorlist += item
        existing_censorlist += ","
    if len(existing_censorlist) > 0:
        if existing_censorlist[-1] == ",":
            existing_censorlist = existing_censorlist[:-1]

    with gr.Blocks() as WebUI:
        horde_url = gr.Textbox("https://stablehorde.net", visible=False)
        gr.Markdown("# Welcome to the Stable Horde Bridge Configurator")
        with gr.Column():
            with gr.Row():
                worker_ID = gr.TextArea(label="Worker ID", lines=1, interactive=False)
                maintenance_mode = gr.TextArea(label="Current Maintenance Mode Status", lines=1, interactive=False)
                worker_stats = gr.TextArea(label="Worker Statistics", lines=3, interactive=False)
            with gr.Row():
                worker_name = gr.Textbox(label="Worker Name", value=stable_diffusion_bridge_data.worker_name)
                api_key = gr.Textbox(label="API Key", value=stable_diffusion_bridge_data.api_key, type="password")
                priority_usernames = gr.Textbox(label="Priority Usernames", value=existing_priority_usernames)
            with gr.Row():
                max_threads = gr.Slider(
                    1, 20, step=1, label="Number of Threads", value=stable_diffusion_bridge_data.max_threads
                )
                queue_size = gr.Slider(0, 2, step=1, label="Queue Size", value=stable_diffusion_bridge_data.queue_size)
                allow_unsafe_ip = gr.Checkbox(
                    label="Allow Requests From Suspicious IP Addresses",
                    value=stable_diffusion_bridge_data.allow_unsafe_ip,
                )
                require_upfront_kudos = gr.Checkbox(
                    label="Require Users To Have Kudos Before Processing",
                    value=stable_diffusion_bridge_data.require_upfront_kudos,
                )
        with gr.Tab("Image Generation"):
            with gr.Tab("Image Generation"):
                with gr.Row():
                    max_power = gr.Slider(
                        2, 288, step=2, label="Max Power", value=stable_diffusion_bridge_data.max_power
                    )
                    allow_img2img = gr.Checkbox(
                        label="Allow img2img Requests", value=stable_diffusion_bridge_data.allow_img2img
                    )
                    allow_painting = gr.Checkbox(
                        label="Allow Inpainting Requests", value=stable_diffusion_bridge_data.allow_painting
                    )
                    allow_post_processing = gr.Checkbox(
                        label="Allow Requests Requiring Post-Processing",
                        value=stable_diffusion_bridge_data.allow_post_processing,
                    )
                    allow_controlnet = gr.Checkbox(
                        label="Allow Requests Requiring ControlNet",
                        value=stable_diffusion_bridge_data.allow_controlnet,
                    )
            with gr.Tab("NSFW"):
                with gr.Row():
                    nsfw = gr.Checkbox(label="Enable NSFW", value=stable_diffusion_bridge_data.nsfw)
                    censor_nsfw = gr.Checkbox(
                        label="Censor NSFW Images", value=stable_diffusion_bridge_data.censor_nsfw
                    )
                blacklist = gr.Textbox(
                    label="Blacklisted Words or Phrases - Seperate with commas", value=existing_blacklist
                )
                censorlist = gr.Textbox(
                    label="Censored Words or Phrases - Seperate with commas", value=existing_censorlist
                )
            with gr.Tab("Model Manager"):
                with gr.Row():
                    dynamic_models = gr.Checkbox(
                        label="Enable Dynamic Models", value=stable_diffusion_bridge_data.dynamic_models
                    )
                    number_of_dynamic_models = gr.Number(
                        label="Number of Models To Be Dynamically Loading",
                        value=stable_diffusion_bridge_data.number_of_dynamic_models,
                        precision=0,
                    )
                    max_models_to_download = gr.Number(
                        label="Maximum Number of Models To Download",
                        value=stable_diffusion_bridge_data.max_models_to_download,
                        precision=0,
                    )
                models_to_load = gr.CheckboxGroup(
                    choices=model_list,
                    label="Models To Load (Not affected by dynamic models)",
                    value=current_model_list,
                )
                models_to_skip = gr.CheckboxGroup(
                    choices=model_list, label="Models To Skip", value=stable_diffusion_bridge_data.models_to_skip
                )
        with gr.Tab("Image Interrogation"):
            forms = gr.CheckboxGroup(
                label="Interrogation Modes",
                choices=["caption", "nsfw", "interrogation"],
                value=interrogation_bridge_data.forms,
            )
        with gr.Row():
            system = gr.TextArea(label="System Messages", lines=1, interactive=False)
        with gr.Row():
            gr.Button(value="Toggle Maintenance Mode", variant="primary").click(
                update_worker_mode, inputs=[worker_name, worker_ID, maintenance_mode, api_key], outputs=system
            )
            gr.Button(value="Update Bridge", variant="Primary").click(
                Update_Bridge,
                inputs=[
                    horde_url,
                    worker_name,
                    api_key,
                    priority_usernames,
                    max_power,
                    queue_size,
                    max_threads,
                    nsfw,
                    censor_nsfw,
                    blacklist,
                    censorlist,
                    allow_img2img,
                    allow_painting,
                    allow_unsafe_ip,
                    allow_post_processing,
                    allow_controlnet,
                    require_upfront_kudos,
                    dynamic_models,
                    number_of_dynamic_models,
                    max_models_to_download,
                    models_to_load,
                    models_to_skip,
                    forms,
                ],
                outputs=system,
            )
        gr.Markdown(
            """
## Definitions

### Worker Name
This is a the name of your worker.  It needs to be unique to the whole horde
NOTE: You cannot run a interrogation and a stable diffusion worker with the same name

### API Key
This is your Stable Horde API Key

### Priority Usernames
These users (in format username#id e.g. residentchiefnz#3966) will be prioritized over all other workers.
Note: You do not need to add your own name to this list

### Max Power
This number derives the maximum image size your worker can generate.
Common numbers are 2 (256x256), 8 (512x512), 18 (768x768), and 32 (1024x1024)

### Queue Size
This number determines the number of extra jobs that are collected.
When the worker requests jobs it will request 1 job per thread plus this number

### Max Threads
This determines how many jobs will be processed simultaneously.
Each job requires extra VRAM and will slow the speed of generations.
This should be set to provide generations at a minumum of 0.6 megapixels per second
Expected limit per VRAM size: 6Gb = 1 thread, 6-8Gb = 2 threads, 8-12Gb = 3 threads, 12Gb - 24Gb = 4 threads

### Censor NSFW
If this is true and Enable NSFW is false, the worker will accept NSFW requests, but send back a censored image

### Blacklist
Any words in here that match a prompt will result in that job not being picked up by this worker

### Censorlist
Any words in here that match a prompt will always result in a censored image being returned

### Enable Dynamic Models
In addition to the models in "Models to Load" being loaded, the worker will also load models that in high demand

### Max Models To Download
This number is the maximum number of models that the worker will download and run.
Having a high number here will fill up your hard drive.  NOTE: This includes the safety checker and the post-processors

### Models To Skip
These models will never be downloaded to the worker, even if Enable Dynamic Models is selected

### Models To Load
This determines which models to always have available from this worker. Each model takes between 2 and 8Gb of VRAM
"""
        )
        WebUI.queue()
        WebUI.load(load_workerID, inputs=worker_name, outputs=worker_ID, every=15)
        WebUI.load(load_worker_mode, inputs=worker_name, outputs=maintenance_mode, every=15)
        WebUI.load(load_worker_stats, inputs=worker_name, outputs=worker_stats, every=15)
    try:
        WebUI.launch(share=True)
    except KeyboardInterrupt:
        print("CTRL+C Pressed --> Shutdown Server")


if __name__ == "__main__":
    logger.init("Bridge WebUI", status="Initializing")
    stable_diffusion_bridge_data = StableDiffusionBridgeData()
    interrogation_bridge_data = InterrogationBridgeData()
    Start_WebUI(stable_diffusion_bridge_data, interrogation_bridge_data)
