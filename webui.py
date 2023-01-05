import gradio as gr
import requests

from worker.bridge_data.stable_diffusion import StableDiffusionBridgeData
from worker.bridge_data.interrogation import InterrogationBridgeData
from nataili.util import logger


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
        output = "Updated Successfully"
    except Exception as e:
        output = "Failed to update: " + e
    output += data
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
    remote_models = "https://raw.githubusercontent.com/Sygil-Dev/nataili-model-reference/main/db.json"
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
        gr.Markdown("## Welcome to the Stable Horde Bridge Configurator")
        with gr.Column():
            worker_name = gr.Textbox(label="Worker Name", value=stable_diffusion_bridge_data.worker_name)
            api_key = gr.Textbox(label="API Key", value=stable_diffusion_bridge_data.api_key)
            priority_usernames = gr.Textbox(label="Priority Usernames", value=existing_priority_usernames)
            with gr.Row():
                max_threads = gr.Slider(1, 4, step=1, label="Number of Threads", value=stable_diffusion_bridge_data.max_threads)
                queue_size = gr.Slider(0, 2, step=1, label="Queue Size", value=stable_diffusion_bridge_data.queue_size)
                allow_unsafe_ip = gr.Checkbox(
                    label="Allow Requests From Suspicious IP Addresses", value=stable_diffusion_bridge_data.allow_unsafe_ip
                )
                require_upfront_kudos = gr.Checkbox(
                    label="Require Users To Have Kudos Before Processing", value=stable_diffusion_bridge_data.require_upfront_kudos
                )
        with gr.Tab("Image Generation"):
            with gr.Tab("Image Generation"):
                with gr.Row():
                    max_power = gr.Slider(2, 288, step=2, label="Max Power", value=stable_diffusion_bridge_data.max_power)
                    allow_img2img = gr.Checkbox(label="Allow img2img Requests", value=stable_diffusion_bridge_data.allow_img2img)
                    allow_painting = gr.Checkbox(label="Allow Inpainting Requests", value=stable_diffusion_bridge_data.allow_painting)
                    allow_post_processing = gr.Checkbox(
                        label="Allow Requests Requiring Post-Processing", value=stable_diffusion_bridge_data.allow_post_processing
                    )
            with gr.Tab("NSFW"):
                with gr.Row():
                    nsfw = gr.Checkbox(label="Enable NSFW", value=stable_diffusion_bridge_data.nsfw)
                    censor_nsfw = gr.Checkbox(label="Censor NSFW Images", value=stable_diffusion_bridge_data.censor_nsfw)
                blacklist = gr.Textbox(
                    label="Blacklisted Words or Phrases - Seperate with commas", value=existing_blacklist
                )
                censorlist = gr.Textbox(
                    label="Censored Words or Phrases - Seperate with commas", value=existing_censorlist
                )
            with gr.Tab("Model Manager"):
                with gr.Row():
                    dynamic_models = gr.Checkbox(label="Enable Dynamic Models", value=stable_diffusion_bridge_data.dynamic_models)
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
            forms = gr.CheckboxGroup(label="Interrogation Modes", choices=["caption", "nsfw", "interrogation"], value=interrogation_bridge_data.forms)
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
                require_upfront_kudos,
                dynamic_models,
                number_of_dynamic_models,
                max_models_to_download,
                models_to_load,
                models_to_skip,
                forms,
            ],
            outputs=gr.TextArea(label="System Messages"),
        )
    WebUI.launch(share=True)


if __name__ == "__main__":
    stable_diffusion_bridge_data = StableDiffusionBridgeData()
    interrogation_bridge_data = InterrogationBridgeData()
    Start_WebUI(stable_diffusion_bridge_data, interrogation_bridge_data)
