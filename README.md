This repository allows you to set up a Stable Horde Worker to generate or interrogate images for others

# Stable Horde Bridge

This repo contains the latest implementation for the [Stable Horde](https://stablehorde.net) Bridge. This will turn your graphics card(s) into a worker for the Stable Horde and you will receive in turn kudos which will give you priority for your own generations.

To run the bridge, simply run:

* Windows: `horde-bridge.cmd`
* Linux: `horde-bridge.sh`

This will take care of the setup for this environment and then automatically start the bridge.

For more information, see: https://github.com/db0/AI-Horde/blob/main/README_StableHorde.md#joining-the-horde

## Experimental

**run `update-runtime.cmd` or `update-runtime.sh` as dependencies have been updated**

To set your worker to serve inpainting, add "stable_diffusion_inpainting" into your bridgeData's models_to_load list. The rest will be handled automatically. Workers without this model will not receive inpainting requests. Workers with **just** this model will **not** receive any requests **other** than inpainting! This will be fixed as soon as Diffusers are supported fully.

## Model Usage
Many models in this project use the CreativeML OpenRAIL License.  [Please read the full license here.](https://huggingface.co/spaces/CompVis/stable-diffusion-license)