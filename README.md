This repository provides python libraries for running Stable Diffusion

Sample use: 

```python
from nataili.model_manager import ModelManager
from nataili.inference.compvis.txt2img import txt2img
from nataili.util.cache import torch_gc

# The model manager loads and unloads the SD models and has features to download them or find their location
model_manager = ModelManager()
model_manager.init()
# The model to use for the generation. 
model = "stable_diffusion"
success = model_manager.load_model(model)
if success:
    print(f'{model} loaded')
else:
    print(f'{model} load error')
# Other classes exist like img2img
generator = txt2img(model_manager.loaded_models[model]["model"], model_manager.loaded_models[model]["device"], 'output_dir')
generator.generate('a donkey with a hat')
torch_gc()
# The image key in the generator contains a PIL image of the generation
image = generator.images[0]["image"]
image.save('a_donkey_with_a_hat.png', format="Png")
```

You can find more complete scripts in `test.py`

# Stable Horde Bridge

This repo contains the latest implementation for the [Stable Horde](https://stablehorde.net) Bridge. This will turn your graphics card(s) into a worker for the Stable Horde and you will receive in turn kudos which will give you priority for your own generations.

To run the bridge, simply run:

* Windows: `horde-bridge.cmd`
* Linux: `horde-bridge.sh`

This will take care of the setup for this environment and then automatically start the bridge.

For more information, see: https://github.com/db0/AI-Horde/blob/main/README_StableHorde.md#joining-the-horde

## Experimental

The bridge has experimental inpainting support via Diffusers library. This **can** work in parallel with the compvis/ckpt implementation, but it is not suggested unless you have plenty of VRAM to spare.

To set your worker to serve inpainting, add "stable_diffusion_inpainting" into your bridgeData's models_to_load list. The rest will be handled automatically. Workers without this model will not receive inpainting requests. Workers with **just** this model will **not** receive any requests **other** than inpainting! This will be fixed as soon as Diffusers are supported fully.
