# node_image_output.py
# Simple proof of concept to return an image byte stream to the horde worker.
import json
from io import BytesIO

import comfy
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo


class HordeImageOutput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "get_image"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def get_image(self, images, prompt=None, extra_pnginfo=None):

        results = []
        for image in images:
            # Create a PNG
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            # Save the full pipeline and variables into the PNG metadata
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            byte_stream = BytesIO()
            img.save(byte_stream, format="PNG", pnginfo=metadata, compress_level=4)
            byte_stream.seek(0)

            results.append({"imagedata": byte_stream, "type": "PNG"})

        return {"images": results}


NODE_CLASS_MAPPINGS = {"HordeImageOutput": HordeImageOutput}
