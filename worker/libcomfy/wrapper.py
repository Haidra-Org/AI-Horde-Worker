# comfy.py
# Wrapper around comfy to allow usage by the horde worker.
import copy
import glob
import json
import os
import re
from io import BytesIO
from PIL import Image

from loguru import logger

from ComfyUI import execution


class ComfyWrapper:
    def __init__(self):
        self.pipelines = {}

        # XXX Temporary hack to force model directory location in this test
        os.environ["HORDE_MODEL_DIR_CHECKPOINTS"] = self._this_dir("../../nataili/compvis/")

        # Load our pipelines
        self._load_pipelines()

    def _this_dir(self, filename):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

    def _load_node(self, filename):
        try:
            execution.nodes.load_custom_node(self._this_dir(filename))
        except Exception:
            logger.error(f"Failed to load custom pipeline node: {filename}")
            return
        logger.debug(f"Loaded custom pipeline node: {filename}")

    def _load_custom_nodes(self):
        files = glob.glob(self._this_dir("node_*.py"))
        for file in files:
            self._load_node(os.path.basename(file))

    def _load_pipelines(self):
        files = glob.glob(self._this_dir("pipeline_*.json"))
        for file in files:
            try:
                with open(file) as jsonfile:
                    pipeline_name = re.match(r".*pipeline_(.*)\.json", file)[1]
                    data = json.loads(jsonfile.read())
                    self.pipelines[pipeline_name] = data
                    logger.debug(f"Loaded inference pipeline: {pipeline_name}")
            except (OSError, ValueError):
                logger.error(f"Invalid inference pipeline file: {file}")

    # Inject parameters into a pre-configured pipeline
    # We allow "inputs" to be missing from the key name, if it is we insert it.
    def _set(self, dct, **kwargs):
        for key, value in kwargs.items():
            keys = key.split(".")
            if "inputs" not in keys:
                keys.insert(1, "inputs")
            current = dct

            for k in keys[:-1]:
                if k not in current:
                    logger.error(f"Attempt to set unknown pipeline parameter {key}")
                    break
                else:
                    current = current[k]

            current[keys[-1]] = value

    # Execute the named pipeline and pass the pipeline the parameter provided.
    # For the horde we assume the pipeline returns an array of images.
    def run_pipeline(self, pipeline_name, params):

        # Sanity
        if pipeline_name not in self.pipelines:
            logger.error(f"Unknown inference pipeline: {pipeline_name}")
            return

        # Grab a copy of the pipeline
        pipeline = copy.copy(self.pipelines[pipeline_name])
        # Set the pipeline parameters
        self._set(pipeline, **params)
        # Run it!
        exec = execution.PromptExecutor(self)
        # Load our custom nodes
        self._load_custom_nodes()
        exec.execute(pipeline)

        # From the horde point of view, let us assume the output we are interested in
        # is always in a HordeImageOutput node named "output_image". This is an array of
        # dicts of the form:
        # [ {
        #     "imagedata": <BytesIO>,
        #     "type": "PNG"
        #   },
        # ]
        # See node_image_output.py
        return exec.outputs["output_image"]["images"]

    # A simple test function for early development testing
    # Runs the "stable_diffusion" pipeline and create an image named "HORDE-IMAGE.png"
    def test(self):
        params = {
            "sampler.seed": 12345,
            "sampler.cfg": 7.5,
            "sampler.scheduler": "karras",
            "sampler.sampler_name": "dpmpp_2m",
            "sampler.steps": 25,
            "prompt.text": "a closeup photo of a confused dog",
            "negative_prompt.text": "cat, black and white, deformed",
            "model_loader.ckpt_name": "model_1_5.ckpt",
            "empty_latent_image.width": 768,
            "empty_latent_image.height": 768,
        }
        images = self.run_pipeline("stable_diffusion", params)

        # XXX for proof of concept development we just save the image to a file        
        image = Image.open(images[0]["imagedata"])
        image.save("HORDE-IMAGE.png")
        

if __name__ == "__main__":
    job = ComfyWrapper()
    job.test()
