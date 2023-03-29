# comfy.py
# Wrapper around comfy to allow usage by the horde worker.
import copy
import glob
import json
import os
import re
import sys

from loguru import logger

sys.path.append("ComfyUI")
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
        self._load_node("node_model_loader.py")

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

    def _set(self, dct, **kwargs):
        for key, value in kwargs.items():
            keys = key.split(".")
            current = dct

            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            current[keys[-1]] = value

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

    # A simple test function for early development testing
    def test(self):
        params = {
            "sampler.inputs.seed": 12345,
            "sampler.inputs.cfg": 7.5,
            "sampler.inputs.scheduler": "karras",
            "sampler.inputs.sampler_name": "dpmpp_2m",
            "sampler.inputs.steps": 25,
            "prompt.inputs.text": "a closeup photo of a confused dog",
            "negative_prompt.inputs.text": "cat",
            "model_loader.inputs.ckpt_name": "model_1_5.ckpt",
        }
        self.run_pipeline("stable_diffusion", params)


if __name__ == "__main__":
    job = ComfyWrapper()
    job.test()
