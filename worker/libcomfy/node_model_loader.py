# node_model_loader.py
# Simple proof of concept custom node to load models.
# We shall expand this to actually integrate with the horde model manager.
import os

import comfy


class HordeCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": ("<checkpoint file>",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = os.getenv("HORDE_MODEL_DIR_CHECKPOINTS", "./")
        embeddings_path = os.getenv("HORDE_MODEL_DIR_EMBEDDINGS", "./")
        ckpt_path = os.path.join(ckpt_path, ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=True, embedding_directory=embeddings_path
        )
        return out


NODE_CLASS_MAPPINGS = {"HordeCheckpointLoader": HordeCheckpointLoader}
