import os

from nataili.util.load_learned_embed_in_clip import load_learned_embed_in_clip

def process_prompt_tokens(prompt_tokens, model, concepts_dir):
    # compviz codebase
    # tokenizer =  model.cond_stage_model.tokenizer
    # text_encoder = model.cond_stage_model.transformer
    # diffusers codebase
    # tokenizer = pipe.tokenizer
    # text_encoder = pipe.text_encoder

    ext = (".pt", ".bin")
    for token_name in prompt_tokens:
        embedding_path = os.path.join(concepts_dir, token_name)
        if os.path.exists(embedding_path):
            for files in os.listdir(embedding_path):
                if files.endswith(ext):
                    load_learned_embed_in_clip(
                        f"{os.path.join(embedding_path, files)}",
                        model.cond_stage_model.transformer,
                        model.cond_stage_model.tokenizer,
                        f"<{token_name}>",
                    )
        else:
            print(f"Concept {token_name} not found in {concepts_dir}")
            return