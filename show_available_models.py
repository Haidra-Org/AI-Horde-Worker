"""Utility script to show available models."""

# test_download_models
from worker.model_manager import ModelManager

if __name__ == "__main__":
    # TODO: huggingface_hub or some way to use token instead of username/password
    mm = ModelManager(compvis=True)

    filtered_models = mm.compvis.get_filtered_models(type="ckpt")
    ppmodels = ""
    for model_name in filtered_models:
        if model_name == "LDSR":
            continue
        ppmodels += model_name
        if filtered_models[model_name].get("description"):
            ppmodels += f" : {filtered_models[model_name].get('description')}"
        ppmodels += "\n"
    print(f"## Known ckpt Models ##\n{ppmodels}")
    input("Press ENTER to continue")
