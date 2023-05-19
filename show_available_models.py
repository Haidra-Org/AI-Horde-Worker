"""Utility script to show available models."""


# isort: off
import hordelib

hordelib.initialise()

from hordelib.horde import SharedModelManager  # noqa: E402

# isort: on

if __name__ == "__main__":
    # TODO: huggingface_hub or some way to use token instead of username/password
    SharedModelManager.loadModelManagers(
        compvis=True,
    )
    mm = SharedModelManager.manager

    filtered_models = mm.compvis.get_filtered_models(type="compvis")
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
