import torch

from worker.logger import logger
# from nataili.util.voodoo import initialise_voodoo


class ModelManager:
    """
    Contains links to all the other MM classes
    """

    def __init__(
        self,
        aitemplate: bool = False,
        blip: bool = False,
        clip: bool = False,
        compvis: bool = False,
        diffusers: bool = False,
        esrgan: bool = False,
        gfpgan: bool = False,
        safety_checker: bool = False,
        codeformer: bool = False,
        controlnet: bool = False,
    ):
        # initialise_voodoo()
        if aitemplate:
            from nataili.model_manager.aitemplate import AITemplateModelManager

            self.aitemplate = AITemplateModelManager()
        else:
            self.aitemplate = None
        if blip:
            from nataili.model_manager.blip import BlipModelManager

            self.blip = BlipModelManager()
        else:
            self.blip = None
        if clip:
            from nataili.model_manager.clip import ClipModelManager

            self.clip = ClipModelManager()
        else:
            self.clip = None
        if compvis:
            from nataili.model_manager.compvis import CompVisModelManager

            self.compvis = CompVisModelManager()
        else:
            self.compvis = None
        if diffusers:
            from nataili.model_manager.diffusers import DiffusersModelManager

            self.diffusers = DiffusersModelManager()
        else:
            self.diffusers = None
        if esrgan:
            from nataili.model_manager.esrgan import EsrganModelManager

            self.esrgan = EsrganModelManager()
        else:
            self.esrgan = None
        if gfpgan:
            from nataili.model_manager.gfpgan import GfpganModelManager

            self.gfpgan = GfpganModelManager()
        else:
            self.gfpgan = None
        if safety_checker:
            from nataili.model_manager.safety_checker import SafetyCheckerModelManager

            self.safety_checker = SafetyCheckerModelManager()
        else:
            self.safety_checker = None
        if codeformer:
            from nataili.model_manager.codeformer import CodeFormerModelManager

            self.codeformer = CodeFormerModelManager()
        else:
            self.codeformer = None
        if controlnet:
            from nataili.model_manager.controlnet import ControlNetModelManager

            self.controlnet = ControlNetModelManager()
        else:
            self.controlnet = None
        self.cuda_available = torch.cuda.is_available()
        self.models = {}
        self.available_models = []
        self.loaded_models = {}
        self.init()

    def init(self):
        """
        Initialize SuperModelManager's models and available_models from
        the models and available_models of the model types.
        Individual model types are already initialized in their own init() functions
        which are called when the individual model manager is created in __init__.
        """
        model_types = [
            self.aitemplate,
            self.blip,
            self.clip,
            self.compvis,
            self.diffusers,
            self.esrgan,
            self.gfpgan,
            self.safety_checker,
            self.codeformer,
            self.controlnet,
        ]
        # reset available models
        self.available_models = []
        for model_type in model_types:
            if model_type is not None:
                self.models.update(model_type.models)
                self.available_models.extend(model_type.available_models)

    def reload_database(self):
        """
        Horde-specific function to reload the database of available models.
        Note: It is not appropriate to place `model_type.init()` in `init()`
        because individual model types are already initialized after being created
        i.e. if `model_type.init()` is placed in `self.init()`, the database will be
        loaded twice.
        """
        model_types = [
            self.aitemplate,
            self.blip,
            self.clip,
            self.compvis,
            self.diffusers,
            self.esrgan,
            self.gfpgan,
            self.safety_checker,
            self.codeformer,
            self.controlnet,
        ]
        self.available_models = []  # reset available models
        for model_type in model_types:
            if model_type is not None:
                model_type.init()
                self.models.update(model_type.models)
                self.available_models.extend(model_type.available_models)

    def download_model(self, model_name):
        if self.aitemplate is not None and model_name in self.aitemplate.models:
            return self.aitemplate.download_model(model_name)
        if self.blip is not None and model_name in self.blip.models:
            return self.blip.download_model(model_name)
        if self.clip is not None and model_name in self.clip.models:
            return self.clip.download_model(model_name)
        if self.codeformer is not None and model_name in self.codeformer.models:
            return self.codeformer.download_model(model_name)
        if self.compvis is not None and model_name in self.compvis.models:
            return self.compvis.download_model(model_name)
        if self.diffusers is not None and model_name in self.diffusers.models:
            return self.diffusers.download_model(model_name)
        if self.esrgan is not None and model_name in self.esrgan.models:
            return self.esrgan.download_model(model_name)
        if self.gfpgan is not None and model_name in self.gfpgan.models:
            return self.gfpgan.download_model(model_name)
        if self.safety_checker is not None and model_name in self.safety_checker.models:
            return self.safety_checker.download_model(model_name)
        if self.controlnet is not None and model_name in self.controlnet.models:
            return self.controlnet.download_model(model_name)

    def download_all(self):
        if self.aitemplate is not None:
            self.aitemplate.download_ait()
        if self.blip is not None:
            self.blip.download_all_models()
        if self.clip is not None:
            self.clip.download_all_models()
        if self.codeformer is not None:
            self.codeformer.download_all_models()
        if self.compvis is not None:
            self.compvis.download_all_models()
        if self.diffusers is not None:
            self.diffusers.download_all_models()
        if self.esrgan is not None:
            self.esrgan.download_all_models()
        if self.gfpgan is not None:
            self.gfpgan.download_all_models()
        if self.safety_checker is not None:
            self.safety_checker.download_all_models()
        if self.controlnet is not None:
            self.controlnet.download_all_models()

    def validate_model(self, model_name, skip_checksum=False):
        if self.blip is not None and model_name in self.blip.models:
            return self.blip.validate_model(model_name, skip_checksum)
        if self.clip is not None and model_name in self.clip.models:
            return self.clip.validate_model(model_name, skip_checksum)
        if self.codeformer is not None and model_name in self.codeformer.models:
            return self.codeformer.validate_model(model_name, skip_checksum)
        if self.compvis is not None and model_name in self.compvis.models:
            return self.compvis.validate_model(model_name, skip_checksum)
        if self.diffusers is not None and model_name in self.diffusers.models:
            return self.diffusers.validate_model(model_name, skip_checksum)
        if self.esrgan is not None and model_name in self.esrgan.models:
            return self.esrgan.validate_model(model_name, skip_checksum)
        if self.gfpgan is not None and model_name in self.gfpgan.models:
            return self.gfpgan.validate_model(model_name, skip_checksum)
        if self.safety_checker is not None and model_name in self.safety_checker.models:
            return self.safety_checker.validate_model(model_name, skip_checksum)
        if self.controlnet is not None and model_name in self.controlnet.models:
            return self.controlnet.validate_model(model_name, skip_checksum)

    def taint_models(self, models):
        if self.aitemplate is not None and any(model in self.aitemplate.models for model in models):
            self.aitemplate.taint_models(models)
        if self.blip is not None and any(model in self.blip.models for model in models):
            self.blip.taint_models(models)
        if self.clip is not None and any(model in self.clip.models for model in models):
            self.clip.taint_models(models)
        if self.codeformer is not None and any(model in self.codeformer.models for model in models):
            self.codeformer.taint_models(models)
        if self.compvis is not None and any(model in self.compvis.models for model in models):
            self.compvis.taint_models(models)
        if self.diffusers is not None and any(model in self.diffusers.models for model in models):
            self.diffusers.taint_models(models)
        if self.esrgan is not None and any(model in self.esrgan.models for model in models):
            self.esrgan.taint_models(models)
        if self.gfpgan is not None and any(model in self.gfpgan.models for model in models):
            self.gfpgan.taint_models(models)
        if self.safety_checker is not None and any(model in self.safety_checker.models for model in models):
            self.safety_checker.taint_models(models)
        if self.controlnet is not None and any(model in self.controlnet.models for model in models):
            self.controlnet.taint_models(models)

    def unload_model(self, model_name):
        if self.aitemplate is not None and model_name in self.aitemplate.models:
            self.aitemplate.unload_model(model_name)
        if self.blip is not None and model_name in self.blip.models:
            self.blip.unload_model(model_name)
            del self.loaded_models[model_name]
        if self.clip is not None and model_name in self.clip.models:
            self.clip.unload_model(model_name)
            del self.loaded_models[model_name]
        if self.codeformer is not None and model_name in self.codeformer.models:
            self.codeformer.unload_model(model_name)
            del self.loaded_models[model_name]
        if self.compvis is not None and model_name in self.compvis.models:
            self.compvis.unload_model(model_name)
            del self.loaded_models[model_name]
        if self.diffusers is not None and model_name in self.diffusers.models:
            self.diffusers.unload_model(model_name)
            del self.loaded_models[model_name]
        if self.esrgan is not None and model_name in self.esrgan.models:
            self.esrgan.unload_model(model_name)
            del self.loaded_models[model_name]
        if self.gfpgan is not None and model_name in self.gfpgan.models:
            self.gfpgan.unload_model(model_name)
            del self.loaded_models[model_name]
        if self.safety_checker is not None and model_name in self.safety_checker.models:
            self.safety_checker.unload_model(model_name)
            del self.loaded_models[model_name]
        if self.controlnet is not None and model_name in self.controlnet.models:
            self.controlnet.unload_model(model_name)
            del self.loaded_models[model_name]
        # Also remove from super() model list
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]

    def get_loaded_models_names(self, string=False):
        """
        :param string: If True, returns concatenated string of model names
        Returns a list of the loaded model names
        """
        if string:
            return ", ".join(self.loaded_models.keys())
        return list(self.loaded_models.keys())

    def get_available_models_by_types(self, model_types=None):
        if not model_types:
            model_types = ["ckpt", "diffusers"]
        models_available = []
        for model_type in model_types:
            if model_type == "ckpt":
                if self.compvis is not None:
                    for model in self.compvis.models:
                        # We don't want to check the .yaml file as those exist in this repo instead
                        model_files = [
                            filename
                            for filename in self.compvis.get_model_files(model)
                            if not filename["path"].endswith(".yaml")
                        ]
                        if self.compvis.check_available(model_files):
                            models_available.append(model)
            if model_type == "diffusers":
                if self.diffusers is not None:
                    for model in self.diffusers.models:
                        if self.diffusers.check_available(self.diffusers.get_model_files(model)):
                            models_available.append(model)
        return models_available

    def count_available_models_by_types(self, model_types=None):
        return len(self.get_available_models_by_types(model_types))

    def get_available_models(self):
        """
        Returns the available models
        """
        return self.available_models

    def load(
        self,
        model_name,
        half_precision=True,
        gpu_id=0,
        cpu_only=False,
        voodoo=False,
    ):
        """
        model_name: str. Name of the model to load. See available_models for a list of available models.
        half_precision: bool. If True, the model will be loaded in half precision.
        gpu_id: int. The id of the gpu to use. If the gpu is not available, the model will be loaded on the cpu.
        cpu_only: bool. If True, the model will be loaded on the cpu. If True, half_precision will be set to False.
        voodoo: bool. (compvis only) Voodoo ray.
        """
        if not self.cuda_available:
            cpu_only = True
        if self.aitemplate is not None and model_name in self.aitemplate.models:
            return self.aitemplate.load(model_name, gpu_id)
        if self.blip is not None and model_name in self.blip.models:
            success = self.blip.load(
                model_name=model_name, half_precision=half_precision, gpu_id=gpu_id, cpu_only=cpu_only
            )
            if success:
                self.loaded_models.update({model_name: self.blip.loaded_models[model_name]})
            return success
        if self.clip is not None and model_name in self.clip.models:
            success = self.clip.load(
                model_name=model_name, half_precision=half_precision, gpu_id=gpu_id, cpu_only=cpu_only
            )
            if success:
                self.loaded_models.update({model_name: self.clip.loaded_models[model_name]})
            return success
        if self.codeformer is not None and model_name in self.codeformer.models:
            success = self.codeformer.load(
                model_name=model_name, half_precision=half_precision, gpu_id=gpu_id, cpu_only=cpu_only
            )
            if success:
                self.loaded_models.update({model_name: self.codeformer.loaded_models[model_name]})
            return success
        if self.compvis is not None and model_name in self.compvis.models:
            success = self.compvis.load(
                model_name=model_name, half_precision=half_precision, gpu_id=gpu_id, cpu_only=cpu_only, voodoo=voodoo
            )
            if success:
                self.loaded_models.update({model_name: self.compvis.loaded_models[model_name]})
            return success
        if self.diffusers is not None and model_name in self.diffusers.models:
            success = self.diffusers.load(
                model_name=model_name, half_precision=half_precision, gpu_id=gpu_id, cpu_only=cpu_only, voodoo=voodoo
            )
            if success:
                self.loaded_models.update({model_name: self.diffusers.loaded_models[model_name]})
            return success
        if self.esrgan is not None and model_name in self.esrgan.models:
            success = self.esrgan.load(
                model_name=model_name, half_precision=half_precision, gpu_id=gpu_id, cpu_only=cpu_only
            )
            if success:
                self.loaded_models.update({model_name: self.esrgan.loaded_models[model_name]})
            return success
        if self.gfpgan is not None and model_name in self.gfpgan.models:
            success = self.gfpgan.load(model_name=model_name, gpu_id=gpu_id, cpu_only=cpu_only)
            if success:
                self.loaded_models.update({model_name: self.gfpgan.loaded_models[model_name]})
            return success
        if self.safety_checker is not None and model_name in self.safety_checker.models:
            success = self.safety_checker.load(
                model_name=model_name, half_precision=half_precision, gpu_id=gpu_id, cpu_only=True  # for the horde
            )
            if success:
                self.loaded_models.update({model_name: self.safety_checker.loaded_models[model_name]})
            return success
        logger.error(f"{model_name} not found")
        return False
