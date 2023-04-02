import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from uuid import uuid4

import git
import requests
import torch
from tqdm import tqdm
from transformers import logging

# from nataili import disable_download_progress
from worker.cache import get_cache_directory
from worker.logger import logger

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

logging.set_verbosity_error()


class BaseModelManager:
    def __init__(self, download_reference=True):
        self.path = get_cache_directory()
        self.models = {}
        self.available_models = []
        self.loaded_models = {}
        self.tainted_models = []
        self.pkg = importlib_resources.files("nataili")
        self.models_db_name = "models"
        self.models_path = self.pkg / f"{self.models_db_name}.json"
        self.cuda_available = torch.cuda.is_available()
        self.cuda_devices, self.recommended_gpu = self.get_cuda_devices()
        self.remote_db = (
            f"https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/{self.models_db_name}.json"
        )
        self.download_reference = download_reference

    def init(self, list_models=False):
        if self.download_reference:
            self.models = self.download_model_reference()
            logger.info(f"Downloaded model reference. Got {len(self.models)} models.")
        else:
            self.models = json.loads((self.models_path).read_text())
            logger.info(f"Loaded model reference. Got {len(self.models)} models.")
        if list_models:
            for model in self.models:
                logger.info(model)
        models_available = []
        for model in self.models:
            if self.check_model_available(model):
                models_available.append(model)
        self.available_models = models_available
        logger.info(f"Got {len(self.available_models)} available models.")
        if list_models:
            for model in self.available_models:
                logger.info(model)

    def download_model_reference(self):
        try:
            logger.init("Model Reference", status="Downloading")
            response = requests.get(self.remote_db)
            logger.init_ok("Model Reference", status="OK")
            models = response.json()
            return models
        except Exception as e:
            logger.init_err("Model Reference", status=f"Download failed: {e}")
            logger.init_warn("Model Reference", status="Local")
            return json.loads((self.models_path).read_text())

    def get_model(self, model_name):
        return self.models.get(model_name)

    def get_model_files(self, model_name):
        """
        :param model_name: Name of the model
        Returns the files for a model
        """
        if self.models[model_name]["type"] == "diffusers":
            return []
        return self.models[model_name]["config"]["files"]

    def get_model_download(self, model_name):
        """
        :param model_name: Name of the model
        Returns the download details for a model
        """
        return self.models[model_name]["config"]["download"]

    def get_available_models(self):
        """
        Returns the available models
        """
        return self.available_models

    def get_available_models_by_types(self, model_types=None):
        if not model_types:
            model_types = ["ckpt", "diffusers"]
        models_available = []
        for model in self.models:
            if self.models[model]["type"] in model_types and self.check_available(self.get_model_files(model)):
                models_available.append(model)
        return models_available

    def count_available_models_by_types(self, model_types=None):
        return len(self.get_available_models_by_types(model_types))

    def get_loaded_models(self):
        """
        Returns the loaded models
        """
        return self.loaded_models

    def get_loaded_model(self, model_name):
        """
        :param model_name: Name of the model
        Returns the loaded model
        """
        return self.loaded_models[model_name]

    def get_loaded_models_names(self, string=False):
        """
        :param string: If True, returns concatenated string of model names
        Returns a list of the loaded model names
        """
        return ["Deliberate"]
        if string:
            return ", ".join(self.loaded_models.keys())
        return list(self.loaded_models.keys())

    def is_model_loaded(self, model_name):
        """
        :param model_name: Name of the model
        Returns whether the model is loaded
        """
        return model_name in self.loaded_models

    def unload_model(self, model_name):
        """
        :param model_name: Name of the model
        Unloads a model
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            return True
        return False

    def unload_all_models(self):
        """
        Unloads all models
        """
        for model in self.loaded_models:
            del self.loaded_models[model]
        return True

    def taint_model(self, model_name):
        """Marks a model as not valid by removing it from available_models"""
        if model_name in self.available_models:
            self.available_models.remove(model_name)
            self.tainted_models.append(model_name)

    def taint_models(self, models):
        for model in models:
            self.taint_model(model)

    def validate_model(self, model_name, skip_checksum=False):
        """
        :param model_name: Name of the model
        :param skip_checksum: If True, skips checksum validation
        For each file in the model, checks if the file exists and if the checksum is correct
        Returns True if all files are valid, False otherwise
        """
        files = self.get_model_files(model_name)
        logger.debug(f"Validating {model_name} with {len(files)} files")
        logger.debug(files)
        for file_details in files:
            if ".yaml" in file_details["path"]:
                continue
            if not self.check_file_available(file_details["path"]):
                logger.debug(f"File {file_details['path']} not found")
                return False
            if not skip_checksum and not self.validate_file(file_details):
                logger.debug(f"File {file_details['path']} has invalid checksum")
                return False
        return True

    @staticmethod
    def get_file_md5sum_hash(file_name):
        # Bail out if the source file doesn't exist
        if not os.path.isfile(file_name):
            return

        # Check if we have a cached md5 hash for the source file
        # and use that unless our source file is newer than our hash
        md5_file = f"{os.path.splitext(file_name)[0]}.md5"
        source_timestamp = os.path.getmtime(file_name)
        if os.path.isfile(md5_file):
            hash_timestamp = os.path.getmtime(md5_file)
        else:
            hash_timestamp = 0
        if hash_timestamp > source_timestamp:
            # Use our cached hash
            with open(md5_file, "rt") as handle:
                md5_hash = handle.read().split()[0]
            return md5_hash

        # Calculate the hash of the source file
        with open(file_name, "rb") as file_to_check:
            file_hash = hashlib.md5()
            while True:
                chunk = file_to_check.read(8192)  # Changed just because it broke pylint
                if not chunk:
                    break
                file_hash.update(chunk)
        md5_hash = file_hash.hexdigest()

        # Cache this md5 hash we just calculated. Use md5sum format files
        # so we can also use OS tools to manipulate these md5 files
        try:
            with open(md5_file, "wt") as handle:
                handle.write(f"{md5_hash} *{os.path.basename(md5_file)}")
        except (OSError, PermissionError):
            logger.debug("Could not write to md5sum file, ignoring")

        return md5_hash

    @staticmethod
    def get_file_sha256_hash(file_name):
        if not os.path.isfile(file_name):
            raise FileNotFoundError("No file {}".format(file_name))

        # Check if we have a cached sha256 hash for the source file
        # and use that unless our source file is newer than our hash
        sha256_file = f"{os.path.splitext(file_name)[0]}.sha256"
        source_timestamp = os.path.getmtime(file_name)
        if os.path.isfile(sha256_file):
            hash_timestamp = os.path.getmtime(sha256_file)
        else:
            hash_timestamp = 0
        if hash_timestamp > source_timestamp:
            # Use our cached hash
            with open(sha256_file, "rt") as handle:
                sha256_hash = handle.read().split()[0]
            return sha256_hash

        # Calculate the hash of the source file
        with open(file_name, "rb") as file_to_check:
            file_hash = hashlib.sha256()
            while True:
                chunk = file_to_check.read(8192)
                if not chunk:
                    break
                file_hash.update(chunk)
        sha256_hash = file_hash.hexdigest()

        # Cache this sha256 hash we just calculated. Use sha256sum format files
        # so we can also use OS tools to manipulate these md5 files
        try:
            with open(sha256_file, "wt") as handle:
                handle.write(f"{sha256_hash} *{os.path.basename(sha256_file)}")
        except (OSError, PermissionError):
            logger.debug("Could not write to sha256sum file, ignoring")

        return sha256_hash

    def validate_file(self, file_details):
        """
        :param file_details: A single file from the model's files list
        Checks if the file exists and if the checksum is correct
        Returns True if the file is valid, False otherwise
        """
        full_path = f"{self.path}/{file_details['path']}"

        # Default to sha256 hashes
        if "sha256sum" in file_details:
            logger.debug(f"Getting sha256sum of {full_path}")
            sha256_file_hash = self.get_file_sha256_hash(full_path)
            logger.debug(f"sha256sum: {sha256_file_hash}")
            logger.debug(f"Expected: {file_details['sha256sum']}")
            if file_details["sha256sum"] != sha256_file_hash:
                return False
            else:
                return True

        # If sha256 is not available, fall back to md5
        if "md5sum" in file_details:
            logger.debug(f"Getting md5sum of {full_path}")
            md5_file_hash = self.get_file_md5sum_hash(full_path)
            logger.debug(f"md5sum: {md5_file_hash}")
            logger.debug(f"Expected: {file_details['md5sum']}")
            if file_details["md5sum"] != md5_file_hash:
                return False
            else:
                return True

        # If no hashes available, return True for now
        # THIS IS A SECURITY RISK, EVENTUALLY WE SHOULD RETURN FALSE
        # But currently not all models specify hashes
        return True

    def check_file_available(self, file_path):
        """
        :param file_path: Path of the model's file. File is from the model's files list.
        Checks if the file exists
        Returns True if the file exists, False otherwise
        """
        full_path = f"{self.path}/{file_path}"
        return os.path.exists(full_path)

    def check_available(self, files):
        """
        :param files: List of files from the model's files list
        Checks if all files exist
        Returns True if all files exist, False otherwise
        """
        available = True
        for file in files:
            if ".yaml" in file["path"]:
                continue
            if not self.check_file_available(file["path"]):
                available = False
        return available

    def download_file(self, url, file_path):
        """
        :param url: URL of the file to download. URL is from the model's download list.
        :param file_path: Path of the model's file. File is from the model's files list.
        Downloads a file
        """
        full_path = f"{self.path}/{file_path}"
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        pbar_desc = full_path.split("/")[-1]
        r = requests.get(url, stream=True, allow_redirects=True)
        with open(full_path, "wb") as f:
            with tqdm(
                # all optional kwargs
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=pbar_desc,
                total=int(r.headers.get("content-length", 0)),
                disable=disable_download_progress.active,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=16 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    def download_model(self, model_name):
        """
        :param model_name: Name of the model
        Checks if the model is available, downloads the model if it is not available.
        After download, validates the model.
        Returns True if the model is available, False otherwise.

        Supported download types:
        - http(s) (url)
        - git (repo url)

        Other:
        - write content to file
        - symlink file
        - delete file
        - unzip file
        """
        if model_name in self.available_models and model_name not in self.tainted_models:
            logger.info(f"{model_name} is already available.")
            return True
        download = self.get_model_download(model_name)
        files = self.get_model_files(model_name)
        for i in range(len(download)):
            file_path = (
                f"{download[i]['file_path']}/{download[i]['file_name']}"
                if "file_path" in download[i]
                else files[i]["path"]
            )

            if "file_url" in download[i]:
                download_url = download[i]["file_url"]
            if "file_name" in download[i]:
                download_name = download[i]["file_name"]
            if "file_path" in download[i]:
                download_path = download[i]["file_path"]

            if "manual" in download[i]:
                logger.warning(
                    f"The model {model_name} requires manual download from {download_url}. "
                    f"Please place it in {download_path}/{download_name} then press ENTER to continue..."
                )
                input("")
                continue
            # TODO: simplify
            if "file_content" in download[i]:
                file_content = download[i]["file_content"]
                logger.info(f"writing {file_content} to {file_path}")
                os.makedirs(os.path.join(self.path, download_path), exist_ok=True)
                with open(os.path.join(self.path, os.path.join(download_path, download_name)), "w") as f:
                    f.write(file_content)
            elif "symlink" in download[i]:
                logger.info(f"symlink {file_path} to {download[i]['symlink']}")
                symlink = download[i]["symlink"]
                os.makedirs(os.path.join(self.path, download_path), exist_ok=True)
                os.symlink(symlink, os.path.join(self.path, os.path.join(download_path, download_name)))
            elif "git" in download[i]:
                logger.info(f"git clone {download_url} to {file_path}")
                os.makedirs(os.path.join(self.path, file_path), exist_ok=True)
                git.Git(os.path.join(self.path, file_path)).clone(download_url)
            elif "unzip" in download[i]:
                zip_path = f"{self.path}/{download_name}.zip"
                temp_path = f"{self.path}/{str(uuid4())}/"
                os.makedirs(temp_path, exist_ok=True)
                self.download_file(download_url, zip_path)
                logger.info(f"unzip {zip_path}")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_path)
                logger.info(f"moving {temp_path} to {download_path}")
                shutil.move(temp_path, os.path.join(self.path, download_path))
                logger.info(f"delete {zip_path}")
                os.remove(zip_path)
                logger.info(f"delete {temp_path}")
                shutil.rmtree(temp_path)
            else:
                if not self.check_file_available(file_path) or model_name in self.tainted_models:
                    logger.debug(f"Downloading {download_url} to {file_path}")
                    self.download_file(download_url, file_path)
        if not self.validate_model(model_name):
            return False
        self.init()
        return True

    def download_all_models(self):
        """
        Downloads all models
        """
        for model in self.get_filtered_model_names(download_all=True):
            if not self.check_model_available(model):
                logger.init(f"{model}", status="Downloading")
                self.download_model(model)
            else:
                logger.info(f"{model} is already downloaded.")
        return True

    def check_model_available(self, model_name):
        """
        :param model_name: Name of the model
        Checks if the model is available.
        Returns True if the model is available, False otherwise.
        """
        if model_name not in self.models:
            return False
        return self.check_available(self.get_model_files(model_name))

    def get_cuda_devices(self):
        """
        Checks if CUDA is available.
        If CUDA is available, it returns a list of all available CUDA devices.
        If CUDA is not available, it returns an empty list.
        CUDA Device info: id, name, sm
        List is sorted by sm (compute capability) in descending order.
        Also returns the recommended GPU (highest sm).
        """

        if torch.cuda.is_available():
            number_of_cuda_devices = torch.cuda.device_count()
            cuda_arch = []
            for i in range(number_of_cuda_devices):
                cuda_device = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "sm": torch.cuda.get_device_capability(i)[0] * 10 + torch.cuda.get_device_capability(i)[1],
                }
                cuda_arch.append(cuda_device)
            cuda_arch = sorted(cuda_arch, key=lambda k: k["sm"], reverse=True)
            recommended_gpu = [x for x in cuda_arch if x["sm"] == cuda_arch[0]["sm"]]
            return cuda_arch, recommended_gpu
        else:
            return None, None

    def get_filtered_models(self, **kwargs):
        """
        Get all models
        :param kwargs: filter based on metadata of the model reference db
        :return: list of models
        """
        filtered_models = self.models
        for keyword in kwargs:
            iterating_models = filtered_models.copy()
            filtered_models = {}
            for model in iterating_models:
                if iterating_models[model].get(keyword) == kwargs[keyword]:
                    filtered_models[model] = iterating_models[model]
        return filtered_models

    def get_filtered_model_names(self, **kwargs):
        """
        Get all model names
        :param kwargs: filter based on metadata of the model reference db
        :return: list of model names
        """
        filtered_models = self.get_filtered_models(**kwargs)
        return list(filtered_models.keys())
