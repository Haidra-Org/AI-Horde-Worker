# test_download_models
from nataili.model_manager import ModelManager
import creds

# TODO: huggingface_hub or some way to use token instead of username/password
hf_auth = {"username": creds.hf_username, "password": creds.hf_password}
mm = ModelManager(hf_auth=hf_auth)

mm.init()

mm.download_all()