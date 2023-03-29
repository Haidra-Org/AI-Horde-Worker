"""Get and process a job from the horde"""
import base64
import time
import traceback
from base64 import binascii
from io import BytesIO

import rembg
import requests
from nataili import InvalidModelCacheException
from nataili.stable_diffusion.compvis import CompVis
from nataili.stable_diffusion.diffusers.depth2img import Depth2Img
from nataili.stable_diffusion.diffusers.inpainting import inpainting
from nataili.util.logger import logger
from PIL import UnidentifiedImageError

from worker import csam
from worker.consts import KNOWN_INTERROGATORS, POST_PROCESSORS_NATAILI_MODELS
from worker.enums import JobStatus
from worker.jobs.framework import HordeJobFramework
from worker.post_process import post_process
from worker.stats import bridge_stats


class StableDiffusionHordeJob(HordeJobFramework):
    """Get and process a stable diffusion job from the horde"""

    def __init__(self, mm, bd, pop):
        super().__init__(mm, bd, pop)
        self.current_model = None
        self.upload_quality = 95
        self.seed = None
        self.image = None
        self.r2_upload = None
        self.censored = False
        self.available_models = self.model_manager.get_loaded_models_names()
        self.current_model = self.pop.get("model", self.available_models[0])
        self.current_id = self.pop["id"]
        self.current_payload = self.pop["payload"]
        self.r2_upload = self.pop.get("r2_upload", False)
        self.clip_model = None

    @logger.catch(reraise=True)
    def start_job(self):
        """Starts a Stable Diffusion job from a pop request"""
        logger.debug("Starting job in threadpool for model: {}", self.current_model)
        super().start_job()
        if self.status == JobStatus.FAULTED:
            self.start_submit_thread()
            return
        self.stale_time = time.time() + (self.current_payload.get("ddim_steps", 50) * 3) + 10
        if self.current_payload.get("control_type"):
            self.stale_time = self.stale_time * 3
        self.stale_time += 3 * count_parentheses(self.current_payload["prompt"])
        # PoC Stuff
        if "ViT-L/14" in self.available_models:
            logger.debug("ViT-L/14 model loaded")
            self.clip_model = self.model_manager.loaded_models["ViT-L/14"]
        else:
            self.clip_model = None
        # Here starts the Stable Diffusion Specific Logic
        # We allow a generation a plentiful 3 seconds per step before we consider it stale
        # Generate Image
        # logger.info([self.current_id,self.current_payload])
        censor_image = None
        censor_reason = None
        use_nsfw_censor = False
        if self.bridge_data.censor_nsfw and not self.bridge_data.nsfw:
            use_nsfw_censor = True
            censor_image = self.bridge_data.censor_image_sfw_worker
            censor_reason = "SFW worker"
        censorlist_prompt = self.current_payload["prompt"]
        if "###" in censorlist_prompt:
            censorlist_prompt, _censorlist_negprompt = censorlist_prompt.split("###", 1)
        if any(word in censorlist_prompt for word in self.bridge_data.censorlist):
            use_nsfw_censor = True
            censor_image = self.bridge_data.censor_image_censorlist
            censor_reason = "Censorlist"
        elif self.current_payload.get("use_nsfw_censor", False):
            use_nsfw_censor = True
            censor_image = self.bridge_data.censor_image_sfw_request
            censor_reason = "Requested"
        # use_gfpgan = self.current_payload.get("use_gfpgan", True)
        # use_real_esrgan = self.current_payload.get("use_real_esrgan", False)
        source_processing = self.pop.get("source_processing")
        source_image = self.pop.get("source_image")
        source_mask = self.pop.get("source_mask")
        model_baseline = self.model_manager.models[self.current_model].get("baseline")
        # These params will always exist in the payload from the horde
        try:
            gen_payload = {
                "prompt": self.current_payload["prompt"],
                "height": self.current_payload["height"],
                "width": self.current_payload["width"],
                "seed": self.current_payload["seed"],
                "tiling": self.current_payload["tiling"],
                "n_iter": 1,
                "batch_size": 1,
                "save_individual_images": False,
                "save_grid": False,
            }
            # These params might not always exist in the horde payload
            if "ddim_steps" in self.current_payload:
                gen_payload["ddim_steps"] = self.current_payload["ddim_steps"]
            if "sampler_name" in self.current_payload:
                gen_payload["sampler_name"] = self.current_payload["sampler_name"]
                if gen_payload["sampler_name"] == "dpmsolver" and "stable diffusion 2" not in model_baseline:
                    logger.warning(f"dpmsolver cannot be used with {self.current_model}. Falling back to k_euler.")
                    gen_payload["sampler_name"] = "k_euler"
                if gen_payload["sampler_name"] == "DDIM" and source_mask is not None:
                    logger.warning(
                        f"DDIM cannot be used with a mask for {self.current_model}. Falling back to k_euler.",
                    )
                    gen_payload["sampler_name"] = "k_euler"
                if gen_payload["sampler_name"] == "DDIM" and self.current_model == "pix2pix":
                    logger.warning(f"DDIM cannot be used with {self.current_model}. Falling back to k_euler_a.")
                    gen_payload["sampler_name"] = "k_euler_a"
            if "cfg_scale" in self.current_payload:
                gen_payload["cfg_scale"] = self.current_payload["cfg_scale"]
            if "ddim_eta" in self.current_payload:
                gen_payload["ddim_eta"] = self.current_payload["ddim_eta"]
            if "denoising_strength" in self.current_payload and source_image:
                gen_payload["denoising_strength"] = self.current_payload["denoising_strength"]
            if self.current_payload.get("karras", False) and gen_payload["sampler_name"] != "dpmsolver":
                gen_payload["sampler_name"] = gen_payload.get("sampler_name", "k_euler_a") + "_karras"
            if "hires_fix" in self.current_payload and not source_image:
                gen_payload["hires_fix"] = self.current_payload["hires_fix"]
            if (
                "control_type" in self.current_payload
                and source_image
                and source_processing == "img2img"
                and "stable diffusion 2" not in model_baseline
            ):
                gen_payload["control_type"] = self.current_payload["control_type"]
                gen_payload["init_as_control"] = self.current_payload["image_is_control"]
                gen_payload["return_control_map"] = self.current_payload.get("return_control_map", False)
        except KeyError as err:
            logger.error("Received incomplete payload from job. Aborting. ({})", err)
            self.status = JobStatus.FAULTED
            self.start_submit_thread()
            return
        # logger.debug(gen_payload)
        req_type = "txt2img"
        # TODO: Fix img2img for SD2
        if source_image:
            img_source = None
            img_mask = None
            if source_processing == "img2img":
                req_type = "img2img"
            elif source_processing == "inpainting":
                req_type = "inpainting"
            elif source_processing == "outpainting":
                req_type = "outpainting"
            if gen_payload["sampler_name"] == "dpmsolver":
                logger.warning("dpmsolver cannot handle img2img. Falling back to k_euler")
                gen_payload["sampler_name"] = "k_euler"
        # Prevent inpainting from picking text2img and img2img gens (as those go via compvis pipelines)
        if "_inpainting" in self.current_model and req_type not in [
            "inpainting",
            "outpainting",
        ]:
            # Try to find any other compvis model to do text2img or img2img
            for available_model in self.available_models:
                if (
                    "_inpainting" not in available_model
                    and available_model not in POST_PROCESSORS_NATAILI_MODELS | KNOWN_INTERROGATORS
                    and available_model in self.model_manager.compvis.get_loaded_models_names()
                ):
                    logger.debug(
                        [
                            self.current_model,
                            self.model_manager.models[self.current_model].get("baseline"),
                            source_processing,
                            req_type,
                        ],
                    )
                    self.current_model = available_model
                    logger.info(
                        "Model stable_diffusion_inpainting chosen for txt2img or img2img gen, "
                        + f"switching to {self.current_model} instead.",
                    )
                    break

            # if the model persists as inpainting for text2img or img2img, we abort.
            if "_inpainting" in self.current_model:
                # We remove the base64 from the prompt to avoid flooding the output on the error
                if isinstance(self.pop.get("source_image", ""), str) and len(self.pop.get("source_image", "")) > 10:
                    self.pop["source_image"] = len(self.pop.get("source_image", ""))
                if isinstance(self.pop.get("source_mask", ""), str) and len(self.pop.get("source_mask", "")) > 10:
                    self.pop["source_mask"] = len(self.pop.get("source_mask", ""))
                logger.error(
                    "Received an non-inpainting request for inpainting model. This shouldn't happen. "
                    f"Inform the developer. Current payload {self.pop}",
                )
                self.status = JobStatus.FAULTED
                self.start_submit_thread()
                return
                # TODO: Send faulted
        # Reject jobs for SD2Depth/pix2pix if not img2img
        if self.current_model in ["Stable Diffusion 2 Depth", "pix2pix"] and req_type != "img2img":
            # We remove the base64 from the prompt to avoid flooding the output on the error
            if (
                source_image is not None
                and isinstance(self.pop.get("source_image", ""), str)
                and len(self.pop.get("source_image", "")) > 10
            ):
                self.pop["source_image"] = len(self.pop.get("source_image", ""))
            if (
                source_mask is not None
                and isinstance(self.pop.get("source_mask", ""), str)
                and len(self.pop.get("source_mask", "")) > 10
            ):
                self.pop["source_mask"] = len(self.pop.get("source_mask", ""))
            logger.error(
                "Received an non-img2img request for SD2Depth or Pix2Pix model. This shouldn't happen. "
                f"Inform the developer. Current payload {self.pop}",
            )
            self.status = JobStatus.FAULTED
            self.start_submit_thread()
            return
        if "_inpainting" not in self.current_model and req_type == "inpainting":
            # Try to use default inpainting model if available
            if "stable_diffusion_inpainting" in self.available_models:
                self.current_model = "stable_diffusion_inpainting"
            else:
                req_type = "img2img"
        logger.debug(
            f"{req_type} ({self.current_model}) request with id {self.current_id} picked up. Initiating work...",
        )
        try:
            safety_checker = (
                self.model_manager.loaded_models["safety_checker"]["model"]
                if "safety_checker" in self.model_manager.loaded_models
                else None
            )

            if source_image:
                img_source = source_image
            if source_mask:
                img_mask = source_mask
                if img_mask.size != img_source.size:
                    logger.warning(
                        f"Source image/mask mismatch. Resizing mask from {img_mask.size} to {img_source.size}",
                    )
                    img_mask = img_mask.resize(img_source.size)
        except KeyError:
            self.status = JobStatus.FAULTED
            self.start_submit_thread()
            return
        # If the received image is unreadable, we continue as text2img
        except UnidentifiedImageError:
            logger.error("Source image received for img2img is unreadable. Falling back to text2img!")
            req_type = "txt2img"
            if "denoising_strength" in gen_payload:
                del gen_payload["denoising_strength"]
        except binascii.Error:
            logger.error(
                "Source image received for img2img is cannot be base64 decoded (binascii.Error). "
                "Falling back to text2img!",
            )
            req_type = "txt2img"
            if "denoising_strength" in gen_payload:
                del gen_payload["denoising_strength"]
        if self.current_model not in self.model_manager.loaded_models:
            logger.error(f"Required model {self.current_model} appears to be not loaded. Dynamic model? Aborting...")
            self.status = JobStatus.FAULTED
            self.start_submit_thread()
            return
        if req_type in {"img2img", "txt2img"}:
            if req_type == "img2img":
                gen_payload["init_img"] = img_source
                if img_mask:
                    gen_payload["init_mask"] = img_mask
            if self.current_model == "Stable Diffusion 2 Depth":
                if "save_grid" in gen_payload:
                    del gen_payload["save_grid"]
                if "sampler_name" in gen_payload:
                    del gen_payload["sampler_name"]
                if "init_mask" in gen_payload:
                    del gen_payload["init_mask"]
                if "tiling" in gen_payload:
                    del gen_payload["tiling"]
                if "control_type" in gen_payload:
                    del gen_payload["control_type"]
                    del gen_payload["init_as_control"]
                    del gen_payload["return_control_map"]
                generator = Depth2Img(
                    pipe=self.model_manager.loaded_models[self.current_model]["model"],
                    device=self.model_manager.loaded_models[self.current_model]["device"],
                    output_dir="bridge_generations",
                    load_concepts=True,
                    concepts_dir="models/custom/sd-concepts-library",
                    filter_nsfw=use_nsfw_censor,
                    disable_voodoo=self.bridge_data.disable_voodoo,
                )
            else:
                generator = CompVis(
                    model=self.model_manager.loaded_models[self.current_model],
                    model_name=self.current_model,
                    model_baseline=model_baseline,
                    output_dir="bridge_generations",
                    load_concepts=True,
                    concepts_dir="models/custom/sd-concepts-library",
                    safety_checker=safety_checker,
                    filter_nsfw=use_nsfw_censor,
                    disable_voodoo=self.bridge_data.disable_voodoo,
                    control_net_manager=self.model_manager.controlnet or None,
                )
        else:
            # These variables do not exist in the outpainting implementation
            if "save_grid" in gen_payload:
                del gen_payload["save_grid"]
            if "sampler_name" in gen_payload:
                del gen_payload["sampler_name"]
            if "denoising_strength" in gen_payload:
                del gen_payload["denoising_strength"]
            if "tiling" in gen_payload:
                del gen_payload["tiling"]
            if "control_type" in gen_payload:
                del gen_payload["control_type"]
                del gen_payload["init_as_control"]
                del gen_payload["return_control_map"]
            # We prevent sending an inpainting without mask or transparency, as it will crash us.
            if img_mask is None:
                try:
                    if img_source.mode == "P":
                        img_source.convert("RGBA")
                    _red, _green, _blue, _alpha = img_source.split()
                except ValueError:
                    logger.warning("inpainting image doesn't have an alpha channel. Aborting gen")
                    self.status = JobStatus.FAULTED
                    self.start_submit_thread()
                    return
            gen_payload["inpaint_img"] = img_source
            if img_mask:
                gen_payload["inpaint_mask"] = img_mask
            generator = inpainting(
                self.model_manager.loaded_models[self.current_model]["model"],
                self.model_manager.loaded_models[self.current_model]["device"],
                "bridge_generations",
                filter_nsfw=use_nsfw_censor,
                disable_voodoo=self.bridge_data.disable_voodoo,
            )
        try:
            logger.info(
                f"Starting generation for id {self.current_id}: {self.current_model} @ "
                f"{self.current_payload['width']}x{self.current_payload['height']} "
                f"for {self.current_payload.get('ddim_steps',50)} steps. "
                f"Prompt length is {len(self.current_payload['prompt'])} characters "
                f"And it appears to contain {count_parentheses(self.current_payload['prompt'])} weights",
            )
            time_state = time.time()
            try:
                generator.generate(**gen_payload)
            except InvalidModelCacheException:
                # There is no recovering from this right now, remove the model and fault the job
                # The model will be re-cached and re-loaded on the next job that uses this model.
                self.model_manager.unload_model(self.current_model)
                logger.warning(f"Unloaded model {self.current_model}")
                raise

            # Generation is ok
            logger.info(
                f"Generation for id {self.current_id} finished successfully"
                f" in {round(time.time() - time_state,1)} seconds.",
            )
        except Exception as err:
            stack_payload = gen_payload
            stack_payload["request_type"] = req_type
            stack_payload["model"] = self.current_model
            stack_payload["prompt"] = "PROMPT REDACTED"
            logger.error(
                "Something went wrong when processing request. "
                "Please check your trace.log file for the full stack trace. "
                f"Payload: {stack_payload}",
            )
            trace = "".join(traceback.format_exception(type(err), err, err.__traceback__))
            logger.trace(trace)
            self.status = JobStatus.FAULTED
            generator = None
            self.start_submit_thread()
            return
        self.image = generator.images[0]["image"]
        self.seed = generator.images[0]["seed"]
        if generator.images[0].get("censored", False):
            logger.info(f"Image censored with reason: {censor_reason}")
            self.image = censor_image
            self.censored = "censored"
        # We unload the generator from RAM
        generator = None

        # Run the CSAM Checker
        if not self.censored:
            is_csam, similarities, similarity_hits = csam.check_for_csam(
                clip_model=self.clip_model,
                image=self.image,
                prompt=self.current_payload["prompt"],
                model_info=self.model_manager.models[self.current_model],
            )
            if self.clip_model and is_csam:
                logger.warning(f"Current values for id {self.current_id} would create CSAM. Censoring!")
                self.image = self.bridge_data.censor_image_csam
                self.censored = "csam"

        # Run Post-Processors
        for post_processor in self.current_payload.get("post_processing", []):
            # Do not PP when censored
            if self.censored:
                continue
            logger.debug(f"Post-processing with {post_processor}...")
            try:
                if post_processor == "strip_background":
                    session = rembg.new_session("u2net")
                    self.image = rembg.remove(
                        self.image,
                        session=session,
                        only_mask=False,
                        alpha_matting=10,
                        alpha_matting_foreground_threshold=240,
                        alpha_matting_background_threshold=10,
                        alpha_matting_erode_size=10,
                    )
                    del session
                else:
                    strength = self.current_payload.get("facefixer_strength", 0.5)
                    self.image = post_process(post_processor, self.image, self.model_manager, strength=strength)
            except (AssertionError, RuntimeError) as err:
                logger.warning(
                    "Post-Processor '{}' encountered an error when working on image . Skipping! {}",
                    post_processor,
                    err,
                )
            # Edit the webp upload quality if post-processor used
            if self.r2_upload:
                self.upload_quality = 95
            else:
                self.upload_quality = (
                    45 if post_processor in ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"] else 75
                )
        logger.debug("post-processing done...")
        self.start_submit_thread()

    def submit_job(self, endpoint="/api/v2/generate/submit"):
        """Submits the job to the server to earn our kudos."""
        super().submit_job(endpoint=endpoint)

    def prepare_submit_payload(self):
        # images, seed, info, stats = txt2img(**self.current_payload)
        buffer = BytesIO()
        # We send as WebP to avoid using all the horde bandwidth
        self.image.save(buffer, format="WebP", quality=self.upload_quality)
        if self.r2_upload:
            put_response = requests.put(self.r2_upload, data=buffer.getvalue())
            generation = "R2"
            logger.debug("R2 Upload response: {}", put_response)
        else:
            generation = base64.b64encode(buffer.getvalue()).decode("utf8")
        self.submit_dict = {
            "id": self.current_id,
            "generation": generation,
            "seed": self.seed,
        }
        if self.censored:
            self.submit_dict["state"] = self.censored

    def post_submit_tasks(self, submit_req):
        bridge_stats.update_inference_stats(self.current_model, submit_req.json()["reward"])


def count_parentheses(s):
    open_p = False
    count = 0
    for c in s:
        if c == "(":
            open_p = True
        elif c == ")" and open_p:
            open_p = False
            count += 1
    return count
