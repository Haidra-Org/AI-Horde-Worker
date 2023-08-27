"""Get and process a job from the horde"""
import base64
import json
import random
import time
import traceback
from io import BytesIO

import requests
from hordelib.horde import HordeLib
from hordelib.safety_checker import is_image_nsfw

from worker import csam
from worker.enums import JobStatus
from worker.jobs.framework import HordeJobFramework
from worker.jobs.kudos import KudosModel
from worker.logger import logger
from worker.post_process import post_process
from worker.stats import bridge_stats

SAVE_KUDOS_TRAINING_DATA = False
SIMULATE_KUDOS_LOCALLY = False


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
        self.hordelib = HordeLib()
        self.kudos_model = None
        if SIMULATE_KUDOS_LOCALLY:
            self.kudos_model = KudosModel("worker/jobs/kudos-v20-66.ckpt")
        self.job_kudos = 0

    @logger.catch(reraise=True)
    def start_job(self):
        """Starts a Stable Diffusion job from a pop request"""
        logger.debug("Starting job in threadpool for model: {}", self.current_model)
        super().start_job()
        if self.status == JobStatus.FAULTED:
            self.start_submit_thread()
            return
        self.stale_time = time.time() + (self.current_payload.get("ddim_steps", 50) * 5) + 10
        if self.current_payload.get("control_type"):
            self.stale_time = self.stale_time * 3
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
                "ddim_steps": self.current_payload["ddim_steps"],
                "sampler_name": self.current_payload["sampler_name"],
                "cfg_scale": self.current_payload["cfg_scale"],
                "seed": self.current_payload["seed"],
                "tiling": self.current_payload["tiling"],
                "karras": self.current_payload["karras"],
                "clip_skip": self.current_payload.get("clip_skip", 1),
                "n_iter": 1,
            }
            # These params might not always exist in the horde payload
            if source_image:
                gen_payload["source_image"] = source_image
            if source_image and source_mask:
                gen_payload["source_mask"] = source_mask
            if "denoising_strength" in self.current_payload and source_image:
                gen_payload["denoising_strength"] = self.current_payload["denoising_strength"]
            if "hires_fix" in self.current_payload and not source_image:
                gen_payload["hires_fix"] = self.current_payload["hires_fix"]
            if (
                "control_type" in self.current_payload
                and source_image
                and source_processing == "img2img"
                and "stable diffusion 2" not in model_baseline
            ):
                gen_payload["control_type"] = self.current_payload["control_type"]
                gen_payload["image_is_control"] = self.current_payload["image_is_control"]
                gen_payload["return_control_map"] = self.current_payload.get("return_control_map", False)
            if "loras" in self.current_payload:
                gen_payload["loras"] = self.current_payload["loras"]
            if "tis" in self.current_payload:
                gen_payload["tis"] = self.current_payload["tis"]
        except KeyError as err:
            logger.error("Received incomplete payload from job. Aborting. ({})", err)
            self.status = JobStatus.FAULTED
            self.start_submit_thread()
            return
        # logger.debug(gen_payload)
        req_type = "txt2img"
        if source_image:
            if source_processing == "img2img":
                req_type = "img2img"
            elif source_processing == "inpainting":
                req_type = "inpainting"
        # Reject jobs for pix2pix if not img2img
        if self.current_model in ["pix2pix"] and req_type != "img2img":
            logger.error(
                "Received an non-img2img request for Pix2Pix model. This shouldn't happen. "
                f"Inform the developer. Current payload {self.pop}",
            )
            self.status = JobStatus.FAULTED
            self.start_submit_thread()
            return
        logger.debug(
            f"{req_type} ({self.current_model}) request with id {self.current_id} picked up. Initiating work...",
        )
        if req_type == "inpainting" and source_mask is None:
            try:
                if source_image.mode == "P":
                    source_image.convert("RGBA")
                _red, _green, _blue, _alpha = source_image.split()
            except ValueError:
                logger.warning(
                    (
                        "inpainting image doesn't have an alpha channel. "
                        "This shouldn't happen. Continue processing without any mask."
                    ),
                )
        # This might change if we add more pipelines later.
        generator = self.hordelib.basic_inference
        try:
            logger.info(
                f"Starting generation for id {self.current_id}: {self.current_model} @ "
                f"{self.current_payload['width']}x{self.current_payload['height']} "
                f"for {self.current_payload.get('ddim_steps',50)} steps "
                f"{self.current_payload.get('sampler_name','unknown sampler')}. "
                f"Prompt length is {len(self.current_payload['prompt'])} characters "
                f"and it appears to contain {len(gen_payload.get('loras', []))} "
                f"LoRas: {[lora['name'] for lora in gen_payload.get('loras', [])]} "
                f"and {len(gen_payload.get('tis', []))} "
                f"Textual Inversions: {[ti['name'] for ti in gen_payload.get('tis', [])]}",
            )
            time_state = time.time()
            gen_payload["model"] = self.current_model
            gen_payload["source_processing"] = req_type
            # logger.debug(gen_payload)
            self.image = generator(gen_payload)

            if SAVE_KUDOS_TRAINING_DATA or SIMULATE_KUDOS_LOCALLY:
                payload = gen_payload.copy()

            self.seed = int(self.current_payload["seed"])
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
                "Something went wrong when processing the request. "
                "Please check your trace.log file for the full stack trace. "
                f"Payload: {stack_payload}",
            )
            trace = "".join(traceback.format_exception(type(err), err, err.__traceback__))
            logger.trace(trace)
            if "OutOfMemoryError" in str(err):
                logger.error(
                    "This machine ran out of memory when processing the request. "
                    "You very likely need to reduce the max_power parameter in your config."
                    "\nKeep in mind that you may run fine for long periods of time until a worst case job is run.",
                    "\nHowever, If you are only seeing this error after very long periods of time, and you know "
                    "you have enough memory ordinarily for a worst case job, it is possible that there may be a "
                    "memory leak. In that case, report this issue, along with your bridge.log, to the developers.",
                )
                self.status = JobStatus.OUT_OF_MEMORY
            else:
                self.status = JobStatus.FAULTED
            self.start_submit_thread()
            return

        if self.image is None:
            stack_payload = gen_payload
            stack_payload["request_type"] = req_type
            stack_payload["model"] = self.current_model
            stack_payload["prompt"] = "PROMPT REDACTED"
            logger.error(
                "Something went wrong when processing request and image was returned as None. "
                f"Payload: {stack_payload}",
            )
            self.status = JobStatus.FAULTED
            self.start_submit_thread()
            logger.warning(f"Rescue: Attempting to unload {self.current_model}")
            self.model_manager.unload_model(self.current_model)
            return
        if use_nsfw_censor and is_image_nsfw(self.image):
            logger.info(f"Image censored with reason: {censor_reason}")
            self.image = censor_image
            self.censored = "censored"

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
                strength = self.current_payload.get("facefixer_strength", 0.5)
                self.image = post_process(post_processor, self.image, strength=strength)
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

        # Doing this here we include post processing time correctly
        if SAVE_KUDOS_TRAINING_DATA:
            # 15% validation data
            filename = "inference-time-data.json" if random.random() < 0.85 else "inference-time-data-validation.json"
            with open(filename, "at") as logfile:
                payload["time"] = round(time.time() - time_state, 4)
                payload["source_image"] = bool(payload.get("source_image"))
                payload["source_mask"] = bool(payload.get("source_mask"))
                payload["post_processing"] = self.current_payload.get("post_processing", [])
                del payload["prompt"]
                del payload["seed"]
                del payload["model"]
                del payload["tiling"]
                del payload["n_iter"]
                logfile.write(json.dumps(payload))
                logfile.write("\n")

        if SIMULATE_KUDOS_LOCALLY:
            # Award 0.5% additional bonus to kudos basis per model hosted to compensate
            # for the extra time spent loading the models from cache and storage.
            # percentage_bonus_per_model = 0.5  # 1/2 a percent
            # number_of_models = len(self.model_manager.get_loaded_models_names())
            # total_bonus = number_of_models * percentage_bonus_per_model
            # percentage_bonus = 1 + (total_bonus / 100)
            # Apply 25% basis kudos adjustment
            kudos_adjustment = 2.5
            # Calculate the kudos award
            self.job_kudos = self.kudos_model.calculate_kudos(payload, kudos_adjustment)

        self.start_submit_thread()

    def submit_job(self, endpoint="/api/v2/generate/submit"):
        """Submits the job to the server to earn our kudos."""
        super().submit_job(endpoint=endpoint)

    def prepare_submit_payload(self):
        # images, seed, info, stats = txt2img(**self.current_payload)
        buffer = BytesIO()
        # We send as WebP to avoid using all the horde bandwidth
        self.image.save(buffer, format="WebP", quality=self.upload_quality, method=6)
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
        kudos = self.job_kudos if SIMULATE_KUDOS_LOCALLY else submit_req.json()["reward"]
        bridge_stats.update_inference_stats(self.current_model, kudos)


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
