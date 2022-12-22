"""Arg parsing for the main script."""
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-i",
    "--interval",
    action="store",
    required=False,
    type=int,
    default=1,
    help="The amount of seconds with which to check if there's new prompts to generate",
)
arg_parser.add_argument(
    "-a",
    "--api_key",
    action="store",
    required=False,
    type=str,
    help="The API key corresponding to the owner of this Horde instance",
)
arg_parser.add_argument(
    "-n",
    "--worker_name",
    action="store",
    required=False,
    type=str,
    help="The server name for the Horde. It will be shown to the world and there can be only one.",
)
arg_parser.add_argument(
    "-u",
    "--horde_url",
    action="store",
    required=False,
    type=str,
    help="The SH Horde URL. Where the worker will pickup prompts and send the finished generations.",
)
arg_parser.add_argument(
    "--priority_usernames",
    type=str,
    action="append",
    required=False,
    help="Usernames which get priority use in this horde instance. The owner's username is always in this list.",
)
arg_parser.add_argument(
    "-p",
    "--max_power",
    type=int,
    required=False,
    help="How much power this instance has to generate pictures. Min: 2",
)
arg_parser.add_argument(
    "--queue_size",
    type=int,
    required=False,
    help="How many requests to keep in the queue. Min: 0",
)
arg_parser.add_argument(
    "--sfw",
    action="store_true",
    required=False,
    help="Set to true if you do not want this worker generating NSFW images.",
)
arg_parser.add_argument(
    "--blacklist",
    nargs="+",
    required=False,
    help="List the words that you want to blacklist.",
)
arg_parser.add_argument(
    "--censorlist",
    nargs="+",
    required=False,
    help="List the words that you want to censor.",
)
arg_parser.add_argument(
    "--censor_nsfw",
    action="store_true",
    required=False,
    help="Set to true if you want this worker worker to censor NSFW images.",
)
arg_parser.add_argument(
    "--allow_img2img",
    action="store_true",
    required=False,
    help="Set to true if you want this worker worker to allow img2img request.",
)
arg_parser.add_argument(
    "--allow_painting",
    action="store_true",
    required=False,
    help="Set to true if you want this worker worker to allow inpainting/outpainting requests.",
)
arg_parser.add_argument(
    "--allow_unsafe_ip",
    action="store_true",
    required=False,
    help="Set to true if you want this worker worker to allow img2img requests from unsafe IPs.",
)
arg_parser.add_argument(
    "-m",
    "--model",
    action="store",
    required=False,
    help="Which model to run on this horde.",
)
arg_parser.add_argument("--debug", action="store_true", default=False, help="Show debugging messages.")
arg_parser.add_argument(
    "-v",
    "--verbosity",
    action="count",
    default=0,
    help=(
        "The default logging level is ERROR or higher. "
        "This value increases the amount of logging seen in your screen"
    ),
)
arg_parser.add_argument(
    "-q",
    "--quiet",
    action="count",
    default=0,
    help=(
        "The default logging level is ERROR or higher. "
        "This value decreases the amount of logging seen in your screen"
    ),
)
arg_parser.add_argument(
    "--log_file",
    action="store_true",
    default=False,
    help="If specified will dump the log to the specified file",
)
arg_parser.add_argument(
    "--skip_md5",
    action="store_true",
    default=False,
    help="If specified will not check the downloaded model md5sum.",
)
arg_parser.add_argument(
    "--disable_voodoo",
    action="store_true",
    default=False,
    help=(
        "If specified this worker will not use voodooray to offload models into RAM and save VRAM"
        " (useful for cloud providers)."
    ),
)
arg_parser.add_argument(
    "--disable_xformers",
    action="store_true",
    default=False,
    help=(
        "If specified this worker will not try use xformers to speed up generations."
        " This should normally be automatic, but in case you need to disable it manually, you can do so here."
    ),
)
arg_parser.add_argument(
    "--disable_local_ray_temp",
    action="store_true",
    default=False,
    help=("If specified this worker will make the system use the default ray path for temp files instead of local."),
)
arg_parser.add_argument(
    "--hf_token",
    action="store",
    type=str,
    required=False,
    help="When defined, will use this huggingface token to download models.",
)
arg_parser.add_argument(
    "-y",
    "--yes",
    action="store_true",
    required=False,
    help="Specify this argument to autodownload all missing models defined in your bridgeData.py",
)
arg_parser.add_argument(
    "--disable_dynamic_models",
    action="store_true",
    default=False,
    help=("If specified this worker will not use dynamic models."),
)

args = arg_parser.parse_args()
