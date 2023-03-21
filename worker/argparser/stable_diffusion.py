"""Arg parsing for the main script."""
from nataili import disable_progress, disable_voodoo, disable_xformers, enable_local_ray_temp

from worker.argparser.framework import arg_parser

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
    "-m",
    "--model",
    action="store",
    required=False,
    help="Which model to run on this horde.",
)
arg_parser.add_argument(
    "--disable_dynamic_models",
    action="store_true",
    default=False,
    help=("If specified this worker will not use dynamic models."),
)
arg_parser.add_argument(
    "--disable_post_processing",
    action="store_true",
    default=False,
    help=("If specified this worker will not load post-processors."),
)
arg_parser.add_argument(
    "--disable_controlnet",
    action="store_true",
    default=False,
    help=("If specified this worker will not pick up controlnet jobs"),
)

args = arg_parser.parse_args()

disable_xformers.toggle(args.disable_xformers)
enable_local_ray_temp.toggle(not args.disable_local_ray_temp)
disable_voodoo.toggle(args.disable_voodoo)
if disable_voodoo.active:
    enable_local_ray_temp.disable()
disable_progress.activate()
