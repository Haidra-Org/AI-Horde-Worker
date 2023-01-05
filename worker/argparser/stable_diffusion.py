"""Arg parsing for the main script."""
from worker.argparser.framework import arg_parser

arg_parser.add_argument(
    "-p",
    "--max_power",
    type=int,
    required=False,
    help="How much power this instance has to generate pictures. Min: 2",
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

args = arg_parser.parse_args()
