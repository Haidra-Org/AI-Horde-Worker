"""Arg parsing for the main script."""
from nataili import disable_progress, disable_voodoo, disable_xformers, enable_local_ray_temp

from worker.argparser.framework import arg_parser

arg_parser.add_argument(
    "-f",
    "--forms",
    action="append",
    type=list,
    required=False,
    default=None,
    help="Which forms to run on this horde.",
)

args = arg_parser.parse_args()

disable_xformers.activate()
enable_local_ray_temp.disable()
disable_voodoo.activate()
disable_progress.activate()
