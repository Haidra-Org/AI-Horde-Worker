"""Arg parsing for the main script."""
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

