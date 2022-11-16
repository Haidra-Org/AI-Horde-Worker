from nataili import disable_local_ray_temp, disable_voodoo, disable_xformers

from .argparser import args
from .bridge_data import BridgeData
from .enums import JobStatus
from .stats import BridgeStats

bridge_stats = BridgeStats()

disable_xformers.toggle(args.disable_xformers)
disable_local_ray_temp.toggle(args.disable_local_ray_temp)
disable_voodoo.toggle(args.disable_voodoo)
if disable_voodoo.active:
    disable_local_ray_temp.activate()

from .job import HordeJob
