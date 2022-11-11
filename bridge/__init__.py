from .argparser import args
from .enums import JobStatus
from nataili import disable_voodoo, disable_xformers, disable_local_ray_temp
from .bridgedata import BridgeData

disable_xformers.toggle(args.disable_xformers)
disable_local_ray_temp.toggle(args.disable_local_ray_temp)
disable_voodoo.toggle(args.disable_voodoo)
if disable_voodoo.active:
    disable_local_ray_temp.activate()

from .job import HordeJob
