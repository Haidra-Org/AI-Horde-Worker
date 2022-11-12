from enum import IntEnum


class JobStatus(IntEnum):
    FAULTED = -1
    INIT = 0
    POLLING = 1
    WORKING = 2
    FINALIZING = 3
    DONE = 4
