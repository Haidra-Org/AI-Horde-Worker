"""Convenience enums for job status"""
from enum import IntEnum


class JobStatus(IntEnum):
    """Job status enum"""

    OUT_OF_MEMORY = -2
    FAULTED = -1
    INIT = 0
    POLLING = 1
    WORKING = 2
    FINALIZING = 3
    DONE = 4
    FINALIZING_FAULTED = 5
