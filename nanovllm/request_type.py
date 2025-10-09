import enum

class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """
    ADD = b'\x00'
    ABORT = b'\x01'
    START_DP_WAVE = b'\x02'
    UTILITY = b'\x03'
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b'\x04'
    # Sentinel used within EngineCore.
    SHUTDOWN = b'\x05'
