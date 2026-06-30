import enum


class MemoryLocation(enum.Enum):
    HOST = 0
    DEVICE = 1


class ScalarType(enum.Enum):
    UNDEFINED = 0
    BOOL = 1
    FLOAT16 = 2
    FLOAT32 = 3
    FLOAT64 = 4
    INT8 = 5
    INT16 = 6
    INT32 = 7
    INT64 = 8
    UINT8 = 9
    UINT16 = 10
    UINT32 = 11
    UINT64 = 12


class Device:
    def __init__(self) -> None: ...
    def dispose(self) -> None: ...

    @staticmethod
    def create_device(device_index: int = 0, enable_validation_layers: bool = False) -> "Device": ...

    def allocate_tensor_dlpack(self, shape: list[int], scalar_type: ScalarType, location: MemoryLocation) -> object: ...


def create_device(device_index: int = 0, enable_validation_layers: bool = False) -> Device: ...

