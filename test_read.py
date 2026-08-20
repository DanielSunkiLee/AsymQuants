from typing import NamedTuple, Any
import numpy as np
import numpy.typing as npt
from enum import IntEnum

class GGUFValueType(IntEnum):
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

    @staticmethod
    def get_type(val: Any) -> GGUFValueType:
        if isinstance(val, (str, bytes, bytearray)):
            return GGUFValueType.STRING
        elif isinstance(val, list):
            return GGUFValueType.ARRAY
        elif isinstance(val, float):
            return GGUFValueType.FLOAT32
        elif isinstance(val, bool):
            return GGUFValueType.BOOL
        elif isinstance(val, int):
            return GGUFValueType.INT32
        # TODO: need help with 64-bit types in Python
        else:
            raise ValueError(f"Unknown type: {type(val)}")

class ReaderField(NamedTuple) :
    offset: int
    name: str
    parts: list[npt.NDArray[Any]] = []
    data: list[int] = [-1]
    types: list[GGUFValueType] = []

    def contents(self, index_or_slice: int | slice = slice(None)) -> Any :
        if self.types:
            to_string = lambda x: str(x.tobytes(), encoding='utf-8')
            main_type = self.types[0]

            if main_type == GGUFValueType.ARRAY :
                sub_type = self.types[-1]

                if sub_type == GGUFValueType.STRING :
                    indices = self.data[index_or_slice]

                    if isinstance(index_or_slice,int) :
                        return to_string(self.parts[indices])
                    else :
                        return [to_string(self.parts[idx]) for idx in indices]
                else :
                    if isinstance(index_or_slice,int) :
                        return self.parts[self.data[index_or_slice]].tolist()[0]
                    else :
                        return [pv for idx in self.data[index_or_slice] for pv in self.parts[idx].tolist()]
            
            if main_type == GGUFValueType.STRING:
                return to_string(self.parts[-1])
            else :
                return self.parts[-1].tolist()[0]
        return None