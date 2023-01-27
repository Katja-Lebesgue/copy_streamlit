from enum import EnumMeta
from typing import Any

def get_enum_values(_enum: EnumMeta, sort: bool = False) -> list[Any]:
    values = [item.value for item in _enum]
    if sort:
        values.sort()
    return values