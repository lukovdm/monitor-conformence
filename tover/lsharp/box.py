from typing import Literal


def box_compare(b1: bool | Literal["unknown"], b2: bool | Literal["unknown"]) -> bool:
    if b1 == "unknown" or b2 == "unknown":
        return True

    return b1 == b2
