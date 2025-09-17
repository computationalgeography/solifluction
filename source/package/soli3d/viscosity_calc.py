from typing import Any

import lue.framework as lfr


def viscosity_exp_temp(temperature: Any, a: float, b: float) -> Any:

    return a * lfr.exp(-temperature * b)
