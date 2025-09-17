from typing import Any

import lue.framework as lfr


def phase_detect_from_temperature(temperature: Any) -> Any:

    phase_state: Any = lfr.where(
        (temperature > 0),  # fluid or unfrozen
        1,
        0,
    )

    return phase_state
