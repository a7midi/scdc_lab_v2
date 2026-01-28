from __future__ import annotations

from typing import Dict, Set


def active_set_from_state(state: Dict[int, int], vacuum_value: int = 0) -> Set[int]:
    """Return vertices whose state differs from vacuum_value.

    Kept as a tiny helper to avoid circular imports and to be usable from experiments.
    """
    vv = int(vacuum_value)
    return {int(v) for v, a in state.items() if int(a) != vv}
