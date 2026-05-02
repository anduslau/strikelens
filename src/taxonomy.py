"""Helpers for StrikeLens label taxonomy."""

FINE_TO_COARSE_STRIKE_TYPE = {
    "jab": "punches",
    "cross": "punches",
    "hook": "punches",
    "uppercut": "punches",
    "cut_kick": "linear_kicks",
    "hopstep_kick": "linear_kicks",
    "roundhouse_kick": "round_kicks",
    "cheapshot_kick": "round_kicks",
    "back_kick": "spinning_kicks",
    "spinninghook_kick": "spinning_kicks",
    "tornado_kick": "spinning_kicks",
    "double_kick": "double_kicks",
    "frontdouble_kick": "double_kicks",
    "axe_kick": "axe_kicks",
    "hopaxe_kick": "axe_kicks",
    "crescentaxe_kick": "axe_kicks",
}


def get_coarse_strike_type(fine_label: str) -> str:
    """Map a fine strike label to its coarse movement family."""
    normalized_label = str(fine_label or "").strip().lower()
    if not normalized_label:
        return "unknown"
    return FINE_TO_COARSE_STRIKE_TYPE.get(normalized_label, "unknown")
