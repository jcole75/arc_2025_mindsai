from __future__ import annotations

from typing import Any


# Shared scoring thresholds
Z_SCORE_THRESHOLD = 999999
MIN_VOTES_SINGLE_PRED = 999999
MIN_VOTES_AMBIGUOUS = 999999

# Grid constraints
MAX_GRID_SIZE = 100
MAX_SYMBOLS = 10
MAX_PERMUTATIONS_TO_SAMPLE = 500

# Symbol encoding configuration
SYMBOL_ENCODING: dict[str, Any] = {
    "enabled": False,
    "scheme": "letters",  # currently supports 'letters' for a-j
}
