"""Published seed sets (§4.6 Seed Rotation) and the registry that loads them.

§2.1.1 requires the client's request-issue / sample-order RNG and the per-query salt
to be seeded from a set MLCommons published for the submission's cohort, and §9.1's
"Seed-set validity" row rejects a submission whose points disagree or whose bound set
is not one of those published.

The set list ships as data (``data/seed_sets.yaml``) rather than code so a newly
published set does not need a checker release: point ``--seed-sets`` or
``$MLPERF_ENDPOINTS_SEED_SETS`` at a newer file and the check follows it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = [
    "SEED_SETS_ENV_VAR",
    "SeedSet",
    "SeedSetError",
    "bundled_seed_sets_path",
    "load_seed_sets",
]

#: Environment variable naming a replacement seed-set file.
SEED_SETS_ENV_VAR = "MLPERF_ENDPOINTS_SEED_SETS"

_BUNDLED = Path(__file__).parent / "data" / "seed_sets.yaml"

#: The RNG seed fields a set defines, in the spelling `seedset.yaml` uses.
_SEED_FIELDS = ("scheduler_rng_seed", "sample_index_rng_seed", "model_seed")


class SeedSetError(ValueError):
    """Raised when a seed-set file cannot be read or is malformed."""


@dataclass(frozen=True)
class SeedSet:
    """One published seed set.

    Attributes:
        id: The set's identifier, e.g. ``"A"``. A submission names this in every
            point's ``seed_set`` field.
        scheduler_rng_seed: Seed for the request-issue scheduler RNG.
        sample_index_rng_seed: Seed for the sample-order RNG.
        model_seed: Seed passed through to the model / per-query salt.
        cohorts: Cohorts this set was published for. Empty when the upstream
            registry carries no cohort keys, which makes §4.6's four-cohort
            adoption window unevaluable — see :mod:`submission_checker.seed_sets`.
    """

    id: str
    scheduler_rng_seed: int
    sample_index_rng_seed: int
    model_seed: int
    cohorts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def seeds(self) -> dict[str, int]:
        """The set's seeds keyed by field name, for comparison against a point."""
        return {name: getattr(self, name) for name in _SEED_FIELDS}


def bundled_seed_sets_path() -> Path:
    """Path to the seed-set file shipped with this checker."""
    return _BUNDLED


def load_seed_sets(path: Path | None = None) -> dict[str, SeedSet]:
    """Load the published seed sets, newest override winning.

    Resolution order: the explicit *path*, then ``$MLPERF_ENDPOINTS_SEED_SETS``, then
    the bundled file.

    Args:
        path: An explicit seed-set file, e.g. from ``--seed-sets``.

    Returns:
        Every set keyed by its id.

    Raises:
        SeedSetError: If the chosen file is missing, unreadable, not valid YAML, or
            does not carry a well-formed ``seed_sets`` list.
    """
    chosen = path or _env_path() or _BUNDLED
    try:
        raw = yaml.safe_load(chosen.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SeedSetError(f"Cannot read seed-set file {chosen}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SeedSetError(f"Invalid YAML in seed-set file {chosen}: {exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("seed_sets"), list):
        raise SeedSetError(f"{chosen} must be a mapping with a 'seed_sets' list")

    sets: dict[str, SeedSet] = {}
    for index, entry in enumerate(raw["seed_sets"]):
        if not isinstance(entry, dict):
            raise SeedSetError(f"{chosen}: seed_sets[{index}] is not a mapping")
        set_id = entry.get("id")
        if not isinstance(set_id, str) or not set_id:
            raise SeedSetError(f"{chosen}: seed_sets[{index}] has no 'id'")
        try:
            seeds = {name: int(entry[name]) for name in _SEED_FIELDS}
        except (KeyError, TypeError, ValueError) as exc:
            raise SeedSetError(
                f"{chosen}: seed set {set_id!r} must define integer"
                f" {', '.join(_SEED_FIELDS)}: {exc}"
            ) from exc
        cohorts = entry.get("cohorts") or []
        if not isinstance(cohorts, list):
            raise SeedSetError(f"{chosen}: seed set {set_id!r} has a non-list 'cohorts'")
        sets[set_id] = SeedSet(id=set_id, cohorts=tuple(str(c) for c in cohorts), **seeds)

    if not sets:
        raise SeedSetError(f"{chosen} defines no seed sets")
    return sets


def _env_path() -> Path | None:
    """The seed-set file named by the environment, if any."""
    value = os.environ.get(SEED_SETS_ENV_VAR)
    return Path(value) if value else None
