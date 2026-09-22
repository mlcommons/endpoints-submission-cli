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

from .cohorts import Cohort, adoption_window

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
        cohorts: The cohorts during which a *new* submission may adopt this set —
            §4.6's four-cohort adoption window, derived from the registry's
            publication cohort. Empty only when the file declares no ``cohort-id``,
            in which case the adoption test cannot run.
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

    entries, published = _unwrap(chosen, raw)

    # §4.6 states the window in terms of the publication cohort, so the registry
    # carries one `cohort-id` rather than listing the window per set.
    cohorts: tuple[str, ...] = ()
    if published is not None:
        parsed = Cohort.parse(published)
        if parsed is None:
            raise SeedSetError(
                f"{chosen}: cohort-id {published!r} is not of the form YYYY-MM-C0/C1"
            )
        cohorts = tuple(str(c) for c in adoption_window(parsed))

    sets: dict[str, SeedSet] = {}
    for index, entry in enumerate(entries):
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
        # A per-set `cohorts` list overrides the derived window, so a future registry
        # that publishes sets on different schedules needs no loader change.
        explicit = entry.get("cohorts")
        if explicit is not None and not isinstance(explicit, list):
            raise SeedSetError(f"{chosen}: seed set {set_id!r} has a non-list 'cohorts'")
        window = tuple(str(c) for c in explicit) if explicit is not None else cohorts
        sets[set_id] = SeedSet(id=set_id, cohorts=window, **seeds)

    if not sets:
        raise SeedSetError(f"{chosen} defines no seed sets")
    return sets


def _unwrap(path: Path, raw: object) -> tuple[list[object], str | None]:
    """Return ``(seed-set entries, publication cohort)`` from either registry shape.

    Upstream nests everything under a ``cohort:`` mapping that also carries the
    ``cohort-id`` §4.6 counts from. The older flat ``seed_sets:`` form is still
    accepted so an operator can point ``--seed-sets`` at a file predating that
    change; such a file simply declares no cohort, and the adoption test stands down.
    """
    if not isinstance(raw, dict):
        raise SeedSetError(f"{path} must be a mapping")
    cohort = raw.get("cohort")
    if isinstance(cohort, dict):
        entries = cohort.get("seed_sets")
        published = cohort.get("cohort-id")
        if not isinstance(entries, list):
            raise SeedSetError(f"{path}: cohort.seed_sets must be a list")
        return entries, str(published) if published is not None else None
    entries = raw.get("seed_sets")
    if not isinstance(entries, list):
        raise SeedSetError(f"{path} must define cohort.seed_sets (or a top-level seed_sets list)")
    return entries, None


def _env_path() -> Path | None:
    """The seed-set file named by the environment, if any."""
    value = os.environ.get(SEED_SETS_ENV_VAR)
    return Path(value) if value else None
