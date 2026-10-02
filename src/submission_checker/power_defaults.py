# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""MLCommons default power reference values — Appendix D of the Endpoints rules.

These are what the checker falls back on when ``system_power.json`` leaves a value
absent (Appendix E.1): "a figure asserted without a public source is treated as
absent, the default applies, and the result is tagged 'MLC Estimated Power'".

Only the defaults that can be selected mechanically are here. D.1's two scale-up
references depend on the link protocol, which the descriptor does not declare, so an
absent scale-up figure has no fallback and is reported rather than guessed.

Appendix D says MLCommons revisits these "at least once per quarter". When that
starts happening, `data/seed_sets.yaml` and `data/approved_drafters.yaml` are the
pattern for moving this table out of the release.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "PowerDefault",
    "accelerator_default",
    "cpu_default",
    "nic_default",
    "switch_default",
]


@dataclass(frozen=True)
class PowerDefault:
    """One Appendix D value: watts, and the subsection it came from."""

    watts: float
    source: str


# D.2 — (architecture, max cores inclusive or None for "above the previous row").
_X86_SMALL = PowerDefault(350.0, "D.2")
_X86_LARGE = PowerDefault(500.0, "D.2")
_ARM_SMALL = PowerDefault(300.0, "D.2")
_X86_CORE_SPLIT = 64
_ARM_CORE_LIMIT = 128  # D.2's WG open item: no ARM default above this

_ARM_WORDS = re.compile(r"\b(grace|neoverse|arm|graviton|axion|ampere|altra|agi)\b")
_X86_WORDS = re.compile(r"\b(xeon|epyc|intel|amd|x86|x86_64)\b")
_CORES_IN_NAME = re.compile(r"(\d+)[\s-]*core")

# D.3, most specific first: "gb300" must win over "b300" for an NVIDIA GB300.
_ACCELERATORS: tuple[tuple[str, float], ...] = (
    ("gb300", 1400.0),
    ("gb200", 1200.0),
    ("b300", 1100.0),
    ("b200", 1000.0),
    ("mi455x", 2500.0),
    ("mi355x", 1400.0),
    ("mi350x", 1000.0),
    ("tpuv7", 1000.0),
    ("tn3", 700.0),
)

# D.4 reference switches: watts by cabling. A cabling with no published figure is
# absent, and D.4 says that switch "cannot serve as a reference for that case".
_SWITCHES: tuple[tuple[str, dict[str, float]], ...] = (
    ("sn6810ld", {"passive": 1960.0, "active_optical": 1960.0}),  # "not differentiated"
    ("sn5610", {"passive": 900.0, "active_optical": 2080.0}),
    ("sn5400", {"passive": 670.0}),
    ("sn4700", {"passive": 630.0}),
)

#: D.4's scale-out NIC default — "a PCIe network adapter".
_NIC = PowerDefault(75.0, "D.4")


def _squash(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def cpu_default(model: str | None, cores: int | None) -> PowerDefault | None:
    """D.2's per-CPU TDP, from the architecture the model name implies and its cores.

    The architecture is read from the model name, since §8.2 has no field for it.
    Where the core count is not supplied it is read from the name too ("AMD Venice
    256-core"). Returns ``None`` where either cannot be established, and for ARM above
    128 cores, which D.2 leaves undefined.
    """
    if not model:
        return None
    text = model.lower()
    if cores is None:
        match = _CORES_IN_NAME.search(text)
        cores = int(match.group(1)) if match else None
    if cores is None:
        return None
    if _ARM_WORDS.search(text):
        return _ARM_SMALL if cores <= _ARM_CORE_LIMIT else None
    if _X86_WORDS.search(text):
        return _X86_SMALL if cores <= _X86_CORE_SPLIT else _X86_LARGE
    return None


def accelerator_default(model: str | None) -> PowerDefault | None:
    """D.3's per-accelerator TDP for a listed model, or ``None``."""
    if not model:
        return None
    squashed = _squash(model)
    for key, watts in _ACCELERATORS:
        if key in squashed:
            return PowerDefault(watts, "D.3")
    return None


def switch_default(model: str | None, cabling: str | None) -> PowerDefault | None:
    """D.4's typical power for a reference switch under the declared cabling."""
    if not model or not cabling:
        return None
    squashed = _squash(model)
    for key, by_cabling in _SWITCHES:
        if key in squashed:
            watts = by_cabling.get(cabling)
            return PowerDefault(watts, "D.4") if watts is not None else None
    return None


def nic_default() -> PowerDefault:
    """D.4's scale-out NIC default."""
    return _NIC
