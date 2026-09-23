# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for the published seed-set registry (§4.6 Seed Rotation)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from submission_checker.seed_sets import (
    SEED_SETS_ENV_VAR,
    SeedSetError,
    bundled_seed_sets_path,
    load_seed_sets,
)

#: The values published in `mlcommons/endpoints_policies` PR #117's seedset.yaml.
#: Asserted literally: these are the seeds every v1.0 submission is checked against,
#: so a typo here would silently reject every compliant bundle.
_PR117_SET_A = {
    "scheduler_rng_seed": 10487924139932647040,
    "sample_index_rng_seed": 586478644936801402,
    "model_seed": 9315206023656308754,
}


def _write_registry(path: Path, entries: list[dict]) -> Path:
    path.write_text(yaml.safe_dump({"seed_sets": entries}))
    return path


class TestBundledRegistry:
    def test_mirrors_pr117(self) -> None:
        seed_set = load_seed_sets()["A"]
        assert seed_set.seeds == _PR117_SET_A

    def test_seeds_are_not_the_v07_constant(self) -> None:
        """v0.7 fixed every seed at 42; v1.0 rotates them, which is why the rule changed."""
        assert all(value != 42 for value in load_seed_sets()["A"].seeds.values())

    def test_adoption_window_is_derived_from_the_publication_cohort(self) -> None:
        """§4.6: adoptable for the publication cohort and the three following it."""
        assert load_seed_sets()["A"].cohorts == (
            "2026-10-C1",
            "2026-11-C0",
            "2026-11-C1",
            "2026-12-C0",
        )

    def test_bundled_file_ships_with_the_package(self) -> None:
        assert bundled_seed_sets_path().is_file()


class TestRegistryShapes:
    """Upstream nests under `cohort:`; the older flat form is still read."""

    def test_upstream_cohort_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "seeds.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "cohort": {
                        "version": 1.0,
                        "cohort-id": "2027-03-C0",
                        "seed_sets": [
                            {
                                "id": "Z",
                                "scheduler_rng_seed": 1,
                                "sample_index_rng_seed": 2,
                                "model_seed": 3,
                            }
                        ],
                    }
                }
            )
        )
        z = load_seed_sets(path)["Z"]
        assert z.cohorts == ("2027-03-C0", "2027-03-C1", "2027-04-C0", "2027-04-C1")

    def test_flat_shape_still_parses_but_declares_no_window(self, tmp_path: Path) -> None:
        """A file predating the cohort wrapper leaves the adoption test unevaluable."""
        path = tmp_path / "seeds.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "seed_sets": [
                        {
                            "id": "Z",
                            "scheduler_rng_seed": 1,
                            "sample_index_rng_seed": 2,
                            "model_seed": 3,
                        }
                    ]
                }
            )
        )
        assert load_seed_sets(path)["Z"].cohorts == ()

    def test_per_set_cohorts_override_the_derived_window(self, tmp_path: Path) -> None:
        path = tmp_path / "seeds.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "cohort": {
                        "cohort-id": "2026-10-C1",
                        "seed_sets": [
                            {
                                "id": "Z",
                                "scheduler_rng_seed": 1,
                                "sample_index_rng_seed": 2,
                                "model_seed": 3,
                                "cohorts": ["2030-01-C0"],
                            }
                        ],
                    }
                }
            )
        )
        assert load_seed_sets(path)["Z"].cohorts == ("2030-01-C0",)


class TestOverrides:
    def test_explicit_path_wins(self, tmp_path: Path) -> None:
        path = _write_registry(
            tmp_path / "seeds.yaml",
            [{"id": "Z", "scheduler_rng_seed": 1, "sample_index_rng_seed": 2, "model_seed": 3}],
        )
        sets = load_seed_sets(path)
        assert list(sets) == ["Z"]
        assert sets["Z"].seeds == {
            "scheduler_rng_seed": 1,
            "sample_index_rng_seed": 2,
            "model_seed": 3,
        }

    def test_environment_variable_is_honoured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write_registry(
            tmp_path / "seeds.yaml",
            [{"id": "E", "scheduler_rng_seed": 1, "sample_index_rng_seed": 2, "model_seed": 3}],
        )
        monkeypatch.setenv(SEED_SETS_ENV_VAR, str(path))
        assert list(load_seed_sets()) == ["E"]

    def test_explicit_path_beats_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_path = _write_registry(
            tmp_path / "env.yaml",
            [{"id": "E", "scheduler_rng_seed": 1, "sample_index_rng_seed": 2, "model_seed": 3}],
        )
        arg_path = _write_registry(
            tmp_path / "arg.yaml",
            [{"id": "A", "scheduler_rng_seed": 4, "sample_index_rng_seed": 5, "model_seed": 6}],
        )
        monkeypatch.setenv(SEED_SETS_ENV_VAR, str(env_path))
        assert list(load_seed_sets(arg_path)) == ["A"]

    def test_cohorts_are_read_when_present(self, tmp_path: Path) -> None:
        """Forward-compatibility: the loader already reads the key §4.6 needs."""
        path = _write_registry(
            tmp_path / "seeds.yaml",
            [
                {
                    "id": "A",
                    "scheduler_rng_seed": 1,
                    "sample_index_rng_seed": 2,
                    "model_seed": 3,
                    "cohorts": ["2026-09-C0", "2026-09-C1"],
                }
            ],
        )
        assert load_seed_sets(path)["A"].cohorts == ("2026-09-C0", "2026-09-C1")


class TestMalformedRegistries:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SeedSetError, match="Cannot read"):
            load_seed_sets(tmp_path / "absent.yaml")

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "seeds.yaml"
        path.write_text("seed_sets: [\n")
        with pytest.raises(SeedSetError, match="Invalid YAML"):
            load_seed_sets(path)

    def test_not_a_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "seeds.yaml"
        path.write_text("- just\n- a\n- list\n")
        with pytest.raises(SeedSetError, match="must be a mapping"):
            load_seed_sets(path)

    def test_malformed_cohort_id(self, tmp_path: Path) -> None:
        path = tmp_path / "seeds.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "cohort": {
                        "cohort-id": "not-a-cohort",
                        "seed_sets": [
                            {
                                "id": "A",
                                "scheduler_rng_seed": 1,
                                "sample_index_rng_seed": 2,
                                "model_seed": 3,
                            }
                        ],
                    }
                }
            )
        )
        with pytest.raises(SeedSetError, match="YYYY-MM-C0/C1"):
            load_seed_sets(path)

    def test_cohort_without_seed_sets(self, tmp_path: Path) -> None:
        path = tmp_path / "seeds.yaml"
        path.write_text(yaml.safe_dump({"cohort": {"cohort-id": "2026-10-C1"}}))
        with pytest.raises(SeedSetError, match="cohort.seed_sets must be a list"):
            load_seed_sets(path)

    def test_empty_seed_set_list(self, tmp_path: Path) -> None:
        with pytest.raises(SeedSetError, match="defines no seed sets"):
            load_seed_sets(_write_registry(tmp_path / "seeds.yaml", []))

    def test_entry_without_id(self, tmp_path: Path) -> None:
        path = _write_registry(
            tmp_path / "seeds.yaml",
            [{"scheduler_rng_seed": 1, "sample_index_rng_seed": 2, "model_seed": 3}],
        )
        with pytest.raises(SeedSetError, match="has no 'id'"):
            load_seed_sets(path)

    def test_entry_missing_a_seed(self, tmp_path: Path) -> None:
        path = _write_registry(
            tmp_path / "seeds.yaml", [{"id": "A", "scheduler_rng_seed": 1, "model_seed": 3}]
        )
        with pytest.raises(SeedSetError, match="must define integer"):
            load_seed_sets(path)

    def test_entry_with_a_non_integer_seed(self, tmp_path: Path) -> None:
        path = _write_registry(
            tmp_path / "seeds.yaml",
            [
                {
                    "id": "A",
                    "scheduler_rng_seed": "not-a-number",
                    "sample_index_rng_seed": 2,
                    "model_seed": 3,
                }
            ],
        )
        with pytest.raises(SeedSetError, match="must define integer"):
            load_seed_sets(path)
