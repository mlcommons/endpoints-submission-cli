"""System description model — §8.2 hardware and software metadata."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StringConstraints,
    field_validator,
    model_validator,
)

__all__ = [
    "AcceleratorInfo",
    "ConfigSummary",
    "DatasetAccuracyScores",
    "Division",
    "NodeType",
    "SystemAvailabilityStatus",
    "SystemDescription",
]


class Division(str, Enum):
    """Submission division (§2)."""

    STANDARDIZED = "Standardized"
    SERVICED = "Serviced"
    RDI = "RDI"


class SystemAvailabilityStatus(str, Enum):
    """System availability status (§8.2)."""

    AVAILABLE = "Available"
    PREVIEW = "Preview"
    RDI = "RDI"


#: Accelerator fields §8.2.1 moved from the node into ``accelerator_info[]``.
_ACCELERATOR_FIELDS = (
    "accelerator_model_name",
    "accelerators_per_node",
    "accelerator_memory_capacity",
    "accelerator_memory_type",
    "accelerator_interconnect",
    "accelerator_host_interconnect",
)


class AcceleratorInfo(BaseModel):
    """One accelerator configuration within a node type (§8.2.1).

    v0.7 carried these fields flat on the node, which could describe only one kind of
    accelerator per node; v1.0 nests a list so a heterogeneous node can disclose each.
    """

    model_config = ConfigDict(extra="allow")

    accelerator_model_name: str | None = None
    accelerators_per_node: int | None = None
    accelerator_memory_capacity: str | None = None
    accelerator_memory_type: str | None = None
    accelerator_interconnect: str | None = None
    accelerator_host_interconnect: str | None = None


class NodeType(BaseModel):
    """Per-node hardware and software configuration (§8.2.1)."""

    model_config = ConfigDict(extra="allow")

    system_node_ensemble_id: int | None = None
    number_of_nodes: int | None = None
    host_processor_model_name: str | None = None
    host_processors_per_node: int | None = None
    host_processor_core_count: int | None = None
    host_processor_vcpu_count: int | None = None
    host_memory_capacity: str | None = None
    host_memory_configuration: str
    accelerator_info: list[AcceleratorInfo] = Field(default_factory=list)
    host_network_card_count: str
    host_networking: str | None = None
    host_storage_capacity: str | None = None
    host_storage_type: str | None = None
    other_hardware: str | None = None
    hw_notes: str | None = None
    cooling: str | None = None
    inference_backend: str | None = None
    driver: str
    operating_system: str | None = None
    filesystem: str
    container_link: str | None = None
    other_software_stack: str | None = None
    sw_notes: str | None = None

    @field_validator("system_node_ensemble_id", mode="before")
    @classmethod
    def _coerce_to_int(cls, v: object) -> object:
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return v
        return v

    @model_validator(mode="before")
    @classmethod
    def _lift_flat_accelerator_fields(cls, data: object) -> object:
        """Fold v0.7's flat ``accelerator_*`` node fields into ``accelerator_info[]``.

        The same deprecation shape used elsewhere in this package: read the old
        spelling, normalise to the new one, and let ``extra="allow"`` keep the
        originals so nothing the submitter wrote is lost. An explicit
        ``accelerator_info`` always wins — a submitter who wrote both meant the new one.
        """
        if not isinstance(data, dict) or data.get("accelerator_info"):
            return data
        flat = {name: data[name] for name in _ACCELERATOR_FIELDS if data.get(name) is not None}
        if flat:
            data = {**data, "accelerator_info": [flat]}
        return data

    @model_validator(mode="after")
    def _require_core_or_vcpu_count(self) -> NodeType:
        """A node must disclose at least one of physical core count or vCPU count."""
        if self.host_processor_core_count is None and self.host_processor_vcpu_count is None:
            raise ValueError(
                "node_types entry must specify host_processor_core_count or"
                " host_processor_vcpu_count"
            )
        return self


class ConfigSummary(BaseModel):
    """Structured parallelism/config summary — the object form of ``config_summary``.

    Moved here from the retired ``run_metadata.json`` model: policies PR #119 deleted
    that file and §8.2 absorbed its fields into the system description.
    """

    model_config = ConfigDict(extra="allow")

    disaggregated: bool | None = None
    expert_parallel: int | None = None
    tensor_parallel: int | None = None
    pipeline_parallel: int | None = None
    data_parallel: int | None = None
    batch: int | None = None


#: ``config_summary`` accepts either the structured object or free-form prose.
_ConfigSummaryStr = Annotated[str, StringConstraints(min_length=4)]

#: The three spellings the docs use for the availability / publication-status field.
_AVAILABILITY_SPELLINGS = (
    "publication_status",
    "system_availability_status",
    "availability_status",
)


class DatasetAccuracyScores(BaseModel):
    """Per-dataset accuracy scores for ``measured_accuracy_score`` (§8.2).

    Maps a single dataset to a ``scores`` dictionary of ``score_name -> score_value``,
    e.g. ``{"scores": {"exact_match": 84.01, "rouge1": 38.73}}``. Each score value must
    be a float — strings are rejected rather than parsed (ints widen to float).
    """

    model_config = ConfigDict(extra="allow")

    scores: dict[str, StrictFloat]


class SystemDescription(BaseModel):
    """Parsed contents of a point's ``system_desc.json`` (§8.2).

    Written into every ``r<N>/`` directory since policies PR #119; the points of one
    curve must agree on everything but ``tps_utilization``.

    Flat structure: org/system/model/dataset fields at the top level,
    per-node hardware and software metadata in the ``node_types`` list.
    """

    model_config = ConfigDict(extra="allow")

    # Org / submission metadata. §8.2 moved these out of the system description in
    # v1.0; still read when present because the existing corpus carries them.
    submitter_org_names: str | None = None
    submitter_contact: str | None = None
    submission_id: str | None = None
    submission_date: str | None = None
    publish_date: str | None = None

    # System metadata
    system_name: str
    system_category: str | None = None
    #: §8.2's name for availability. §8.2.1's template and v0.7 both call it
    #: ``system_availability_status``; a third spelling, ``availability_status``,
    #: appears elsewhere in the docs. All three are accepted — see
    #: :meth:`_reconcile_availability_spellings`.
    publication_status: SystemAvailabilityStatus
    max_supported_concurrency: int
    system_size: str
    system_node_ensemble_count: int
    system_node_ensemble_total: int
    serving_framework: str | None = None
    shortened_system_name: str | None = None
    endpoint_url: str | None = None
    node_types: list[NodeType]

    # Deployment configuration, absorbed into §8.2 when policies PR #119 removed
    # run_metadata.json. Optional for now: no submission in the corpus predates the
    # move, so requiring them would reject every existing bundle.
    node_config: str | None = None
    config_summary: ConfigSummary | _ConfigSummaryStr | None = None
    config_summary_notes: str | None = None
    disaggregated: bool | None = None
    expert_parallel: int | None = None
    tensor_parallel: int | None = None
    pipeline_parallel: int | None = None
    data_parallel: int | None = None
    batch: int | None = None
    link_config: str | None = None
    #: ``system_tps / max(system_tps)`` over the point's own curve. A per-point value
    #: in a file §8.2 calls a system description — raised with the WG; see
    #: :meth:`~submission_checker.checker.SubmissionChecker._check_tps_utilization`.
    tps_utilization: float | None = None

    # Division / model metadata. §8.2 keeps `division` and `model_name`; everything
    # else here moved to point.yaml (§8.3) and is optional, not dropped — an existing
    # bundle that still carries it is not wrong, and model_id feeds the model-name
    # consistency check.
    division: Division
    model_id: str | None = None
    model_name: str | None = None
    model_precision: str | None = None
    link_to_model: str | None = None
    link_to_model_transformation: str | None = None
    model_notes: str | None = None

    # Dataset metadata — moved to point.yaml (§8.3).
    dataset_id: str | None = None
    dataset_name: str | None = None
    input_token_average: float | None = None
    output_token_average: float | None = None
    dataset_type: str | None = None
    dataset_link: str | None = None

    # Accuracy, also moved to point.yaml. A scalar (str/float) is accepted for now;
    # the structured per-dataset form {dataset: {"scores": {name: value}}} is also
    # accepted and will become the only valid form in a future round.
    measured_accuracy_score: str | float | dict[str, DatasetAccuracyScores] | None = None

    @field_validator("division", mode="before")
    @classmethod
    def _coerce_division(cls, v: object) -> object:
        if isinstance(v, str):
            mapping = {"standardized": "Standardized", "serviced": "Serviced", "rdi": "RDI"}
            normalized = mapping.get(v.strip().lower())
            if normalized is not None:
                return normalized
            raise ValueError(f"Unknown division {v!r}. Must be one of: standardized, serviced, rdi")
        return v

    @model_validator(mode="before")
    @classmethod
    def _reconcile_availability_spellings(cls, data: object) -> object:
        """Accept all three spellings of the availability field, rejecting disagreement.

        §8.2's table says ``publication_status``, §8.2.1's template and v0.7 say
        ``system_availability_status``, and ``availability_status`` appears elsewhere in
        the docs. Guessing which one a submitter meant is only safe when they agree, so
        two conflicting values are an error rather than a silent pick.
        """
        if not isinstance(data, dict):
            return data
        present = {
            name: data[name] for name in _AVAILABILITY_SPELLINGS if data.get(name) not in (None, "")
        }
        if not present:
            return data
        distinct = {str(v).strip().lower() for v in present.values()}
        if len(distinct) > 1:
            pairs = ", ".join(f"{k}={v!r}" for k, v in sorted(present.items()))
            raise ValueError(f"Conflicting availability values: {pairs}")
        if data.get("publication_status") in (None, ""):
            data = {**data, "publication_status": next(iter(present.values()))}
        return data

    @field_validator("publication_status", mode="before")
    @classmethod
    def _coerce_availability(cls, v: object) -> object:
        if isinstance(v, str):
            mapping = {"available": "Available", "preview": "Preview", "rdi": "RDI"}
            normalized = mapping.get(v.strip().lower())
            if normalized is not None:
                return normalized
            raise ValueError(f"Unknown availability {v!r}. Must be one of: available, preview, rdi")
        return v

    @field_validator("input_token_average", "output_token_average", mode="before")
    @classmethod
    def _coerce_tokens_to_float(cls, v: object) -> object:
        if isinstance(v, str) and v:
            try:
                return float(v)
            except ValueError:
                return v
        return v

    @field_validator("measured_accuracy_score", mode="before")
    @classmethod
    def _coerce_empty_accuracy_to_none(cls, v: object) -> object:
        if v == "" or v is None:
            return None
        return v
