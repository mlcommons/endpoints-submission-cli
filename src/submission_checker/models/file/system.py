"""System description model — §8.2 hardware and software metadata."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, StrictFloat, field_validator, model_validator

__all__ = [
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


class NodeType(BaseModel):
    """Per-node hardware and software configuration."""

    model_config = ConfigDict(extra="allow")

    system_node_ensemble_id: int | None = None
    number_of_nodes: int | None = None
    host_processor_model_name: str | None = None
    host_processors_per_node: int | None = None
    host_processor_core_count: int | None = None
    host_processor_vcpu_count: int | None = None
    host_memory_capacity: str | None = None
    host_memory_configuration: str
    accelerator_model_name: str | None = None
    accelerators_per_node: int | None = None
    accelerator_memory_capacity: str | None = None
    accelerator_memory_type: str
    accelerator_interconnect: str | None = None
    accelerator_host_interconnect: str | None = None
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

    @model_validator(mode="after")
    def _require_core_or_vcpu_count(self) -> NodeType:
        """A node must disclose at least one of physical core count or vCPU count."""
        if self.host_processor_core_count is None and self.host_processor_vcpu_count is None:
            raise ValueError(
                "node_types entry must specify host_processor_core_count or"
                " host_processor_vcpu_count"
            )
        return self


class DatasetAccuracyScores(BaseModel):
    """Per-dataset accuracy scores for ``measured_accuracy_score`` (§8.2).

    Maps a single dataset to a ``scores`` dictionary of ``score_name -> score_value``,
    e.g. ``{"scores": {"exact_match": 84.01, "rouge1": 38.73}}``. Each score value must
    be a float — strings are rejected rather than parsed (ints widen to float).
    """

    model_config = ConfigDict(extra="allow")

    scores: dict[str, StrictFloat]


class SystemDescription(BaseModel):
    """Parsed contents of ``systems/<system_id>.json`` (§8.2).

    Flat structure: org/system/model/dataset fields at the top level,
    per-node hardware and software metadata in the ``node_types`` list.
    """

    model_config = ConfigDict(extra="allow")

    # Org / submission metadata
    submitter_org_names: str
    submitter_contact: str
    submission_id: str | None = None
    submission_date: str | None = None
    publish_date: str | None = None

    # System metadata
    system_name: str
    system_category: str
    system_availability_status: SystemAvailabilityStatus
    max_supported_concurrency: int
    system_size: str
    system_node_ensemble_count: int
    system_node_ensemble_total: int
    serving_framework: str | None = None
    node_types: list[NodeType]

    # Division / model metadata
    division: Division
    model_id: str
    model_name: str | None = None
    model_precision: str
    link_to_model: str
    link_to_model_transformation: str | None = None
    model_notes: str | None = None

    # Dataset metadata
    dataset_id: str
    dataset_name: str
    input_token_average: float
    output_token_average: float
    dataset_type: str
    dataset_link: str

    # Accuracy. A scalar (str/float) is accepted for now; the structured per-dataset
    # form {dataset: {"scores": {score_name: score_value}}} is also accepted and will
    # become the only valid form in a future round.
    measured_accuracy_score: str | float | dict[str, DatasetAccuracyScores]

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

    @field_validator("system_availability_status", mode="before")
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
