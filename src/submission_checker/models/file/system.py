"""System description model — §8.2 hardware and software metadata."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = ["Division", "NodeType", "SystemAvailabilityStatus", "SystemDescription"]


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
    host_memory_configuration: str | None = None
    accelerator_model_name: str | None = None
    accelerators_per_node: int | None = None
    accelerator_memory_capacity: str | None = None
    accelerator_memory_type: str | None = None
    accelerator_interconnect: str | None = None
    accelerator_host_interconnect: str | None = None
    host_network_card_count: str | None = None
    host_networking: str | None = None
    host_storage_capacity: str | None = None
    host_storage_type: str | None = None
    other_hardware: str | None = None
    hw_notes: str | None = None
    cooling: str | None = None
    inference_backend: str | None = None
    driver: str | None = None
    operating_system: str | None = None
    filesystem: str | None = None
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


class SystemDescription(BaseModel):
    """Parsed contents of ``systems/<system_id>.json`` (§8.2).

    Flat structure: org/system/model/dataset fields at the top level,
    per-node hardware and software metadata in the ``node_types`` list.
    """

    model_config = ConfigDict(extra="allow")

    # Org / submission metadata
    submitter_org_names: str
    submitter_contact: str | None = None
    submission_id: str | None = None
    submission_date: str | None = None
    publish_date: str | None = None

    # System metadata
    system_name: str
    system_category: str
    system_availability_status: SystemAvailabilityStatus
    max_supported_concurrency: int
    system_size: str | None = None
    system_node_ensemble_count: int | None = None
    system_node_ensemble_total: int | None = None
    serving_framework: str | None = None
    node_types: list[NodeType]

    # Division / model metadata
    division: Division
    model_id: str | None = None
    model_name: str | None = None
    model_precision: str | None = None
    link_to_model: str | None = None
    link_to_model_transformation: str | None = None
    model_notes: str | None = None

    # Dataset metadata
    dataset_id: str | None = None
    dataset_name: str | None = None
    input_token_average: float | None = None
    output_token_average: float | None = None
    dataset_type: str | None = None
    dataset_link: str | None = None

    # Accuracy
    measured_accuracy_score: str | float | None = None

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
