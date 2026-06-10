"""System description model — §8.2 hardware and software metadata."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = ["Division", "NodeType", "SystemDescription"]


class Division(str, Enum):
    """Submission division (§2)."""

    STANDARDIZED = "Standardized"
    SERVICED = "Serviced"
    RDI = "RDI"


class NodeType(BaseModel):
    """Per-node hardware and software configuration."""

    model_config = ConfigDict(extra="allow")

    system_node_ensemble_id: Optional[int] = None
    number_of_nodes: Optional[int] = None
    host_processor_model_name: Optional[str] = None
    host_processors_per_node: Optional[int] = None
    host_processor_core_count: Optional[int] = None
    host_processor_vcpu_count: Optional[int] = None
    host_memory_capacity: Optional[str] = None
    host_memory_configuration: Optional[str] = None
    accelerator_model_name: Optional[str] = None
    accelerators_per_node: Optional[int] = None
    accelerator_memory_capacity: Optional[str] = None
    accelerator_memory_type: Optional[str] = None
    accelerator_interconnect: Optional[str] = None
    accelerator_host_interconnect: Optional[str] = None
    host_network_card_count: Optional[str] = None
    host_networking: Optional[str] = None
    host_storage_capacity: Optional[str] = None
    host_storage_type: Optional[str] = None
    other_hardware: Optional[str] = None
    hw_notes: Optional[str] = None
    cooling: Optional[str] = None
    inference_backend: Optional[str] = None
    driver: Optional[str] = None
    operating_system: Optional[str] = None
    filesystem: Optional[str] = None
    container_link: Optional[str] = None
    other_software_stack: Optional[str] = None
    sw_notes: Optional[str] = None

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
    submitter_contact: Optional[str] = None
    submission_id: Optional[str] = None
    submission_date: Optional[str] = None
    publish_date: Optional[str] = None

    # System metadata
    system_name: str
    system_category: str
    system_availability_status: str
    system_size: Optional[str] = None
    system_node_ensemble_count: Optional[int] = None
    system_node_ensemble_total: Optional[int] = None
    serving_framework: Optional[str] = None
    node_types: list[NodeType]

    # Division / model metadata
    division: Division
    model_id: Optional[str] = None
    model_name: Optional[str] = None
    model_precision: Optional[str] = None
    link_to_model: Optional[str] = None
    link_to_model_transformation: Optional[str] = None
    model_notes: Optional[str] = None

    # Dataset metadata
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    input_token_average: Optional[float] = None
    output_token_average: Optional[float] = None
    dataset_type: Optional[str] = None
    dataset_link: Optional[str] = None

    # Accuracy
    measured_accuracy_score: Optional[Union[str, float]] = None

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
