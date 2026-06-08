"""System description model — §8.2 hardware and software metadata."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

__all__ = ["Division", "PublicationStatus", "SystemDescription"]


class Division(str, Enum):
    """Submission division (§2)."""

    STANDARDIZED = "Standardized"
    SERVICED = "Serviced"
    RDI = "RDI"


class PublicationStatus(str, Enum):
    """Publication status category (§7)."""

    AVAILABLE = "Available"
    PREVIEW = "Preview"
    RDI = "RDI"


class OrganizationMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    submitter_org_name: str
    submitter_contact: Optional[str] = None
    submission_id: Optional[str] = None
    submission_date: Optional[str] = None
    publish_date: Optional[str] = None


class ProcessorInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    host_processor_model_name: Optional[str] = None
    host_processors_per_node: Optional[int] = None
    host_processor_core_count: Optional[int] = None
    host_processor_vcpu_count: Optional[int] = None

    @model_validator(mode="after")
    def _require_core_or_vcpu(self) -> ProcessorInfo:
        if self.host_processor_core_count is None and self.host_processor_vcpu_count is None:
            raise ValueError(
                "At least one of host_processor_core_count"
                " or host_processor_vcpu_count must be provided"
            )
        return self


class HostMemoryInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    host_memory_capacity: Optional[str] = None
    host_memory_configuration: Optional[str] = None


class AcceleratorInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    accelerator_model_name: Optional[str] = None
    accelerators_per_node: Optional[int] = None
    accelerator_memory_capacity: Optional[str] = None
    accelerator_memory_type: Optional[str] = None
    accelerator_interconnect: Optional[str] = None
    accelerator_host_interconnect: Optional[str] = None


class NetworkingInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    host_networking: Optional[str] = None
    host_network_card_count: Optional[str] = None


class StorageInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    host_storage_capacity: Optional[str] = None
    host_storage_type: Optional[str] = None


class HardwareEnsemble(BaseModel):
    model_config = ConfigDict(extra="allow")

    processor: Optional[ProcessorInfo] = None
    host_memory: Optional[HostMemoryInfo] = None
    accelerator: Optional[AcceleratorInfo] = None
    networking: Optional[NetworkingInfo] = None
    storage: Optional[StorageInfo] = None
    other_hardware: Optional[str] = None
    hw_notes: Optional[str] = None
    cooling: Optional[str] = None


class SoftwareEnsemble(BaseModel):
    model_config = ConfigDict(extra="allow")

    inference_backend: Optional[str] = None
    driver: Optional[str] = None
    operating_system: Optional[str] = None
    filesystem: Optional[str] = None
    container_link: Optional[str] = None
    other_software_stack: Optional[str] = None
    sw_notes: Optional[str] = None


class NodeType(BaseModel):
    model_config = ConfigDict(extra="allow")

    system_node_ensemble_id: Optional[int] = None

    @field_validator("system_node_ensemble_id", mode="before")
    @classmethod
    def _coerce_to_int(cls, v: object) -> object:
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return v
        return v
    number_of_nodes: Optional[int] = None
    hardware_ensemble: Optional[HardwareEnsemble] = None
    software_ensemble: Optional[SoftwareEnsemble] = None


class SystemMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    system_name: str
    system_category: str
    system_availability_status: str
    system_size: Optional[str] = None
    system_node_ensemble_count: Optional[int] = None
    system_node_ensemble_total: Optional[int] = None


class SystemUnderTest(BaseModel):
    model_config = ConfigDict(extra="allow")

    system_metadata: SystemMetadata
    node_types: list[NodeType]
    serving_framework: Optional[str] = None


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    division: Division
    model_id: Optional[str] = None
    model_name: Optional[str] = None
    model_precision: Optional[str] = None
    link_to_model: Optional[str] = None
    link_to_model_transformation: Optional[str] = None
    model_notes: Optional[str] = None


class DatasetMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    input_token_average: Optional[float] = None
    output_token_average: Optional[float] = None

    @field_validator("input_token_average", "output_token_average", mode="before")
    @classmethod
    def _coerce_to_float(cls, v: object) -> object:
        if isinstance(v, str):
            return float(v)
        return v
    dataset_type: Optional[str] = None
    dataset_link: Optional[str] = None


class Accuracy(BaseModel):
    model_config = ConfigDict(extra="allow")

    measured_accuracy_score: Optional[Union[str, float]] = None

    @field_validator("measured_accuracy_score", mode="before")
    @classmethod
    def _coerce_empty_to_none(cls, v: object) -> object:
        if v == "" or v is None:
            return None
        return v


class SystemDescription(BaseModel):
    """Parsed contents of ``systems/<system_id>.json`` (§8.2).

    Mirrors the nested structure produced by ``get-mlperf-multi-node-system-info``.
    All hardware and software metadata lives inside ``system_under_test.node_types``.
    """

    model_config = ConfigDict(extra="allow")

    organization_metadata: OrganizationMetadata
    system_under_test: SystemUnderTest
    model_metadata: ModelMetadata
    dataset_metadata: Optional[DatasetMetadata] = None
    accuracy: Optional[Accuracy] = None
