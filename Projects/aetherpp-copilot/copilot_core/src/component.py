from enum import Enum

from typing import List
from pydantic import BaseModel


class ComponentType(str, Enum):
    CommandComponent = "CommandComponent"
    CommandComponentPreview = "CommandComponent@1-preview"
    DataTransferComponent = "DataTransferComponent"
    DistributedComponent = "DistributedComponent"
    HDInsightComponent = "HDInsightComponent"
    SparkComponent = "spark"
    ParallelComponent = "ParallelComponent"
    ScopeComponent = "ScopeComponent"
    StarliteComponent = "StarliteComponent"
    SweepComponent = "SweepComponent"
    HemeraComponent = "HemeraComponent"
    AE365ExePoolComponent = "AE365ExePoolComponent"
    AetherBridgeComponent = "AetherBridgeComponent"



class Component(BaseModel):
    name: str
    display_name: str
    description: str
    component_type: str
    path: str
    inputs: List[str] = []
