"""Contracts for future analysis artifacts. No runner, scheduler or CV models.

Persist a run manifest under matches/<id>/analysis/<run-id>/run.json using
the existing WorkspaceStore.asset_dir(id, 'analysis'). Stages exchange artifact
types and paths, never reach into another algorithm's internal representation.
"""
from typing import Any, Literal
from pydantic import BaseModel, Field


class StageDefinition(BaseModel):
    stage_id: str
    version: str
    requires: list[str]
    produces: list[str]
    configuration: dict[str, Any] = Field(default_factory=dict)


class AnalysisArtifact(BaseModel):
    artifact_id: str
    artifact_type: str
    match_id: str
    source_media_id: str
    run_id: str
    stage_id: str
    stage_version: str
    relative_path: str


class StageRun(BaseModel):
    definition: StageDefinition
    status: Literal['pending', 'running', 'complete', 'failed', 'cancelled'] = 'pending'
    artifacts: list[AnalysisArtifact] = Field(default_factory=list)


class AnalysisRun(BaseModel):
    run_id: str
    match_id: str
    source_media_id: str
    stages: list[StageRun] = Field(default_factory=list)


# Illustrative dependency contracts only; nothing executes or produces data.
STAGES = (
    StageDefinition(stage_id='person_detection', version='1', requires=['media.video'], produces=['detections.person']),
    StageDefinition(stage_id='person_tracking', version='1', requires=['detections.person'], produces=['tracks.person']),
    StageDefinition(stage_id='team_referee_classification', version='1', requires=['tracks.person'], produces=['classifications.team_role']),
)
