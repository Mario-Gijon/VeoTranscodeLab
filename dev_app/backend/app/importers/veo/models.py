"""Data models for sanitized inspection output and in-memory imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VeoCandidate:
    identifier: str
    url: str
    safe_url: str
    filename: str
    content_type: str | None
    role: str
    confidence: float
    reasons: list[str]
    discovery_phases: list[str] = field(default_factory=list)
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_s: float | None = None
    field_length: float | None = None
    field_width: float | None = None
    calibration_version: str | None = None
    cookies: list[dict] = field(default_factory=list, repr=False)

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("url")
        value.pop("cookies")
        return value


@dataclass
class VeoInspection:
    identifier: str
    match_url: str
    safe_match_url: str
    metadata: dict[str, Any] = field(default_factory=dict)
    candidates: list[VeoCandidate] = field(default_factory=list)
    interaction: dict[str, Any] = field(default_factory=dict)
    diagnostic_path: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "match_url": self.safe_match_url,
            "metadata": self.metadata,
            "candidates": [candidate.public_dict() for candidate in self.candidates],
            "interaction": self.interaction,
            "diagnostic_path": self.diagnostic_path,
        }
