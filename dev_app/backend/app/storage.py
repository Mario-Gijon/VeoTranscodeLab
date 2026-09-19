"""Small JSON-backed workspace store for the local-only developer app."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class WorkspaceStore:
    """Persist workspace metadata, while media itself stays in referenced files."""

    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = runtime_root
        self.state_path = runtime_root / "workspaces.json"
        self._lock = threading.RLock()
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        (runtime_root / "media").mkdir(exist_ok=True)
        (runtime_root / "frames").mkdir(exist_ok=True)
        (runtime_root / "cache").mkdir(exist_ok=True)
        if not self.state_path.exists():
            self._write({"matches": []})

    @staticmethod
    def now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def new_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def _read(self) -> dict[str, Any]:
        with self.state_path.open("r", encoding="utf-8") as source:
            return json.load(source)

    def _write(self, state: dict[str, Any]) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as destination:
            json.dump(state, destination, indent=2, sort_keys=True)
        temporary.replace(self.state_path)

    def matches(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._read()["matches"]

    def match(self, match_id: str) -> dict[str, Any] | None:
        return next((item for item in self.matches() if item["id"] == match_id), None)

    def create_match(self, name: str, description: str, calibration: dict[str, Any] | None) -> dict[str, Any]:
        with self._lock:
            state = self._read()
            item = {
                "id": self.new_id("match"),
                "name": name,
                "description": description,
                "created_at": self.now(),
                "updated_at": self.now(),
                "calibration": calibration,
                "media": [],
            }
            state["matches"].append(item)
            self._write(state)
            return item

    def update_match(self, match_id: str, callback) -> dict[str, Any] | None:
        with self._lock:
            state = self._read()
            item = next((value for value in state["matches"] if value["id"] == match_id), None)
            if item is None:
                return None
            callback(item)
            item["updated_at"] = self.now()
            self._write(state)
            return item

    def media(self, media_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        for match in self.matches():
            item = next((value for value in match["media"] if value["id"] == media_id), None)
            if item is not None:
                return match, item
        return None
