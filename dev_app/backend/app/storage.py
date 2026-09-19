"""Small JSON-backed workspace store for the local-only developer app."""

from __future__ import annotations

import json
import threading
import uuid
import re
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
        if not self.state_path.exists():
            self._write({"matches": []})
        # One-time metadata migration. Source assets and the legacy manifest remain
        # untouched so older workspaces are recoverable without copying videos.
        marker = runtime_root / "matches" / ".migrated"
        if not marker.exists():
            for match in self._read()["matches"]:
                path = self.match_dir(match["id"]) / "match.json"
                if not path.exists():
                    self._save_match(match)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()

    def match_dir(self, match_id: str) -> Path:
        if not re.fullmatch(r"match_[a-zA-Z0-9_-]+", match_id):
            raise ValueError("Invalid match identifier")
        directory = self.runtime_root / "matches" / match_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def asset_dir(self, match_id: str, category: str) -> Path:
        if category not in {"media", "frames", "calibration", "cache", "diagnostics"}:
            raise ValueError("Invalid asset category")
        path = self.match_dir(match_id) / category
        path.mkdir(exist_ok=True)
        return path

    def _save_match(self, match: dict) -> None:
        path = self.match_dir(match["id"]) / "match.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(match, indent=2), encoding="utf-8")
        temporary.replace(path)

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
            return [json.loads(path.read_text()) for path in sorted((self.runtime_root / "matches").glob("match_*/match.json"))]

    def match(self, match_id: str) -> dict[str, Any] | None:
        return next((item for item in self.matches() if item["id"] == match_id), None)

    def create_match(self, name: str, description: str, calibration: dict[str, Any] | None) -> dict[str, Any]:
        with self._lock:
            item = {
                "id": self.new_id("match"),
                "name": name,
                "description": description,
                "created_at": self.now(),
                "updated_at": self.now(),
                "calibration": calibration,
                "media": [],
                "metadata": {},
                "artifacts": [],
            }
            self._save_match(item)
            return item

    def update_match(self, match_id: str, callback) -> dict[str, Any] | None:
        with self._lock:
            item = self.match(match_id)
            if item is None:
                return None
            callback(item)
            item["updated_at"] = self.now()
            self._save_match(item)
            return item

    def media(self, media_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        for match in self.matches():
            item = next((value for value in match["media"] if value["id"] == media_id), None)
            if item is not None:
                return match, item
        return None
