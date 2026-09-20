from pathlib import Path

import pytest

from dev_app.backend.app.importers.veo.classifier import classify_resource
from dev_app.backend.app.importers.veo.models import importable
from dev_app.backend.app.main import with_camera_directions
from dev_app.backend.app.storage import WorkspaceStore


def test_constrained_camera_directions_is_importable_asset():
    candidate = classify_resource("det", "https://cdn.example/detections_constrained.det")
    assert candidate.role == "camera_directions"
    assert importable(candidate)


def test_camera_directions_role_persists_in_match_store(tmp_path: Path):
    store = WorkspaceStore(tmp_path)
    match = store.create_match("Test", "", None)
    path = tmp_path / "detections_constrained.det"
    path.write_text("0,0,1,0\n", encoding="utf-8")
    store.attach_media(match["id"], {"id": "media_det", "name": path.name, "path": str(path), "role": "camera_directions", "kind": "resource"})
    saved = store.match(match["id"])
    assert saved and saved["media"][0]["role"] == "camera_directions"


def test_missing_camera_directions_is_explicit():
    with pytest.raises(Exception) as error:
        with_camera_directions({"media": []})
    assert "camera directions missing" in str(error.value).lower()
