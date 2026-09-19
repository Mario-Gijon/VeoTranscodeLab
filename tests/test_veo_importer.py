from __future__ import annotations

from pathlib import Path
import shutil

from dev_app.backend.app import main
from dev_app.backend.app.importers.veo.classifier import classify_resource, ranked_candidates, sanitize_url
from dev_app.backend.app.importers.veo.models import VeoCandidate, VeoInspection
from dev_app.backend.app.importers.veo.service import ImportJob, VeoImportService
from dev_app.backend.app.schemas import VeoInspectRequest
from dev_app.backend.app.storage import WorkspaceStore


ROOT = Path(__file__).resolve().parents[1]


def test_url_sanitization_and_conservative_resource_classification():
    safe = sanitize_url("https://cdn.example/video.mp4?token=secret&quality=high#t=15")
    assert "secret" not in safe and "token=REDACTED" in safe and "#" not in safe
    panorama = classify_resource("one", "https://cdn.example/panorama_transcode.mp4", "video/mp4", {"width": 2048, "height": 2048})
    ball = classify_resource("two", "https://cdn.example/follow-camera-ball_tracking.mp4", "video/mp4")
    calibration = classify_resource("three", "https://cdn.example/camera_alignment.veo", "application/json")
    assert panorama.role == "panorama" and panorama.confidence >= 0.9
    assert ball.role == "ball_tracking"
    assert calibration.role == "calibration"
    assert ranked_candidates([ball, panorama], "panorama") == [panorama]


def test_public_inspection_manifest_never_exposes_full_signed_url(tmp_path):
    candidate = classify_resource("asset_1", "https://cdn.example/panorama.mp4?X-Amz-Signature=secret", "video/mp4")
    inspection = VeoInspection("inspect_test", "https://app.veo.co/match/1", "https://app.veo.co/match/1", candidates=[candidate])
    manifest = inspection.public_dict()
    assert "url" not in manifest["candidates"][0]
    assert "secret" not in str(manifest)


def test_import_job_populates_normal_workspace_from_local_fixture(tmp_path, monkeypatch):
    service = VeoImportService(WorkspaceStore(tmp_path), tmp_path)
    calibration = ROOT / "data" / "camera_alignment-67a3f0b3-002e-491c-a48c-298f6f0160d1.veo"
    candidate = VeoCandidate("cal", calibration.as_uri(), calibration.as_uri(), calibration.name, "application/json", "calibration", 1.0, ["fixture"])
    inspection = VeoInspection("inspect_fixture", "file://fixture", "file://fixture", metadata={"title": "Fixture match"}, candidates=[candidate])
    service._inspections[inspection.identifier] = inspection
    def fake_download(_url, destination, progress, cancelled, cookies):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(calibration, destination)
        progress(destination.stat().st_size, destination.stat().st_size)
    monkeypatch.setattr("dev_app.backend.app.importers.veo.service.download_stream", fake_download)
    job = ImportJob("job_fixture", inspection.identifier, {"calibration": "cal"})
    service._run_import(job)
    match = service.store.match(job.match_id or "")
    assert job.status == "completed"
    assert match is not None and match["name"] == "Fixture match"
    assert match["calibration"]["field_length"] > 0


def test_inspection_api_handler_uses_mocked_browser_result(monkeypatch):
    candidate = classify_resource("asset_mock", "https://cdn.example/camera_alignment.veo?token=secret", "application/json")
    fake = VeoInspection("inspect_mock", "https://app.veo.co/matches/x", "https://app.veo.co/matches/x", candidates=[candidate])
    monkeypatch.setattr(main.veo_imports, "inspect", lambda _url: fake)
    response = main.inspect_veo(VeoInspectRequest(match_url="https://app.veo.co/matches/x"))
    assert response["id"] == "inspect_mock"
    assert "secret" not in str(response)
