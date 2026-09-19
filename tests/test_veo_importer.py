from __future__ import annotations

from pathlib import Path
import shutil

from dev_app.backend.app import main
from dev_app.backend.app.importers.veo.classifier import classify_resource, ranked_candidates, sanitize_url
from dev_app.backend.app.importers.veo.inspector import AFTER_INTERACTIVE_PHASE, INITIAL_PHASE, activate_interactive, phase_diagnostics
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


def test_default_and_interactive_discovery_phases_drive_classification_and_diagnostics():
    tracking = classify_resource("tracking", "https://cdn.example/video.mp4?token=private", "video/mp4", discovery_phases=[INITIAL_PHASE])
    dual_lens = classify_resource("dual", "https://cdn.example/transcode-123.mp4?token=private", "video/mp4", {"width": 2048, "height": 2048}, [AFTER_INTERACTIVE_PHASE])
    calibration = classify_resource("calibration", "https://cdn.example/camera_alignment-123.veo?token=private", "application/json", discovery_phases=[AFTER_INTERACTIVE_PHASE])
    assert tracking.role == "ball_tracking" and tracking.confidence >= .9
    assert "initial/default" in " ".join(tracking.reasons)
    assert dual_lens.role == "panorama" and dual_lens.confidence >= .9
    assert "after Interactive" in " ".join(dual_lens.reasons)
    assert calibration.role == "calibration"
    diagnostics = phase_diagnostics([tracking, dual_lens, calibration], {
        "interactive_control_found": True,
        "interactive_click_succeeded": True,
        "interactive_attempts": ["interactive button"],
        "menu_controls_inspected": [],
    })
    assert [item["identifier"] for item in diagnostics["default_phase_resources"]] == ["tracking"]
    assert [item["identifier"] for item in diagnostics["post_interactive_resources"]] == ["dual", "calibration"]
    assert [item["identifier"] for item in diagnostics["final_classified_assets"]["tracking"]] == ["tracking"]
    assert [item["identifier"] for item in diagnostics["final_classified_assets"]["dual_lens"]] == ["dual"]
    assert [item["identifier"] for item in diagnostics["final_classified_assets"]["calibration"]] == ["calibration"]
    assert "private" not in str(diagnostics)


def test_interactive_activation_prefers_accessibility_queries_and_supports_spanish_label():
    class Control:
        def is_visible(self, **_kwargs):
            return True
        def click(self, **_kwargs):
            clicks.append("clicked")

    class EmptyLocator:
        def count(self):
            return 0
        def nth(self, _index):
            raise AssertionError("No element should be selected")

    class OneLocator:
        def count(self):
            return 1
        def nth(self, _index):
            return Control()

    class Page:
        def get_by_role(self, role, **kwargs):
            calls.append((role, kwargs["name"]))
            return OneLocator() if role == "button" else EmptyLocator()
        def get_by_text(self, *_args, **_kwargs):
            return EmptyLocator()

    calls, clicks, phases = [], [], []
    result = activate_interactive(Page(), lambda: phases.append(AFTER_INTERACTIVE_PHASE) or INITIAL_PHASE, phases.append)
    assert result["interactive_control_found"] is True
    assert result["interactive_click_succeeded"] is True
    assert clicks == ["clicked"] and phases == [AFTER_INTERACTIVE_PHASE]
    assert calls[0][0] == "button" and calls[0][1].search("Interactivo")


def test_interactive_diagnostic_distinguishes_found_control_from_failed_click():
    class Control:
        def is_visible(self, **_kwargs):
            return True
        def click(self, **_kwargs):
            raise RuntimeError("disabled in fixture")

    class Locator:
        def count(self):
            return 1
        def nth(self, _index):
            return Control()

    class Page:
        def get_by_role(self, *_args, **_kwargs):
            return Locator()
        def get_by_text(self, *_args, **_kwargs):
            return Locator()
        def locator(self, *_args, **_kwargs):
            return Locator()

    phase = [INITIAL_PHASE]
    result = activate_interactive(
        Page(),
        lambda: phase.append(AFTER_INTERACTIVE_PHASE) or INITIAL_PHASE,
        lambda prior: phase.__setitem__(0, prior),
    )
    assert result["interactive_control_found"] is True
    assert result["interactive_click_succeeded"] is False
    assert phase[0] == INITIAL_PHASE


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
