from dev_app.backend.app.importers.veo.discovery import StructuredDiscovery, referenced_urls
from dev_app.backend.app.importers.veo.inspector import INITIAL_PHASE, camera_directions_for_tracking, standard_video_camera_directions


def test_referenced_urls_and_camera_directions_are_discovered_from_json():
    tracking = "https://c.veocdn.com/m/standard/machine/a/video.mp4?sig=private"
    directions = "https://c.veocdn.com/m/standard/machine/a/detections_constrained.det?sig=private"
    payload = {"videos": [
        {"url": tracking, "render_type": "standard", "camera_directions": directions},
        {"url": "https://c.veocdn.com/m/other/video.mp4", "render_type": "standard", "camera_directions": None},
    ]}
    pairs = standard_video_camera_directions(payload)
    assert pairs == [(tracking, directions)]
    resources = {tracking.split("?")[0]: {"url": tracking, "phases": [INITIAL_PHASE]}}
    assert camera_directions_for_tracking(resources, [(tracking, directions, INITIAL_PHASE)]) == (tracking, directions)
    refs = referenced_urls(payload)
    assert any(item["field"].endswith("camera_directions") and "private" not in item["url"] for item in refs)


def test_match_dataset_evidence_stays_separate_from_platform_capabilities_and_unknowns():
    discovery = StructuredDiscovery()
    discovery.observe_json("https://api.veo.co/matches/1/events?token=private", {
        "match": {"title": "Fixture"}, "events": [{"type": "goal"}], "analytics_available": True,
    }, "after_interactive")
    manifest = discovery.manifest({"title": "Fixture"}, {}, {"source": "test"})
    assert manifest["datasets"]["events"]["status"] == "observed"
    assert manifest["datasets"]["player_tracking"]["status"] == "not_observed"
    assert manifest["capabilities"]["analytics_available"]["status"] == "observed"
    assert "private" not in str(manifest)


def test_unknown_match_json_is_bounded_and_sanitized():
    discovery = StructuredDiscovery()
    discovery.observe_json("https://api.veo.co/matches/1?signature=private", {"match": {"id": "m1"}, "custom_blob": "x" * 60_000}, "initial")
    manifest = discovery.manifest({}, {}, {})
    assert len(manifest["unknown_structured_data"]) == 1
    assert "private" not in str(manifest)
