"""Network-led, phase-aware Playwright inspection of an authorized Veo match."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request

from .classifier import classify_resource, sanitize_url
from .downloader import authorized_opener, public_url
from .models import VeoCandidate, VeoInspection


INITIAL_PHASE = "initial"
AFTER_INTERACTIVE_PHASE = "after_interactive"
MAX_RESOURCES = 100
INITIAL_CAPTURE_MS = 9_000
POST_INTERACTIVE_CAPTURE_MS = 9_000
INTERACTIVE_LABEL = re.compile(r"^\s*(interactive|interactivo)\s*$", re.IGNORECASE)
PRIMARY_INTERACTIVE_SELECTOR = "button.interactive-player"
MENU_LABEL = re.compile(
    r"interactive|interactivo|view|vista|camera|cámara|mode|modo|menu|more|más|options|opciones|"
    r"standard|estándar|panoramic|panorámica|tracking|seguimiento",
    re.IGNORECASE,
)


def resource_links(value, limit=200) -> list[str]:
    """Discover actual resource URLs supplied by player JSON, without guessing IDs."""
    found = []

    def visit(item, depth=0):
        if depth > 12 or len(found) >= limit:
            return
        if isinstance(item, str) and item.startswith("https://"):
            path = urlsplit(item).path.lower()
            if any(term in path for term in (".mp4", ".veo", ".m3u8", ".mpd", "camera_alignment")):
                found.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child, depth + 1)
        elif isinstance(item, list):
            for child in item[:200]:
                visit(child, depth + 1)

    visit(value)
    return found


def normalized_match_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme != "https" or not parsed.hostname or not (parsed.hostname == "veo.co" or parsed.hostname.endswith(".veo.co")) or parsed.username or parsed.password:
        raise ValueError("Enter an authorized https://app.veo.co/ match URL.")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def probe_media(url: str, cookies: list[dict] | None = None) -> dict:
    executable = shutil.which("ffprobe")
    if executable is None:
        return {}
    try:
        public_url(url)
        # Bound inspection traffic to an 8 MiB prefix. Signed URLs never enter
        # ffprobe's command line; a trailing moov atom may make this inconclusive.
        with authorized_opener(cookies or []).open(Request(url, headers={"Range": "bytes=0-8388607"}), timeout=10) as response:
            data = response.read(8388608)
        with tempfile.NamedTemporaryFile(suffix=".mp4") as local:
            local.write(data)
            local.flush()
            command = [executable, "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,avg_frame_rate", "-of", "json", local.name]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
        payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
        stream = next((value for value in payload.get("streams", []) if value.get("codec_type") == "video"), {})
        rate = stream.get("avg_frame_rate", "0/0")
        numerator, denominator = (float(value) for value in rate.split("/"))
        return {"width": stream.get("width"), "height": stream.get("height"), "fps": numerator / denominator if denominator else None, "duration_s": float(payload.get("format", {}).get("duration")) if payload.get("format", {}).get("duration") else None}
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return {}


def wait_for_player(page) -> bool:
    """Wait briefly for the default player/video before revealing its controls."""
    try:
        videos = page.locator("video")
        if videos.count() == 0:
            return False
        videos.nth(0).wait_for(state="attached", timeout=5_000)
        page.wait_for_timeout(300)
        return True
    except Exception:
        return False


def _hover_player(page) -> bool:
    """Reveal player controls without clicking unrelated page content."""
    for selector in ("video", "[data-testid*='player']", "[class*='player']"):
        try:
            locator = page.locator(selector)
            if locator.count() and locator.nth(0).is_visible(timeout=500):
                locator.nth(0).hover(timeout=1_500)
                page.wait_for_timeout(350)
                return True
        except Exception:
            continue
    return False


def _unobstructed(button) -> bool:
    """Allow a force click only when the real button owns its center pixel."""
    try:
        box = button.bounding_box()
        if not box:
            return False
        return bool(button.evaluate(
            """(element, point) => {
                const top = document.elementFromPoint(point.x, point.y);
                return Boolean(top && (top === element || element.contains(top)));
            }""",
            {"x": box["x"] + box["width"] / 2, "y": box["y"] + box["height"] / 2},
        ))
    except Exception:
        return False


def _click_first_visible(locator, attempt: str, attempts: list[str], before_click, restore_after_failure) -> tuple[bool, bool]:
    """Click one visible, user-facing control; never manufacture a player URL."""
    try:
        count = min(locator.count(), 12)
    except Exception:
        return False, False
    found = False
    for index in range(count):
        candidate = locator.nth(index)
        previous_phase = None
        try:
            if not candidate.is_visible(timeout=500):
                continue
            found = True
            previous_phase = before_click()
            candidate.click(timeout=2_000)
            attempts.append(attempt)
            return True, True
        except Exception:
            if previous_phase is not None:
                restore_after_failure(previous_phase)
            continue
    return found, False


def _control_label(control) -> str:
    try:
        parts = [
            control.get_attribute("aria-label"),
            control.get_attribute("title"),
            control.get_attribute("data-testid"),
            control.inner_text(timeout=300),
        ]
        return " ".join(part.strip() for part in parts if part and part.strip())[:160]
    except Exception:
        return ""


def _activate_primary_interactive(page, before_click, restore_after_failure) -> dict[str, object]:
    """Use the observed Veo semantic button before any text/accessibility fallback."""
    result: dict[str, object] = {
        "primary_selector": PRIMARY_INTERACTIVE_SELECTOR,
        "primary_selector_found": False,
        "primary_selector_count": 0,
        "primary_selector_visible": False,
        "primary_selector_hover_required": False,
        "primary_selector_click_succeeded": False,
        "primary_selector_force_click_used": False,
    }
    try:
        locator = page.locator(PRIMARY_INTERACTIVE_SELECTOR)
        count = min(locator.count(), 12)
    except Exception:
        return result
    result["primary_selector_count"] = count
    result["primary_selector_found"] = count > 0
    if not count:
        return result

    def visible_buttons():
        visible = []
        for index in range(count):
            try:
                button = locator.nth(index)
                if button.is_visible(timeout=500):
                    visible.append(button)
            except Exception:
                continue
        return visible

    buttons = visible_buttons()
    if not buttons:
        result["primary_selector_hover_required"] = True
        _hover_player(page)
        buttons = visible_buttons()
    result["primary_selector_visible"] = bool(buttons)
    for button in buttons:
        previous_phase = before_click()
        try:
            button.click(timeout=2_000)
            result["primary_selector_click_succeeded"] = True
            return result
        except Exception:
            restore_after_failure(previous_phase)
        # `force=True` is constrained to an attached, visible, exact semantic
        # button whose center is not covered by another element.
        if not _unobstructed(button):
            continue
        previous_phase = before_click()
        try:
            button.click(timeout=2_000, force=True)
            result["primary_selector_click_succeeded"] = True
            result["primary_selector_force_click_used"] = True
            return result
        except Exception:
            restore_after_failure(previous_phase)
    return result


def activate_interactive(page, before_click, restore_after_failure) -> dict[str, object]:
    """Open Interactive explicitly, including accessible menu-based controls.

    Accessibility paths are intentional first choices. The menu pass only opens
    visible controls whose own text indicates a view/menu; it does not click
    arbitrary page content or guess private player endpoints.
    """
    primary = _activate_primary_interactive(page, before_click, restore_after_failure)
    attempts: list[str] = ["button.interactive-player"] if primary["primary_selector_click_succeeded"] else []
    menu_controls: list[str] = []
    found = bool(primary["primary_selector_found"])
    if primary["primary_selector_click_succeeded"]:
        return {**primary, "interactive_control_found": True, "interactive_click_succeeded": True, "interactive_attempts": attempts, "menu_controls_inspected": menu_controls}

    def direct_control() -> tuple[bool, bool]:
        found = False
        locators = (
            (page.get_by_role("button", name=INTERACTIVE_LABEL), "interactive button"),
            (page.get_by_role("tab", name=INTERACTIVE_LABEL), "interactive tab"),
            (page.get_by_role("menuitem", name=INTERACTIVE_LABEL), "interactive menu item"),
            (page.get_by_role("radio", name=INTERACTIVE_LABEL), "interactive radio"),
            (page.get_by_text(INTERACTIVE_LABEL, exact=True), "interactive visible text"),
        )
        for locator, attempt in locators:
            visible, succeeded = _click_first_visible(locator, attempt, attempts, before_click, restore_after_failure)
            found = found or visible
            if succeeded:
                return found, True
        return found, False

    fallback_found, succeeded = direct_control()
    found = found or fallback_found
    if succeeded:
        return {**primary, "interactive_control_found": True, "interactive_click_succeeded": True, "interactive_attempts": attempts, "menu_controls_inspected": menu_controls}

    try:
        controls = page.locator("button, [role='button'], [aria-haspopup='menu'], [aria-expanded]")
        count = min(controls.count(), 40)
    except Exception:
        controls, count = None, 0
    for index in range(count):
        assert controls is not None
        control = controls.nth(index)
        try:
            if not control.is_visible(timeout=250):
                continue
        except Exception:
            continue
        label = _control_label(control)
        try:
            is_menu_trigger = control.get_attribute("aria-haspopup") == "menu"
        except Exception:
            is_menu_trigger = False
        if not is_menu_trigger and (not label or not MENU_LABEL.search(label)):
            continue
        menu_controls.append(label or "unnamed menu control")
        try:
            control.click(timeout=1_500)
            page.wait_for_timeout(300)
        except Exception:
            continue
        discovered, succeeded = direct_control()
        found = found or discovered
        if succeeded:
            return {**primary, "interactive_control_found": True, "interactive_click_succeeded": True, "interactive_attempts": attempts, "menu_controls_inspected": menu_controls}
    return {**primary, "interactive_control_found": found, "interactive_click_succeeded": False, "interactive_attempts": attempts, "menu_controls_inspected": menu_controls}


def _candidate_reference(candidate: VeoCandidate) -> dict[str, object]:
    return {"identifier": candidate.identifier, "filename": candidate.filename, "safe_url": candidate.safe_url, "discovery_phases": candidate.discovery_phases}


def phase_diagnostics(candidates: list[VeoCandidate], interaction: dict[str, object]) -> dict[str, object]:
    def in_phase(phase: str) -> list[dict[str, object]]:
        return [_candidate_reference(candidate) for candidate in candidates if phase in candidate.discovery_phases]

    def final(role: str) -> list[dict[str, object]]:
        return [_candidate_reference(candidate) for candidate in candidates if candidate.role == role]

    expected_after_click = [
        candidate for candidate in candidates
        if AFTER_INTERACTIVE_PHASE in candidate.discovery_phases
        and (candidate.filename.lower().startswith("transcode-") and candidate.filename.lower().endswith(".mp4")
             or candidate.filename.lower().endswith(".veo") or "camera_alignment" in candidate.filename.lower())
    ]

    return {
        "default_phase_resources": in_phase(INITIAL_PHASE),
        "interactive_control_found": interaction["interactive_control_found"],
        "interactive_click_succeeded": interaction["interactive_click_succeeded"],
        "interactive_activation_confirmed": bool(interaction["interactive_click_succeeded"] and expected_after_click),
        "primary_selector": interaction.get("primary_selector", PRIMARY_INTERACTIVE_SELECTOR),
        "primary_selector_found": interaction.get("primary_selector_found", False),
        "primary_selector_count": interaction.get("primary_selector_count", 0),
        "primary_selector_visible": interaction.get("primary_selector_visible", False),
        "primary_selector_hover_required": interaction.get("primary_selector_hover_required", False),
        "primary_selector_click_succeeded": interaction.get("primary_selector_click_succeeded", False),
        "primary_selector_force_click_used": interaction.get("primary_selector_force_click_used", False),
        "interactive_attempts": interaction["interactive_attempts"],
        "menu_controls_inspected": interaction["menu_controls_inspected"],
        "post_interactive_resources": in_phase(AFTER_INTERACTIVE_PHASE),
        "final_classified_assets": {
            "tracking": final("ball_tracking"),
            "dual_lens": final("panorama"),
            "calibration": final("calibration"),
        },
    }


def inspect_match(match_url: str, profile_dir: Path, diagnostics_dir: Path) -> VeoInspection:
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError("Playwright is unavailable. Install the backend requirements and run 'playwright install chromium'.") from exc
    target = normalized_match_url(match_url)
    public_url(target)
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.chmod(0o700)
    seen: dict[str, dict[str, object]] = {}
    calibration_details: dict[str, tuple[float, float, object]] = {}
    metadata: dict[str, str] = {}
    phase = INITIAL_PHASE

    def record_resource(url: str, content_type: str | None, discovered_phase: str) -> None:
        normalized = urlunsplit((*urlsplit(url)[:4], ""))
        path = urlsplit(normalized).path.lower()
        if not (any(token in path for token in (".mp4", ".m3u8", ".mpd", ".veo", "camera_alignment")) or (content_type and content_type.startswith("video/"))):
            return
        if normalized not in seen and len(seen) >= MAX_RESOURCES:
            return
        resource = seen.setdefault(normalized, {"url": normalized, "content_type": content_type, "phases": []})
        if content_type and not resource["content_type"]:
            resource["content_type"] = content_type
        if discovered_phase not in resource["phases"]:
            resource["phases"].append(discovered_phase)

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(str(profile_dir), headless=True)
        page = context.pages[0] if context.pages else context.new_page()

        def remember(response):
            content = response.headers.get("content-type")
            if response.status not in {200, 206}:
                return
            record_resource(response.url, content, phase)
            size = response.headers.get("content-length", "")
            if content and "json" in content and size.isdigit() and int(size) <= 2_000_000:
                try:
                    payload = response.json()
                    for link in resource_links(payload, limit=MAX_RESOURCES):
                        record_resource(link, None, phase)
                    if isinstance(payload, dict) and isinstance(payload.get("match"), dict):
                        for key in ("title", "date", "home_team", "away_team"):
                            if isinstance(payload["match"].get(key), str):
                                metadata[key] = payload["match"][key]
                except Exception:
                    pass

        page.on("response", remember)  # Attached before page/player initialization.
        interaction: dict[str, object] = {
            "interactive_control_found": False,
            "interactive_click_succeeded": False,
            "interactive_attempts": [],
            "menu_controls_inspected": [],
        }
        try:
            page.goto(target, wait_until="domcontentloaded", timeout=30_000)
            player_ready = wait_for_player(page)
            page.wait_for_timeout(INITIAL_CAPTURE_MS)

            def start_interactive_phase() -> str:
                nonlocal phase
                previous_phase = phase
                phase = AFTER_INTERACTIVE_PHASE
                return previous_phase

            def restore_phase(previous_phase: str) -> None:
                nonlocal phase
                phase = previous_phase

            interaction = activate_interactive(page, start_interactive_phase, restore_phase)
            interaction["player_ready"] = player_ready
            if interaction["interactive_click_succeeded"]:
                page.wait_for_timeout(POST_INTERACTIVE_CAPTURE_MS)

            for resource in list(seen.values()):
                url = str(resource["url"])
                path = urlsplit(url).path.lower()
                if ".veo" not in path and "camera_alignment" not in path:
                    continue
                try:
                    public_url(url)
                    cookies_for_url = context.cookies([url])
                    with authorized_opener(cookies_for_url).open(Request(url), timeout=8) as response:
                        body = response.read(2_000_001)
                    if len(body) > 2_000_000:
                        continue
                    from analysis.veo_metadata import load_veo_metadata
                    with tempfile.NamedTemporaryFile(suffix=".veo") as local:
                        local.write(body)
                        local.flush()
                        parsed = load_veo_metadata(local.name)
                    calibration_details[url] = (parsed.field_length, parsed.field_width, parsed.raw_json.get("calibration_version"))
                except Exception:
                    pass
            cookies = {str(resource["url"]): context.cookies([str(resource["url"])]) for resource in seen.values()}
        finally:
            context.close()

    entries: list[tuple[dict[str, object], VeoCandidate]] = []
    for index, resource in enumerate(seen.values()):
        url = str(resource["url"])
        entries.append((resource, classify_resource(f"asset_{index + 1}", url, resource["content_type"], discovery_phases=list(resource["phases"]))))

    # Spend bounded media probes on the observed default/Interactive resources
    # before unrelated responses. Classification remains phase-aware either way.
    def probe_priority(entry: tuple[dict[str, object], VeoCandidate]) -> tuple[int, str]:
        _, candidate = entry
        initial_video = INITIAL_PHASE in candidate.discovery_phases and candidate.filename.lower() == "video.mp4"
        interactive_transcode = AFTER_INTERACTIVE_PHASE in candidate.discovery_phases and candidate.filename.lower().startswith("transcode-")
        return (0 if initial_video else 1 if interactive_transcode else 2, candidate.filename)

    indices = sorted((index for index, entry in enumerate(entries) if entry[1].filename.lower().endswith(".mp4")), key=lambda index: probe_priority(entries[index]))[:8]
    for index in indices:
        resource, candidate = entries[index]
        url = str(resource["url"])
        entries[index] = (resource, classify_resource(candidate.identifier, url, resource["content_type"], probe_media(url, cookies.get(url, [])), candidate.discovery_phases))

    candidates = []
    for resource, candidate in entries:
        url = str(resource["url"])
        candidate.cookies = cookies.get(url, [])
        if url in calibration_details:
            candidate.field_length, candidate.field_width, version = calibration_details[url]
            candidate.calibration_version = str(version) if version is not None else None
        candidates.append(candidate)
    inspection = VeoInspection(
        identifier=f"inspect_{uuid.uuid4().hex[:12]}",
        match_url=target,
        safe_match_url=sanitize_url(target),
        metadata=metadata,
        candidates=candidates,
        interaction=phase_diagnostics(candidates, interaction),
    )
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostic = diagnostics_dir / f"{inspection.identifier}.json"
    diagnostic.write_text(json.dumps(inspection.public_dict(), indent=2), encoding="utf-8")
    inspection.diagnostic_path = str(diagnostic)
    return inspection
