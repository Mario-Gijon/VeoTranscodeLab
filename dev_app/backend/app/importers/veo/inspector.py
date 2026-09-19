"""Network-led Playwright match inspection, with no HTML scraper dependency."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .classifier import classify_resource, sanitize_url
from .models import VeoInspection
from .downloader import public_url, authorized_opener
from urllib.request import Request


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
            local.write(data); local.flush()
            command = [executable, "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,avg_frame_rate", "-of", "json", local.name]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
        payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
        stream = next((value for value in payload.get("streams", []) if value.get("codec_type") == "video"), {})
        rate = stream.get("avg_frame_rate", "0/0")
        numerator, denominator = (float(value) for value in rate.split("/"))
        return {"width": stream.get("width"), "height": stream.get("height"), "fps": numerator / denominator if denominator else None, "duration_s": float(payload.get("format", {}).get("duration")) if payload.get("format", {}).get("duration") else None}
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return {}


def inspect_match(match_url: str, profile_dir: Path, diagnostics_dir: Path) -> VeoInspection:
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError("Playwright is unavailable. Install the backend requirements and run 'playwright install chromium'.") from exc
    target = normalized_match_url(match_url)
    public_url(target)
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.chmod(0o700)
    seen: dict[str, tuple[str, str | None]] = {}
    calibration_details = {}
    metadata = {}
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(str(profile_dir), headless=True)
        page = context.pages[0] if context.pages else context.new_page()
        def remember(response):
            url = response.url
            content = response.headers.get("content-type")
            path = urlsplit(url).path.lower()
            if response.status not in {200, 206} or len(seen) >= 100:
                return
            normalized = urlunsplit((*urlsplit(url)[:4], ""))
            if any(token in path for token in (".mp4", ".m3u8", ".mpd", ".veo", "camera_alignment")) or (content and content.startswith("video/")):
                seen.setdefault(normalized, (normalized, content))
            size = response.headers.get("content-length", "")
            if content and "json" in content and size.isdigit() and int(size) <= 2_000_000:
                try:
                    payload = response.json()
                    for link in resource_links(payload, limit=max(0, 100 - len(seen))):
                        link = urlunsplit((*urlsplit(link)[:4], ""))
                        seen.setdefault(link, (link, None))
                    # Only use explicitly named scalar match metadata, not a
                    # page title (which could just be a login/loading page).
                    if isinstance(payload, dict):
                        match_data = payload.get("match", {})
                        if isinstance(match_data, dict):
                            for key in ("title", "date", "home_team", "away_team"):
                                if isinstance(match_data.get(key), str):
                                    metadata[key] = match_data[key]
                except Exception:
                    pass
        page.on("response", remember)
        try:
            page.goto(target, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(9000)
            # The browser may not request a mode until the user activates it.
            # We do not guess private APIs or bypass unavailable player controls.
            for url, _ in list(seen.values())[:100]:
                if ".veo" in urlsplit(url).path or "camera_alignment" in urlsplit(url).path:
                    try:
                        public_url(url)
                        cookies_for_url = context.cookies([url])
                        with authorized_opener(cookies_for_url).open(Request(url), timeout=8) as response:
                            body = response.read(2_000_001)
                        if len(body) > 2_000_000:
                            continue
                        from analysis.veo_metadata import load_veo_metadata
                        with tempfile.NamedTemporaryFile(suffix=".veo") as local:
                            local.write(body); local.flush()
                            parsed = load_veo_metadata(local.name)
                        calibration_details[url] = (parsed.field_length, parsed.field_width, parsed.raw_json.get("calibration_version"))
                    except Exception:
                        pass
            cookies = {url: context.cookies([url]) for url in seen}
        finally:
            context.close()
    candidates = []
    for index, (url, content) in enumerate(seen.values()):
        candidate = classify_resource(f"asset_{index + 1}", url, content)
        if index < 8 and candidate.filename.lower().endswith(".mp4"):
            candidate = classify_resource(candidate.identifier, url, content, probe_media(url, cookies.get(url, [])))
        candidate.cookies = cookies.get(url, [])
        if url in calibration_details:
            candidate.field_length, candidate.field_width, version = calibration_details[url]
            candidate.calibration_version = str(version) if version is not None else None
        candidates.append(candidate)
    inspection = VeoInspection(identifier=f"inspect_{uuid.uuid4().hex[:12]}", match_url=target, safe_match_url=sanitize_url(target), metadata=metadata, candidates=candidates)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostic = diagnostics_dir / f"{inspection.identifier}.json"
    diagnostic.write_text(json.dumps(inspection.public_dict(), indent=2), encoding="utf-8")
    inspection.diagnostic_path = str(diagnostic)
    return inspection
