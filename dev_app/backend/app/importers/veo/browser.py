"""Dedicated Playwright profile launcher; it never reads a normal browser profile."""

from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path


def playwright_available() -> bool:
    try:
        import playwright.sync_api  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def connect(profile_dir: Path) -> subprocess.Popen:
    """Open a normal headed browser whose cookies stay only in runtime/veo-profile."""
    if not playwright_available():
        raise RuntimeError("Playwright is not installed. Install the backend requirements and Chromium first.")
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.chmod(0o700)
    return subprocess.Popen([sys.executable, "-m", "dev_app.backend.app.importers.veo.browser", str(profile_dir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    import signal
    from playwright.sync_api import sync_playwright
    stopping = False
    def stop(*_):
        global stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    status_file = Path(sys.argv[1]) / "connect-status.json"
    try:
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(sys.argv[1], headless=False)
            page = context.pages[0] if context.pages else context.new_page()
            status_file.write_text(json.dumps({"error": None}))
            try:
                page.goto("https://app.veo.co/", wait_until="domcontentloaded")
                while not stopping and context.pages:
                    try:
                        context.pages[0].wait_for_timeout(250)
                    except Exception:
                        break  # Closing the last user window is normal.
            finally:
                context.close()
    except Exception:
        status_file.write_text(json.dumps({"error": "Veo browser could not start or navigate. Check Chromium installation, desktop display, network, and profile lock."}))
