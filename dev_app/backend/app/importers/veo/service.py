"""Single-process Veo inspection registry and cancellable import job manager."""

from __future__ import annotations

import shutil
import threading
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ...services import calibration_summary, inspect_media, validate_media_role
from ...storage import WorkspaceStore
from .browser import connect, playwright_available
from .downloader import download_stream
from .inspector import inspect_match
from .models import VeoInspection


@dataclass
class ImportJob:
    identifier: str
    inspection_id: str
    selected_assets: dict[str, str]
    status: str = "queued"
    current_asset: str | None = None
    bytes_downloaded: int = 0
    bytes_total: int | None = None
    error: str | None = None
    match_id: str | None = None
    completed_assets: int = 0
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)

    def public_dict(self) -> dict[str, Any]:
        value = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "cancelled"}
        count = max(len(self.selected_assets), 1)
        fraction = self.bytes_downloaded / self.bytes_total if self.bytes_total else 0
        value["progress"] = min(1., (self.completed_assets + fraction) / count) if self.status != "completed" else 1.
        return value


class VeoImportService:
    def __init__(self, store: WorkspaceStore, runtime_root: Path) -> None:
        self.store = store
        self.runtime_root = runtime_root
        self.profile_dir = runtime_root / "veo-profile"
        self.diagnostics_dir = runtime_root / "veo-inspections"
        self._inspections: dict[str, VeoInspection] = {}
        self._jobs: dict[str, ImportJob] = {}
        self._connect_process = None
        self._lock = threading.RLock()

    def session(self) -> dict[str, Any]:
        status_path = self.profile_dir / "connect-status.json"
        try:
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
        except (OSError, ValueError):
            status = {}
        return {"playwright_available": playwright_available(), "profile_exists": self.profile_dir.exists(), "connect_browser_running": bool(self._connect_process and self._connect_process.poll() is None), "error": status.get("error")}

    def connect(self) -> dict[str, Any]:
        with self._lock:
            if self._connect_process is None or self._connect_process.poll() is not None:
                self._connect_process = connect(self.profile_dir)
        return self.session()

    def disconnect(self) -> dict[str, Any]:
        with self._lock:
            if any(job.status in {"queued", "downloading"} for job in self._jobs.values()):
                raise ValueError("Cancel or finish the active import before disconnecting.")
            if self._connect_process and self._connect_process.poll() is None:
                self._connect_process.terminate()
                self._connect_process.wait(timeout=15)
            self._connect_process = None
            if self.profile_dir.exists():
                shutil.rmtree(self.profile_dir)
            self._inspections.clear()
        return self.session()

    def inspect(self, match_url: str) -> VeoInspection:
        with self._lock:
            if self.session()["connect_browser_running"]:
                raise ValueError("Close the Connect Veo browser after login, then inspect the match.")
            inspection = inspect_match(match_url, self.profile_dir, self.diagnostics_dir)
            self._inspections[inspection.identifier] = inspection
        return inspection

    def start_import(self, inspection_id: str, selected_assets: dict[str, str]) -> ImportJob:
        with self._lock:
            if inspection_id not in self._inspections:
                raise ValueError("Inspection is no longer available. Inspect the match again.")
            if not selected_assets or any(role not in {"panorama", "ball_tracking", "calibration"} for role in selected_assets):
                raise ValueError("Select at least one supported asset role.")
            if len(set(selected_assets.values())) != len(selected_assets):
                raise ValueError("A resource can only have one role in an import.")
            candidates = {candidate.identifier: candidate for candidate in self._inspections[inspection_id].candidates}
            for role, identifier in selected_assets.items():
                candidate = candidates.get(identifier)
                if candidate is None or candidate.role == "manifest":
                    raise ValueError("Select a discovered file, not a streaming manifest.")
                if (role == "calibration") != (candidate.role == "calibration"):
                    raise ValueError("Calibration and video roles cannot be interchanged.")
            if any(job.status in {"queued", "downloading"} for job in self._jobs.values()):
                raise ValueError("An import is already active. Wait or cancel it first.")
            job = ImportJob(self.store.new_id("veo_job"), inspection_id, selected_assets)
            self._jobs[job.identifier] = job
        threading.Thread(target=self._run_import, args=(job,), daemon=True).start()
        return job

    def job(self, job_id: str) -> ImportJob | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> ImportJob | None:
        job = self.job(job_id)
        if job and job.status in {"queued", "downloading"}:
            job.cancelled.set()
        return job

    def _run_import(self, job: ImportJob) -> None:
        try:
            if job.cancelled.is_set():
                raise InterruptedError()
            inspection = self._inspections[job.inspection_id]
            selected = {role: next((item for item in inspection.candidates if item.identifier == candidate_id), None) for role, candidate_id in job.selected_assets.items()}
            if any(candidate is None for candidate in selected.values()):
                raise ValueError("One or more selected assets are not part of this inspection.")
            match = self.store.create_match(inspection.metadata.get("title") or "Imported Veo match", f"Imported from {inspection.safe_match_url}", None)
            job.match_id = match["id"]
            self.store.update_match(match["id"], lambda item: item.update(metadata=inspection.metadata))
            diagnostic = self.store.asset_dir(match["id"], "diagnostics") / "veo-inspection.json"
            diagnostic.write_text(json.dumps(inspection.public_dict(), indent=2), encoding="utf-8")
            for role, candidate in selected.items():
                if job.cancelled.is_set():
                    raise InterruptedError("Import cancelled.")
                assert candidate is not None
                job.status, job.current_asset, job.bytes_downloaded, job.bytes_total = "downloading", candidate.filename, 0, None
                target_dir = self.store.asset_dir(match["id"], "calibration" if role == "calibration" else "media")
                filename = Path(candidate.filename).name
                if filename in {"", ".", ".."}:
                    raise ValueError("Invalid asset filename")
                destination = target_dir / f"{role}_{filename}"
                download_stream(candidate.url, destination, lambda current, total: self._progress(job, current, total), job.cancelled.is_set, candidate.cookies)
                if job.cancelled.is_set():
                    raise InterruptedError()
                if role == "calibration":
                    calibration = calibration_summary(destination, "veo_import")
                    self.store.update_match(match["id"], lambda item: item.update(calibration=calibration))
                else:
                    details = inspect_media(destination, "video")
                    validate_media_role(details, "dual_lens" if role == "panorama" else "tracking")
                    record = {"id": self.store.new_id("media"), "name": candidate.filename, "path": str(destination), "source": "veo_import", "role": "dual_lens" if role == "panorama" else "tracking", "size_bytes": destination.stat().st_size, **details}
                    self.store.update_match(match["id"], lambda item: item["media"].append(record))
                job.completed_assets += 1
                job.bytes_downloaded, job.bytes_total = 0, None
            job.status, job.current_asset = "completed", None
        except InterruptedError:
            job.status, job.current_asset = "cancelled", None
        except Exception:
            # Network exception strings often contain full signed URLs.
            job.status, job.current_asset, job.error = "failed", None, "Import failed: resource unavailable, access expired, or invalid media. Completed files remain in the match; inspect again before retrying."

    @staticmethod
    def _progress(job: ImportJob, current: int, total: int | None) -> None:
        job.bytes_downloaded, job.bytes_total = current, total
