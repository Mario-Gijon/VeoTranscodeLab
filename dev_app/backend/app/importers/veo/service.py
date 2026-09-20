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
from .models import VeoInspection, importable


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
    assets: dict[str, dict] = field(default_factory=dict)
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)

    def public_dict(self) -> dict[str, Any]:
        value = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "cancelled"}
        assets = list(self.assets.values())
        known = assets and all(asset.get("bytes_total") for asset in assets)
        value["progress"] = min(1., sum(asset["bytes_downloaded"] for asset in assets) / sum(asset["bytes_total"] for asset in assets)) if known else None
        if self.status == "completed":
            value["progress"] = 1.
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

    def active_for_match(self, match_id: str) -> bool:
        return any(job.match_id == match_id and job.status in {"queued", "downloading"} for job in self._jobs.values())

    def job_for_match(self, match_id: str) -> ImportJob | None:
        return next((job for job in reversed(list(self._jobs.values())) if job.match_id == match_id), None)

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

    def start_import(self, inspection_id: str, selected_assets: dict[str, str], match_id: str | None = None) -> ImportJob:
        with self._lock:
            if inspection_id not in self._inspections:
                raise ValueError("Inspection is no longer available. Inspect the match again.")
            if set(selected_assets) != {"panorama", "ball_tracking", "calibration"}:
                raise ValueError("Tracking, Dual lens and Calibration must all be resolved before importing.")
            if len(set(selected_assets.values())) != len(selected_assets):
                raise ValueError("A resource can only have one role in an import.")
            candidates = {candidate.identifier: candidate for candidate in self._inspections[inspection_id].candidates}
            for role, identifier in selected_assets.items():
                candidate = candidates.get(identifier)
                if candidate is None or candidate.role != role or not importable(candidate):
                    raise ValueError("Select a validated file for each required role. Retry inspection if a resource is missing.")
            if any(job.status in {"queued", "downloading"} for job in self._jobs.values()):
                raise ValueError("An import is already active. Wait or cancel it first.")
            job = ImportJob(self.store.new_id("veo_job"), inspection_id, selected_assets)
            inspection = self._inspections[inspection_id]
            if match_id:
                match = self.store.match(match_id)
                if not match or match.get("metadata", {}).get("veo_url") != inspection.safe_match_url:
                    raise ValueError("Retry inspection with the original Veo match URL.")
            else:
                match = self.store.create_match(inspection.metadata.get("title") or "Imported Veo match", "", None)
                self.store.update_match(match["id"], lambda item: item.update(metadata={**inspection.metadata, "veo_url": inspection.safe_match_url}))
            job.match_id = match["id"]
            job.assets = {role: {"status": "waiting", "bytes_downloaded": 0, "bytes_total": candidates[identifier].size_bytes, "filename": candidates[identifier].filename} for role, identifier in selected_assets.items()}
            self.store.update_match(job.match_id, lambda item: item.update(import_status="queued"))
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
            match = self.store.match(job.match_id) if job.match_id else self.store.create_match(inspection.metadata.get("title") or "Imported Veo match", "", None)
            if match is None:
                raise ValueError("Match no longer exists")
            job.match_id = match["id"]
            self.store.update_match(match["id"], lambda item: item.update(import_status="downloading"))
            diagnostic = self.store.asset_dir(match["id"], "diagnostics") / "veo-inspection.json"
            diagnostic.write_text(json.dumps(inspection.public_dict(), indent=2), encoding="utf-8")
            for role, candidate in selected.items():
                if job.cancelled.is_set():
                    raise InterruptedError("Import cancelled.")
                assert candidate is not None
                asset = job.assets.setdefault(role, {"status": "waiting", "bytes_downloaded": 0, "bytes_total": None, "filename": candidate.filename})
                media_role = "dual_lens" if role == "panorama" else "tracking"
                existing = match.get("calibration") if role == "calibration" else next((item for item in match["media"] if item.get("role") == media_role), None)
                if existing and Path(existing["path"]).is_file():
                    if role == "calibration":
                        calibration_summary(Path(existing["path"]), existing.get("source", "path"))
                    else:
                        validate_media_role(inspect_media(Path(existing["path"]), "video"), media_role)
                    size = Path(existing["path"]).stat().st_size
                    asset.update(status="complete", bytes_downloaded=size, bytes_total=size)
                    job.completed_assets += 1
                    continue
                job.status, job.current_asset, job.bytes_downloaded, job.bytes_total = "downloading", candidate.filename, 0, None
                asset["status"] = "downloading"
                target_dir = self.store.asset_dir(match["id"], "calibration" if role == "calibration" else "media")
                filename = Path(candidate.filename).name
                if filename in {"", ".", ".."}:
                    raise ValueError("Invalid asset filename")
                destination = target_dir / ("camera_alignment.veo" if role == "calibration" else f"{media_role}.mp4")
                # Completed bytes are validated before publication to Match metadata.
                # A unique staging name also avoids overwriting interrupted runs.
                staging = destination.with_name(f"{destination.stem}-{self.store.new_id('download')}{destination.suffix}")
                try:
                    download_stream(candidate.url, staging, lambda current, total: self._progress(job, current, total, role), job.cancelled.is_set, candidate.cookies)
                    if role == "calibration":
                        calibration = calibration_summary(staging, "veo_import")
                    else:
                        details = inspect_media(staging, "video")
                        validate_media_role(details, media_role)
                    staging.replace(destination)
                finally:
                    staging.unlink(missing_ok=True)
                if role == "calibration":
                    calibration.update(path=str(destination), id=str(destination), name=candidate.filename)
                    self.store.set_calibration(match["id"], calibration)
                else:
                    record = {"id": self.store.new_id("media"), "name": candidate.filename, "path": str(destination), "source": "veo_import", "ownership": "managed", "role": media_role, "size_bytes": destination.stat().st_size, **details}
                    self.store.attach_media(match["id"], record)
                size = destination.stat().st_size
                asset.update(status="complete", bytes_downloaded=size, bytes_total=size)
                job.completed_assets += 1
                job.bytes_downloaded, job.bytes_total = 0, None
            job.status, job.current_asset = "completed", None
        except InterruptedError:
            job.status, job.current_asset = "cancelled", None
        except Exception:
            # Network exception strings often contain full signed URLs.
            job.status, job.current_asset, job.error = "failed", None, "Import failed: resource unavailable, access expired, or invalid media. Completed files remain in the match; inspect again before retrying."
        finally:
            for asset in job.assets.values():
                if asset["status"] == "downloading":
                    asset["status"] = job.status
                elif asset["status"] == "waiting" and job.status == "cancelled":
                    asset["status"] = "cancelled"
            if job.match_id:
                self.store.update_match(job.match_id, lambda item: item.update(import_status=job.status, import_progress=job.public_dict()))

    @staticmethod
    def _progress(job: ImportJob, current: int, total: int | None, role: str | None = None) -> None:
        job.bytes_downloaded, job.bytes_total = current, total
        if role:
            job.assets[role].update(bytes_downloaded=current, bytes_total=total)
