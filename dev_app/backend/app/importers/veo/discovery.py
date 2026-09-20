"""Shared, bounded interpretation of Veo match discovery evidence.

Browser clients collect page-specific evidence; this module turns that evidence
into a safe manifest without treating platform code as match data.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .classifier import sanitize_url


MAX_UNKNOWN_ITEMS = 20
MAX_UNKNOWN_BYTES = 48_000
MAX_REFERENCES = 300
SECRET_KEYS = ("token", "signature", "authorization", "cookie", "password", "secret", "credential")
DATASET_KEYS = {
    "highlights": ("highlights", "ai_highlights"),
    "events": ("events", "annotations", "timeline"),
    "lineup": ("lineup", "lineups"),
    "roster": ("roster", "players"),
    "player_tracking": ("player_tracking", "playertracking", "tracks", "track_ids"),
    "player_moments": ("player_moments", "playermoments"),
    "heatmaps": ("heatmaps", "heatmap"),
    "physical_metrics": ("physical_metrics", "physical_stats", "physicalmetrics"),
    "match_report": ("match_report", "matchreport"),
}
CAPABILITY_KEYS = {
    "has_tracking_data": ("has_tracking_data", "tracking_available", "player_tracking"),
    "analytics_available": ("analytics", "physical_metrics", "heatmaps"),
    "events_available": ("events", "highlights", "annotations"),
    "player_moments_available": ("player_moments", "playermoments"),
    "interactive_available": ("interactive", "radar", "panorama"),
}
MATCH_KEYS = ("id", "match_id", "title", "date", "start_time", "duration", "sport", "home_team", "away_team", "score", "periods", "field_length", "field_width")


def _safe_value(value: Any, key: str | None = None) -> Any:
    if key and any(secret in key.lower() for secret in SECRET_KEYS):
        return "<REDACTED>"
    if isinstance(value, dict):
        return {str(name): _safe_value(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_safe_value(item) for item in value[:200]]
    if isinstance(value, str) and value.startswith(("https://", "http://")):
        return sanitize_url(value)
    return value


def _compact(value: Any) -> Any:
    safe = _safe_value(value)
    text = repr(safe)
    if len(text) <= MAX_UNKNOWN_BYTES:
        return safe
    return {"truncated": True, "preview": text[:MAX_UNKNOWN_BYTES]}


def referenced_urls(value: Any, limit: int = MAX_REFERENCES) -> list[dict[str, str]]:
    """Find URL fields in JSON, preserving their field path and no credentials."""
    found: list[dict[str, str]] = []

    def visit(item: Any, path: str, depth: int) -> None:
        if depth > 14 or len(found) >= limit:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                child_path = f"{path}.{key}" if path else str(key)
                if isinstance(child, str) and child.startswith(("https://", "http://")):
                    found.append({"url": sanitize_url(child), "field": child_path})
                else:
                    visit(child, child_path, depth + 1)
        elif isinstance(item, list):
            for index, child in enumerate(item[:200]):
                visit(child, f"{path}[{index}]", depth + 1)

    visit(value, "", 0)
    return found


def _key_set(value: Any) -> set[str]:
    keys: set[str] = set()
    def visit(item: Any, depth: int) -> None:
        if depth > 8:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                keys.add(str(key).lower())
                visit(child, depth + 1)
        elif isinstance(item, list):
            for child in item[:100]:
                visit(child, depth + 1)
    visit(value, 0)
    return keys


def _semantic_datasets(keys: set[str]) -> list[str]:
    return [name for name, hints in DATASET_KEYS.items() if any(hint in keys for hint in hints)]


def _semantic_capabilities(keys: set[str]) -> dict[str, bool]:
    return {name: any(hint in keys for hint in hints) for name, hints in CAPABILITY_KEYS.items()}


def _dataset_payload(value: Any, hints: tuple[str, ...]) -> dict[str, Any]:
    """Keep only known dataset fields, bounded, instead of whole responses."""
    found: dict[str, Any] = {}
    def visit(item: Any, depth: int) -> None:
        if depth > 8 or len(found) >= 12:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).lower() in hints:
                    found[str(key)] = child
                visit(child, depth + 1)
        elif isinstance(item, list):
            for child in item[:100]:
                visit(child, depth + 1)
    visit(value, 0)
    return _compact(found)


def _endpoint(url: str, phase: str, keys: set[str]) -> dict[str, Any]:
    parsed = urlsplit(url)
    return {"url": sanitize_url(urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))), "phase": phase,
            "category": ",".join(_semantic_datasets(keys)) or "match_metadata", "top_level_keys": sorted(keys)[:30]}


class StructuredDiscovery:
    def __init__(self) -> None:
        self.responses: list[dict[str, Any]] = []
        self.references: list[dict[str, Any]] = []
        self.unknown: list[dict[str, Any]] = []
        self.capabilities: Counter[str] = Counter()
        self.dataset_payloads: dict[str, list[dict[str, Any]]] = {name: [] for name in DATASET_KEYS}

    def observe_json(self, url: str, payload: Any, phase: str, source: str = "json_response") -> None:
        keys = _key_set(payload)
        datasets = _semantic_datasets(keys)
        capabilities = _semantic_capabilities(keys)
        self.responses.append({**_endpoint(url, phase, keys), "source": source, "datasets": datasets})
        for reference in referenced_urls(payload):
            self.references.append({**reference, "phase": phase, "source": source})
        for name, observed in capabilities.items():
            if observed:
                self.capabilities[name] += 1
        for name in datasets:
            self.dataset_payloads[name].append({"endpoint": sanitize_url(url), "phase": phase,
                                                "payload": _dataset_payload(payload, DATASET_KEYS[name])})
        if not datasets and len(self.unknown) < MAX_UNKNOWN_ITEMS and ("match" in keys or "video" in keys or "render_type" in keys):
            self.unknown.append({"endpoint": sanitize_url(url), "phase": phase, "top_level_keys": sorted(keys)[:40], "payload": _compact(payload)})

    def manifest(self, match: dict[str, Any], assets: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
        datasets: dict[str, dict[str, Any]] = {}
        for name in DATASET_KEYS:
            matching = [response for response in self.responses if name in response["datasets"]]
            datasets[name] = {"status": "observed" if matching else "not_observed", "sources": matching[:10],
                              "payloads": self.dataset_payloads[name][:5]}
        capabilities = {name: {"status": "observed" if count else "not_observed", "evidence_count": count}
                        for name, count in self.capabilities.items()}
        for name in CAPABILITY_KEYS:
            capabilities.setdefault(name, {"status": "not_observed", "evidence_count": 0})
        references = list({(item["url"], item["field"], item["phase"]): item for item in self.references}.values())[:MAX_REFERENCES]
        return {"match": _safe_value(match), "assets": _safe_value(assets), "datasets": datasets,
                "capabilities": capabilities, "endpoints": self.responses[:100], "referenced_resources": references,
                "unknown_structured_data": self.unknown, "provenance": _safe_value(provenance)}


def write_manifest(path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
