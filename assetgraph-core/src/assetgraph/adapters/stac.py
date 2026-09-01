from __future__ import annotations

from typing import Any, Iterable, Mapping

from .base import SourceEnvelope


class STACItemAdapter:
    adapter_id = "stac-item-v1"

    def __init__(self, *, preferred_asset_keys: tuple[str, ...] = ("visual", "image", "data")) -> None:
        self.preferred_asset_keys = preferred_asset_keys

    def normalize(self, source: Mapping[str, Any]) -> Iterable[SourceEnvelope]:
        if source.get("type") != "Feature" or "id" not in source:
            raise ValueError("expected a STAC Item/GeoJSON Feature with id")
        props = source.get("properties") or {}
        observed_at = props.get("datetime") or props.get("start_datetime")
        if not observed_at:
            raise ValueError("STAC item requires datetime or start_datetime")
        assets = source.get("assets") or {}
        if not assets:
            raise ValueError("STAC item has no assets")

        selected_key = next((k for k in self.preferred_asset_keys if k in assets), None)
        if selected_key is None:
            selected_key = sorted(assets)[0]
        asset = assets[selected_key]
        href = asset.get("href")
        if not href:
            raise ValueError("selected STAC asset has no href")
        media_type = asset.get("type") or "application/octet-stream"

        sensor = (
            props.get("platform")
            or props.get("instruments", [None])[0]
            or "unknown"
        )
        telemetry = {
            key: props[key]
            for key in ("platform", "instruments", "gsd", "view:off_nadir", "view:azimuth", "proj:epsg")
            if key in props
        }
        yield SourceEnvelope(
            source_id=str(source["id"]),
            sensor=str(sensor),
            observed_at=str(observed_at),
            uri=str(href),
            media_type=str(media_type),
            geometry=source.get("geometry") or {},
            telemetry=telemetry,
            provider=props.get("provider"),
            collection=source.get("collection"),
            metadata={"asset_key": selected_key, "stac_version": source.get("stac_version")},
        )
