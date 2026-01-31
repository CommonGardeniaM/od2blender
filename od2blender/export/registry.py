from __future__ import annotations

from od2blender.export.base import Exporter
from od2blender.export.blender_exporter import BlenderExporter
from od2blender.export.tracks_json import TracksJsonExporter


_EXPORTERS: dict[str, Exporter] = {
    TracksJsonExporter.id: TracksJsonExporter(),
    BlenderExporter.id: BlenderExporter(),
}


def get_exporter(exporter_id: str) -> Exporter | None:
    return _EXPORTERS.get(exporter_id)


def list_exporters() -> list[str]:
    return sorted(_EXPORTERS.keys())
