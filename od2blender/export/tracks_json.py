from __future__ import annotations

from od2blender.export.base import ExportContext, Exporter
from od2blender.json_utils import save_json


class TracksJsonExporter(Exporter):
    id = "tracks"

    def export(self, payload: dict, context: ExportContext) -> dict:
        save_json(payload, context.tracks_path)
        return {"tracks": context.tracks_path}
