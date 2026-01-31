from __future__ import annotations

import subprocess

from od2blender.export.base import ExportContext, Exporter


class BlenderExporter(Exporter):
    id = "blender"

    def export(self, payload: dict, context: ExportContext) -> dict:
        if context.blender_path is None:
            raise FileNotFoundError("Blender executable not found")
        cmd = [
            str(context.blender_path),
            "--background",
            "--factory-startup",
            "--python",
            str(context.importer_path),
            "--",
            "--tracks",
            str(context.tracks_path),
            "--video",
            str(context.video_path),
            "--out",
            str(context.scene_path),
            "--plane-width",
            str(context.plane_width),
            "--z-depth",
            str(context.z_depth),
        ]
        if context.log_info:
            context.log_info("Running Blender: {}".format(" ".join(cmd)))
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.stdout and context.log_info:
            for line in result.stdout.splitlines():
                context.log_info("[blender] {}".format(line))
        if result.stderr and context.log_warn:
            for line in result.stderr.splitlines():
                context.log_warn("[blender] {}".format(line))
        if result.returncode != 0:
            raise RuntimeError(f"Blender importer failed with code {result.returncode}")

        if context.open_blender:
            try:
                subprocess.Popen([str(context.blender_path), str(context.scene_path)])
                if context.log_info:
                    context.log_info("Launched Blender UI")
            except Exception as exc:
                if context.log_warn:
                    context.log_warn(f"Failed to launch Blender UI: {exc}")

        return {"scene": context.scene_path}
