import argparse
from pathlib import Path

import bpy


def export_obj(source: Path, target: Path) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if source.suffix.lower() == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(source))
    elif source.suffix.lower() == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(source))
    else:
        raise ValueError(f"Unsupported source: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.obj_export(filepath=str(target), path_mode="COPY")


parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("target", type=Path)
argv = __import__("sys").argv
args = parser.parse_args(argv[argv.index("--") + 1:] if "--" in argv else argv[1:])
export_obj(args.source, args.target)
