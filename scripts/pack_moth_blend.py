from pathlib import Path

import bpy


root = Path(__file__).resolve().parents[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(root / "assets/models/creatures/moth/moth.glb"))
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(root / "assets/models/creatures/moth/moth.blend"))
print(root / "assets/models/creatures/moth/moth.blend")
