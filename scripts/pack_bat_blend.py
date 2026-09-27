from pathlib import Path

import bpy


root = Path(__file__).resolve().parents[1]
bpy.ops.wm.open_mainfile(filepath=str(root / "assets/models/creatures/bat/vampire/bat_v5.blend"))
try:
    bpy.ops.file.pack_all()
except RuntimeError:
    # Some unused preview/brush references from the source pack are absent.
    pass
bpy.ops.wm.save_as_mainfile(filepath=str(root / "assets/models/creatures/bat/vampire/bat_packed.blend"))
print(root / "assets/models/creatures/bat/vampire/bat_packed.blend")
