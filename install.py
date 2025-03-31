import launch

from pathlib import Path


root_dir = Path.cwd()
extension_dir = Path(root_dir, "extensions", "sd-optim")


with open(Path(extension_dir, "requirements.txt"), "r", encoding="utf-8") as f:
    reqs = f.readlines()

for req in reqs:
    req = req.strip()

    if not launch.is_installed(req.split("==")[0]):
        launch.run_pip(f"install {req}")
