# install.py - V1.2 - Strip inline comments from requirements

import launch
import sys
import os
from pathlib import Path

# (Path determination logic remains the same)
try:
    extension_dir = Path(__file__).parent.resolve()
    if "extensions" not in str(extension_dir):
         import modules.launch_utils
         forge_dir = Path(modules.launch_utils.__file__).parent.parent
         extension_dir = forge_dir / "extensions" / "sd-optim"
         print(f"Install script fallback path: {extension_dir}")
    requirements_path = extension_dir / "requirements.txt"

    if not requirements_path.is_file():
         print(f"Error: requirements.txt not found at {requirements_path}", file=sys.stderr)
         sys.exit(1)

    print(f"Processing requirements from: {requirements_path}")
    with open(requirements_path, "r", encoding="utf-8") as f:
        reqs = f.readlines()

    for req in reqs:
        # --- Strip inline comments FIRST ---
        req_no_comment = req.split('#')[0].strip() # Split at #, take first part, strip whitespace

        # --- Skip empty lines (including lines that were only comments) ---
        if not req_no_comment:
            continue

        # Now use req_no_comment for processing
        package_name = req_no_comment.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].split("!=")[0].strip()

        if package_name and not launch.is_installed(package_name):
            print(f"Attempting to install {req_no_comment}...")
            # --- Pass the comment-stripped version to run_pip ---
            launch.run_pip(f"install {req_no_comment}", f"sd-optim requirement: {package_name}")
        elif package_name:
             print(f"Requirement already satisfied: {package_name} (from {req_no_comment})")
        # else: print(f"Skipping invalid requirement line: {req}")

    print("Finished processing sd-optim requirements.")

except FileNotFoundError as e_nf:
     print(f"Error: Could not find core Forge modules. {e_nf}", file=sys.stderr)
     sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during installation: {e}", file=sys.stderr)
     sys.exit(1)