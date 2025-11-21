#!/usr/bin/env python
"""
Normalize BentoML-generated install.sh scripts to use LF line endings.

This prevents remote builders that run bash from failing on the `set -exuo pipefail`
line when the script is produced on Windows (CRLF endings make `pipefail` look like
`pipefail\r` to bash).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable


def iter_install_scripts(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    return root.rglob("bentos/*/env/python/install.sh")


def normalize_file(path: Path) -> bool:
    data = path.read_bytes()
    if b"\r" not in data:
        return False

    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if normalized != data:
        path.write_bytes(normalized)
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bentoml-home",
        default=os.environ.get("BENTOML_HOME", os.path.join(Path.home(), "bentoml")),
        help="Path to the local BentoML store (defaults to %(default)s).",
    )
    args = parser.parse_args()

    bentoml_home = Path(args.bentoml_home).expanduser().resolve()
    if not bentoml_home.exists():
        print(f"[normalize_bento_install] BentoML home not found: {bentoml_home}")
        return 0

    updated = []
    for script in iter_install_scripts(bentoml_home):
        if normalize_file(script):
            updated.append(script)

    if updated:
        print("[normalize_bento_install] Converted:")
        for path in updated:
            print(f"  - {path}")
    else:
        print("[normalize_bento_install] No CRLF install.sh files detected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

