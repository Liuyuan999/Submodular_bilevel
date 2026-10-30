"""Fetch the pinned OpenROAD GCD/Nangate45 source fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


REVISION = "7678d51077daa1ae82fbe799e096282f9d3dcaad"
REPOSITORY = "https://github.com/The-OpenROAD-Project/OpenROAD"
FILES = {
    "gcd_nangate45.def": "test/gcd_nangate45.def",
    "gcd_nangate45.v": "test/gcd_nangate45.v",
    "Nangate45.lef": "test/Nangate45/Nangate45.lef",
    "Nangate45_typ.lib": "test/Nangate45/Nangate45_typ.lib",
}
SHA256 = {
    "gcd_nangate45.def": "5eb7b2197a635df30ea97871b6b69cb5d64da919de8fb898b53bf8cf89bdeb66",
    "gcd_nangate45.v": "a7ac54960562b5bcc012efe899a059b4cc22fda89f64c36da7f14305e1d945f3",
    "Nangate45.lef": "caa4e9b45e081f93720ce2c4d581497321ddc6fbcb1ffe1b384e324bbd5d43a7",
    "Nangate45_typ.lib": "2efd0b32eb580e4e60e72fc0575bb3bc69aac907c91d908442e4ae6d7fe55895",
}


def fetch(destination):
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    for filename, source_path in FILES.items():
        source = ("https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD/"
                  f"{REVISION}/{source_path}")
        with urlopen(source, timeout=60) as response:
            content = response.read()
        if len(content) < 100 or content.startswith(b"404:"):
            raise RuntimeError(f"Source file unavailable: {source}")
        digest = hashlib.sha256(content).hexdigest()
        if digest != SHA256[filename]:
            raise RuntimeError(f"Upstream content changed for {source_path}: {digest}")
        target = destination / filename
        target.write_bytes(content)
        records[filename] = {"upstream_path": source_path,
                             "source_url": source,
                             "sha256": digest,
                             "bytes": len(content)}
    manifest = {"repository": REPOSITORY, "revision": REVISION,
                "design": "OpenROAD GCD regression fixture",
                "platform": "Nangate45", "files": records}
    (destination / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("openroad/data/upstream"))
    arguments = parser.parse_args()
    manifest = fetch(arguments.output)
    for filename, record in manifest["files"].items():
        print(filename, record["sha256"])


if __name__ == "__main__":
    main()
