"""Run an official DREAMPlace global placement with a pinned configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--install", required=True, type=Path)
    parser.add_argument("--aux", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--benchmark", choices=("mms", "ispd2005"), default="mms")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    source = args.source.resolve()
    install = args.install.resolve()
    aux = args.aux.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    design = aux.stem
    official_config = source / "test" / args.benchmark / f"{design}.json"
    with official_config.open(encoding="utf-8") as source_file:
        config = json.load(source_file)
    config.update({"aux_input": str(aux), "result_dir": str(output_dir),
                   "gpu": int(args.gpu), "num_threads": args.threads,
                   "legalize_flag": 0, "detailed_place_flag": 0,
                   "plot_flag": 0})
    config_path = output_dir / "dreamplace_config.json"
    with config_path.open("w", encoding="utf-8") as destination:
        json.dump(config, destination, indent=2)

    placer = install / "dreamplace" / "Placer.py"
    if not placer.is_file():
        raise FileNotFoundError(f"DREAMPlace is not installed: {placer}")
    started = time.perf_counter()
    subprocess.run([sys.executable, str(placer), str(config_path)],
                   cwd=install, check=True)
    seconds = time.perf_counter() - started
    placement = output_dir / design / f"{design}.gp.pl"
    if not placement.is_file():
        raise FileNotFoundError(f"DREAMPlace did not write a global placement: {placement}")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                       text=True).strip()
    report = {"schema": "dreamplace_placement_v1", "benchmark": args.benchmark,
              "design": design,
              "source_revision": revision, "source_config_sha256": sha256(official_config),
              "aux_sha256": sha256(aux), "gpu": args.gpu, "threads": args.threads,
              "legalize_flag": 0, "detailed_place_flag": 0,
              "placement_seconds": seconds, "placement_sha256": sha256(placement),
              "placement": str(placement)}
    with (output_dir / "placement_report.json").open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
