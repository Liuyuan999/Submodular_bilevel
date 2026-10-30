"""Prepare a macro-anchored binary benchmark from Bookshelf placement files."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def data_lines(path):
    with open(path, encoding="utf-8", errors="replace") as source:
        for raw_line in source:
            line = raw_line.split("#", 1)[0].strip()
            if line and not line.startswith("UCLA"):
                yield line


def bookshelf_files(aux_path):
    aux_path = Path(aux_path).resolve()
    entries = {}
    for line in data_lines(aux_path):
        for item in line.split(":", 1)[-1].split():
            suffix = Path(item).suffix.lower()
            if suffix in {".nodes", ".nets", ".pl", ".wts"}:
                entries[suffix] = (aux_path.parent / item).resolve()
    missing = {".nodes", ".nets", ".pl"} - entries.keys()
    if missing:
        raise ValueError(f"Bookshelf .aux is missing {sorted(missing)}")
    return entries


def read_nodes(path):
    nodes = {}
    for line in data_lines(path):
        if line.startswith(("NumNodes", "NumTerminals")):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Malformed node entry: {line}")
        name = parts[0]
        if name in nodes:
            raise ValueError(f"Duplicate node name: {name}")
        width, height = float(parts[1]), float(parts[2])
        if not math.isfinite(width) or not math.isfinite(height) or width < 0 or height < 0:
            raise ValueError(f"Invalid node dimension: {name}")
        fixed = any(flag.lower().startswith("terminal") for flag in parts[3:])
        nodes[name] = (width, height, fixed)
    if not nodes:
        raise ValueError("No nodes found")
    return nodes


def read_placement(path, nodes):
    positions = {}
    for line in data_lines(path):
        parts = line.replace(":", " ").split()
        if len(parts) < 3 or parts[0] not in nodes:
            continue
        if nodes[parts[0]][2]:
            continue
        try:
            lower_x, lower_y = float(parts[1]), float(parts[2])
        except ValueError as error:
            raise ValueError(f"Malformed placement entry: {line}") from error
        width, height, _ = nodes[parts[0]]
        if not math.isfinite(lower_x) or not math.isfinite(lower_y):
            raise ValueError(f"Nonfinite placement coordinate: {parts[0]}")
        if parts[0] in positions:
            raise ValueError(f"Duplicate placement entry: {parts[0]}")
        positions[parts[0]] = (lower_x + width / 2, lower_y + height / 2)
    missing = [name for name, (_, _, fixed) in nodes.items() if not fixed and name not in positions]
    if missing:
        raise ValueError(f"Placement lacks {len(missing)} movable nodes, starting with {missing[:3]}")
    return positions


def read_net_weights(path):
    if path is None:
        return {}
    weights = {}
    for line in data_lines(path):
        if line.startswith("NumNets"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            weight = float(parts[1])
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"Invalid net weight: {parts[0]}")
            weights[parts[0]] = weight
    return weights


def read_nets(path):
    source = iter(data_lines(path))
    for line in source:
        if line.startswith(("NumNets", "NumPins")):
            continue
        if not line.startswith("NetDegree"):
            raise ValueError(f"Expected NetDegree, got: {line}")
        parts = line.replace(":", " ").split()
        degree = int(parts[1])
        if degree < 0:
            raise ValueError(f"Negative net degree: {line}")
        name = parts[2] if len(parts) > 2 else ""
        pins = []
        for _ in range(degree):
            try:
                pins.append(next(source).split()[0])
            except StopIteration as error:
                raise ValueError(f"Truncated net: {name}") from error
        yield name, tuple(dict.fromkeys(pins))


def classify_macros(nodes, macro_list, area_ratio, dimension_ratio):
    movable = {name: (width, height) for name, (width, height, fixed) in nodes.items() if not fixed}
    if macro_list is not None:
        names = set(data_lines(macro_list))
        invalid = names - movable.keys()
        if invalid:
            raise ValueError(f"Macro list contains non-movable or unknown nodes: {sorted(invalid)[:3]}")
        rule = {"type": "explicit", "path": str(Path(macro_list).resolve())}
    else:
        areas = [width * height for width, height in movable.values() if width * height > 0]
        widths = [width for width, _ in movable.values() if width > 0]
        heights = [height for _, height in movable.values() if height > 0]
        if not areas or not widths or not heights:
            raise ValueError("Cannot infer macros from zero-size nodes")
        median_area, median_width, median_height = median(areas), median(widths), median(heights)
        names = {
            name for name, (width, height) in movable.items()
            if width * height >= area_ratio * median_area
            and (width >= dimension_ratio * median_width or height >= dimension_ratio * median_height)
        }
        rule = {"type": "size_heuristic", "area_ratio": area_ratio, "dimension_ratio": dimension_ratio,
                "median_area": median_area, "median_width": median_width, "median_height": median_height}
    if not names:
        raise ValueError("No movable macros identified; provide --macro-list or adjust classification")
    return names, rule


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(args):
    files = bookshelf_files(args.aux)
    placement_path = Path(args.placement).resolve() if args.placement else files[".pl"]
    nodes = read_nodes(files[".nodes"])
    positions = read_placement(placement_path, nodes)
    macro_names, macro_rule = classify_macros(
        nodes, args.macro_list, args.macro_area_ratio, args.macro_dimension_ratio
    )
    names = [name for name, (_, _, fixed) in nodes.items() if not fixed]
    index = {name: position for position, name in enumerate(names)}
    macro_indices = {index[name] for name in macro_names}
    net_weights = read_net_weights(files.get(".wts"))
    edge_weights = defaultdict(float)
    anchor_scores = defaultdict(float)
    skipped_large_nets = 0
    used_nets = 0
    for net_name, pins in read_nets(files[".nets"]):
        missing_pins = [name for name in pins if name not in nodes]
        if missing_pins:
            raise ValueError(f"Net {net_name} references unknown nodes: {missing_pins[:3]}")
        variable_pins = sorted(index[name] for name in pins if name in index)
        if len(variable_pins) < 2:
            continue
        if len(variable_pins) > args.max_net_degree:
            skipped_large_nets += 1
            continue
        weight = net_weights.get(net_name, 1.0)
        if weight == 0:
            continue
        used_nets += 1
        per_pair = weight / (len(variable_pins) - 1)
        macros = [position for position in variable_pins if position in macro_indices]
        for cell in variable_pins:
            if cell not in macro_indices:
                for macro in macros:
                    anchor_scores[cell, macro] += per_pair
        hub = macros[0] if macros else variable_pins[0]
        for other in variable_pins:
            if other != hub:
                edge_weights[min(hub, other), max(hub, other)] += per_pair

    by_cell = defaultdict(list)
    for (cell, macro), score in anchor_scores.items():
        by_cell[cell].append((score, macro))
    anchors = []
    for cell, scores in by_cell.items():
        scores.sort(reverse=True)
        best_score, best_macro = scores[0]
        next_score = scores[1][0] if len(scores) > 1 else 0.0
        if best_score >= args.anchor_min_weight and best_score > next_score + args.anchor_margin:
            anchors.append([cell, best_macro])
    anchors.sort()
    if not anchors:
        raise ValueError("No unambiguous anchors found; inspect the score threshold and netlist")

    raw_coordinates = [positions[name] for name in names]
    minima = [min(point[axis] for point in raw_coordinates) for axis in (0, 1)]
    maxima = [max(point[axis] for point in raw_coordinates) for axis in (0, 1)]
    if any(high <= low for low, high in zip(minima, maxima)):
        raise ValueError("Movable placement has a constant coordinate axis")
    coordinates = [
        [(point[axis] - minima[axis]) / (maxima[axis] - minima[axis]) - 0.5 for axis in (0, 1)]
        for point in raw_coordinates
    ]
    inputs = {"aux": Path(args.aux).resolve(), "nodes": files[".nodes"],
              "nets": files[".nets"], "placement": placement_path}
    if ".wts" in files:
        inputs["weights"] = files[".wts"]
    if args.macro_list:
        inputs["macro_list"] = Path(args.macro_list).resolve()
    payload = {
        "schema": "macro_anchored_mms_v1",
        "names": names,
        "coordinates": coordinates,
        "macro_indices": sorted(macro_indices),
        "edges": [[first, second, weight] for (first, second), weight in sorted(edge_weights.items())],
        "anchors": anchors,
        "metadata": {
            "inputs": {key: {"path": str(path), "sha256": file_hash(path)} for key, path in inputs.items()},
            "coordinate_bounds": [minima, maxima],
            "macro_rule": macro_rule,
            "anchor_min_weight": args.anchor_min_weight,
            "anchor_margin": args.anchor_margin,
            "max_net_degree": args.max_net_degree,
            "skipped_large_nets": skipped_large_nets,
            "used_nets": used_nets,
            "graph_projection": "macro-first star; weight divided by movable net degree minus one",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as destination:
        json.dump(payload, destination, separators=(",", ":"))
    print(json.dumps({"output": str(output), "nodes": len(names), "macros": len(macro_indices),
                      "edges": len(edge_weights), "anchors": len(anchors),
                      "skipped_large_nets": skipped_large_nets}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aux", required=True, type=Path)
    parser.add_argument("--placement", type=Path)
    parser.add_argument("--macro-list", type=Path)
    parser.add_argument("--macro-area-ratio", type=float, default=10.0)
    parser.add_argument("--macro-dimension-ratio", type=float, default=2.0)
    parser.add_argument("--max-net-degree", type=int, default=64)
    parser.add_argument("--anchor-min-weight", type=float, default=0.05)
    parser.add_argument("--anchor-margin", type=float, default=0.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (args.max_net_degree < 2 or not math.isfinite(args.anchor_min_weight)
            or not math.isfinite(args.anchor_margin) or args.anchor_min_weight < 0
            or args.anchor_margin < 0 or not math.isfinite(args.macro_area_ratio)
            or not math.isfinite(args.macro_dimension_ratio) or args.macro_area_ratio <= 0
            or args.macro_dimension_ratio <= 0):
        parser.error("net degree must be at least two, and anchor thresholds must be nonnegative")
    prepare(args)


if __name__ == "__main__":
    main()
