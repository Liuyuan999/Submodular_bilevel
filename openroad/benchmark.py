"""Verify and prepare the official OpenROAD GCD/Nangate45 RC benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from openroad.fetch_data import FILES, REPOSITORY, REVISION, SHA256
from openroad.parsers import (
    _def_design, _delay_fit, _lef_cells, _named_blocks, _property, _section,
)
from .model import RCModel


def verify_sources(source_root):
    records = {}
    for filename, expected in SHA256.items():
        content = (source_root / filename).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected:
            raise ValueError(f"Source checksum mismatch: {filename}")
        records[filename] = {"sha256": digest, "bytes": len(content),
                             "url": f"{REPOSITORY}/blob/{REVISION}/{FILES[filename]}"}
    return records


def library_data(source, required_types=None):
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    required_units = (r'time_unit\s*:\s*"1ns"', r'capacitive_load_unit\s*\(\s*1\s*,\s*ff\s*\)',
                      r'leakage_power_unit\s*:\s*"1nW"')
    if not all(re.search(pattern, source) for pattern in required_units):
        raise ValueError("Expected ns, fF, nW Liberty units")
    result = {}
    for name, body in _named_blocks(source, "cell"):
        if required_types is not None and name not in required_types:
            continue
        pins, outputs = {}, {}
        for pin_name, pin_body in _named_blocks(body, "pin"):
            direction = re.search(r"\bdirection\s*:\s*(\w+)\s*;", pin_body)
            if direction is None:
                continue
            pin = {"direction": direction.group(1)}
            if pin["direction"] == "input":
                pin["capacitance"] = _property(pin_body, "capacitance")
            elif pin["direction"] == "output":
                function = re.search(r'\bfunction\s*:\s*"([^"]+)"', pin_body)
                pin["function"] = re.sub(r"\s+", "", function.group(1)) if function else None
                fits, samples = [], []
                for table_name in ("cell_rise", "cell_fall"):
                    for _, table in _named_blocks(pin_body, table_name):
                        fits.append(_delay_fit(table))
                        axes = re.findall(r'index_[12]\s*\(\s*"([^"]+)"\s*\)', table)
                        slew, loads = (np.fromstring(axis, sep=",") for axis in axes)
                        values = re.search(r"\bvalues\s*\((.*?)\)\s*;", table, re.S).group(1)
                        rows = np.asarray([np.fromstring(row, sep=",")
                                           for row in re.findall(r'"([^"]+)"', values)])
                        selected = int(np.argmin(np.abs(slew - 0.04)))
                        samples.extend(zip(loads, rows[selected]))
                if fits:
                    resistance, intrinsic = np.median(np.asarray(fits), axis=0)
                    if resistance <= 0 or intrinsic < 0:
                        raise ValueError(f"Nonphysical RC fit for {name}/{pin_name}")
                    loads, measured = np.asarray(samples).T
                    residual = intrinsic + resistance * loads - measured
                    outputs[pin_name] = {"resistance": float(resistance),
                                         "intrinsic_delay": float(intrinsic),
                                         "table_count": len(fits),
                                         "all_load_rmse_ns": float(np.sqrt(np.mean(residual ** 2))),
                                         "all_load_mean_relative_error": float(np.mean(
                                             np.abs(residual) / np.maximum(np.abs(measured), 1e-12)))}
            pins[pin_name] = pin
        result[name] = {"pins": pins, "outputs": outputs,
                        "sequential": bool(re.search(r"\b(?:ff|latch)\s*\(", body)),
                        "leakage": _property(body, "cell_leakage_power") if outputs else 0.0}
    return result


def wire_parameters(lef, layer="metal3"):
    match = re.search(r"(?ms)^LAYER\s+" + re.escape(layer) + r"\s*$(.*?)^END\s+" +
                      re.escape(layer) + r"\s*$", lef)
    if match is None:
        raise ValueError(f"Missing routing layer {layer}")
    body = match.group(1)
    width = float(re.search(r"(?m)^\s*WIDTH\s+([\d.eE+-]+)\s*;", body).group(1))
    area_cap = float(re.search(r"CAPACITANCE\s+CPERSQDIST\s+([\d.eE+-]+)", body).group(1))
    edge_cap = float(re.search(r"EDGECAPACITANCE\s+([\d.eE+-]+)", body).group(1))
    return {"layer": layer, "width_um": width, "area_cap_pf_per_um2": area_cap,
            "edge_cap_pf_per_um": edge_cap,
            "cap_ff_per_um": 1000 * (area_cap * width + 2 * edge_cap),
            "end_cap_ff_per_branch": 1000 * 2 * edge_cap * width,
            "model": "one minimum-width straight segment per driver-receiver branch; no shared tree"}


def net_connections(def_source):
    result = []
    seen = set()
    for match in re.finditer(r"-\s+(\S+)\s+(.*?)\s*;", _section(def_source, "NETS"), re.S):
        name, body = match.groups()
        connection_body = body.split("+", 1)[0]
        pins = re.findall(r"\(\s*(\S+)\s+(\S+)\s*\)", connection_body)
        if not pins:
            raise ValueError(f"Empty net {name}")
        for pin in pins:
            if pin in seen:
                raise ValueError(f"Pin occurs on multiple nets: {pin}")
            seen.add(pin)
        result.append((name, pins))
    return result


def build_instance(source_root, max_cells=0, placement_radius=1.0):
    source_root = Path(source_root)
    sources = verify_sources(source_root)
    if max_cells < 0 or not np.isfinite(placement_radius) or placement_radius <= 0:
        raise ValueError("max_cells must be nonnegative and placement_radius positive")
    lef_source = (source_root / "Nangate45.lef").read_text()
    def_source = (source_root / "gcd_nangate45.def").read_text()
    geometry = _lef_cells(lef_source)
    die, components, terminals, _ = _def_design(def_source, geometry)
    required_types = {component["type"] for component in components.values()}
    required_types.update({name[:-2] + "X2" for name in list(required_types) if name.endswith("_X1")})
    required_types.update({"BUF_X1", "NAND2_X1"})
    library = library_data((source_root / "Nangate45_typ.lib").read_text(), required_types)
    nets = net_connections(def_source)
    wires = wire_parameters(lef_source)
    mapped = {name: cell_type for cell_type, name in re.findall(
        r"\b([A-Za-z0-9_]+_X\d+)\s+([A-Za-z0-9_]+)\s*\(",
        (source_root / "gcd_nangate45.v").read_text())}
    rejections, eligible = {}, {}
    used_components = {name for _, connections in nets for name, _ in connections if name != "PIN"}
    for name in sorted(used_components):
        if name not in components or components[name]["type"] not in library:
            raise ValueError(f"Connected DEF component lacks geometry or Liberty data: {name}")
        component = components[name]
        small = component["type"]
        large = small[:-2] + "X2" if small.endswith("_X1") else ""
        original = library[small]
        reason = None
        if original["sequential"] or small.startswith("CLKBUF"):
            reason = "sequential_or_clock_cell"
        elif not small.endswith("_X1") or large not in library or large not in geometry:
            reason = "no_X1_X2_pair"
        elif component["orientation"] in {"E", "W", "FE", "FW"}:
            reason = "rotated_cell"
        else:
            upgraded = library[large]
            output_names = list(original["outputs"])
            if (len(output_names) != 1 or set(original["pins"]) != set(upgraded["pins"]) or
                    upgraded["sequential"] or set(output_names) != set(upgraded["outputs"])):
                reason = "unsupported_pin_structure"
            elif any(pin["direction"] != upgraded["pins"][pin_name]["direction"] or
                     (pin["direction"] == "output" and
                      (pin["function"] is None or pin["function"] != upgraded["pins"][pin_name]["function"]))
                     for pin_name, pin in original["pins"].items()):
                reason = "different_pin_function"
            elif (original["outputs"][output_names[0]]["resistance"] <
                  upgraded["outputs"][output_names[0]]["resistance"] or
                  original["leakage"] > upgraded["leakage"] or
                  np.prod(geometry[small]) > np.prod(geometry[large]) or
                  any(pin["capacitance"] > upgraded["pins"][pin_name]["capacitance"]
                      for pin_name, pin in original["pins"].items() if pin["direction"] == "input")):
                reason = "nonmonotone_library_pair"
            else:
                margin = np.maximum(geometry[small], geometry[large]) / 2
                position = np.asarray(component["position"])
                if np.any(position < margin) or np.any(position > np.asarray(die) - margin):
                    reason = "reference_outside_max_size_die_box"
        if reason:
            rejections[name] = reason
        else:
            eligible[name] = large
    selected = sorted(eligible)
    if max_cells:
        selected = selected[:max_cells]
    if not selected:
        raise ValueError("No supported combinational X1/X2 cells")
    selected_set = set(selected)
    fixed_keys = sorted({(name, pin) for _, connections in nets for name, pin in connections
                         if name not in selected_set})
    node_names = selected + [name + "/" + pin for name, pin in fixed_keys]
    selected_index = {name: index for index, name in enumerate(selected)}
    fixed_index = {key: len(selected) + index for index, key in enumerate(fixed_keys)}
    positions, resistance, intrinsic, caps, widths, heights, leakage, timing_weights = ([] for _ in range(8))
    for name in selected:
        small, large = components[name]["type"], eligible[name]
        choices = [library[small], library[large]]
        output = next(iter(choices[0]["outputs"]))
        positions.append(components[name]["position"])
        resistance.append([choice["outputs"][output]["resistance"] for choice in choices])
        intrinsic.append([choice["outputs"][output]["intrinsic_delay"] for choice in choices])
        caps.append([max(pin["capacitance"] for pin in choice["pins"].values()
                         if pin["direction"] == "input") for choice in choices])
        widths.append([geometry[small][0], geometry[large][0]])
        heights.append([geometry[small][1], geometry[large][1]])
        leakage.append([choice["leakage"] for choice in choices])
        timing_weights.append(1.0)
    input_driver = library["BUF_X1"]["outputs"]["Z"]
    output_load = library["NAND2_X1"]["pins"]["A1"]["capacitance"]

    def pin_info(name, pin):
        if name == "PIN":
            if pin not in terminals:
                raise ValueError(f"Unknown port: {pin}")
            direction = terminals[pin]["direction"]
            if direction not in {"INPUT", "OUTPUT"}:
                raise ValueError(f"Unsupported port direction: {pin}")
            return {"direction": "output" if direction == "INPUT" else "input",
                    "capacitance": output_load if direction == "OUTPUT" else 0.0}
        cell = library[components[name]["type"]]
        if pin not in cell["pins"] or cell["pins"][pin]["direction"] not in {"input", "output"}:
            raise ValueError(f"Unknown or unsupported connected pin: {name}/{pin}")
        return cell["pins"][pin]

    for name, pin in fixed_keys:
        info = pin_info(name, pin)
        is_driver = info["direction"] == "output"
        positions.append(terminals[pin]["position"] if name == "PIN" else components[name]["position"])
        electrical = (input_driver if name == "PIN" else
                      library[components[name]["type"]]["outputs"].get(pin)) if is_driver else None
        if is_driver and electrical is None:
            raise ValueError(f"No timing fit for fixed driver {name}/{pin}")
        resistance.append([electrical["resistance"] if electrical else 1.0] * 2)
        intrinsic.append([electrical["intrinsic_delay"] if electrical else 0.0] * 2)
        caps.append([info.get("capacitance", 0.0)] * 2)
        widths.append([0.0, 0.0])
        heights.append([0.0, 0.0])
        leakage.append([0.0, 0.0])
        timing_weights.append(float(is_driver))
    arcs, net_records, arc_records = [], [], []
    base_load = np.zeros(len(node_names))
    for net_name, connections in nets:
        drivers, receivers, indices = [], [], []
        for name, pin in connections:
            info = pin_info(name, pin)
            index = selected_index[name] if name in selected_set else fixed_index[name, pin]
            indices.append(index)
            if info["direction"] == "output":
                drivers.append(index)
            else:
                if name in selected_set:
                    pair = [library[variant]["pins"][pin]["capacitance"]
                            for variant in [components[name]["type"], eligible[name]]]
                else:
                    pair = [info["capacitance"]] * 2
                receivers.append((index, pair, name, pin))
        if len(drivers) != 1:
            raise ValueError(f"Net {net_name} has {len(drivers)} drivers; expected one")
        source = drivers[0]
        for target, pair, name, pin in receivers:
            if source == target:
                raise ValueError(f"Combinational self-loop on {net_name}")
            arcs.append([source, target, wires["cap_ff_per_um"], 1.0, pair])
            base_load[source] += wires["end_cap_ff_per_branch"]
            arc_records.append({"net": net_name, "receiver": name + "/" + pin})
        if len(set(indices)) > 1:
            net_records.append({"name": net_name, "nodes": sorted(set(indices)), "weight": 1.0})
    count = len(selected)
    data = {
        "origin": "openroad_gcd_rc_refinement", "die": die, "grid": [8, 8],
        "reference": positions[:count], "fixed_positions": positions[count:],
        "placement_radius": placement_radius,
        "resistance": resistance, "input_cap": caps, "intrinsic_delay": intrinsic,
        "width": widths, "height": heights, "leakage": leakage,
        "wire_cap_base": base_load.tolist(), "lower_timing_weights": timing_weights,
        "upper_timing_weights": timing_weights.copy(), "arcs": arcs, "nets": net_records,
        "wire_smoothing": 0.05, "rc_smoothing": 0.02,
        "normalizers": {"wirelength": 1.0, "timing": 1.0, "density": 1.0,
                        "area": 1.0, "leakage": 1.0},
        "lower_weights": {"area": 0.01, "leakage": 0.01, "density": 0.0},
        "upper_weights": {"wirelength": 0.5, "displacement": 0.1, "density": 0.0,
                          "timing": 1.0, "area": 0.4, "leakage": 0.4},
        "node_names": node_names, "arc_records": arc_records,
        "physical_fixed_cells": [components[name] for name in sorted(components) if name not in selected_set],
    }
    model = RCModel(data)
    initial = model.evaluate(model.reference, np.zeros(count, dtype=np.uint8))
    for key, metric in (("wirelength", "wirelength"), ("timing", "lower_timing"),
                        ("area", "area"), ("leakage", "leakage")):
        data["normalizers"][key] = max(initial[metric], 1e-12)
    data["upper_weights"]["displacement"] = (
        0.1 * data["normalizers"]["wirelength"] ** 2 / (count * placement_radius ** 2))
    selected_types = sorted({variant for name in selected
                             for variant in [components[name]["type"], eligible[name]]})
    data["provenance"] = {
        "repository": REPOSITORY, "revision": REVISION, "sources": sources,
        "authoritative_connectivity": "placed/routed DEF NETS connections before routing clauses",
        "scope": "local continuous refinement of all eligible combinational cells in the published GCD DEF",
        "component_count_nonfiller": len(components), "connected_component_count": len(used_components),
        "net_count": len(nets), "arc_count": len(arcs), "terminal_count": len(terminals),
        "eligible_count": len(eligible), "selected_count": count,
        "selection": "all eligible cells sorted by DEF instance name; optional prefix for verification",
        "selected_pairs": {name: [components[name]["type"], eligible[name]] for name in selected},
        "rejections": rejections, "rejection_counts": dict(Counter(rejections.values())),
        "verilog_crosscheck": {
            "same_name_and_type": sum(mapped.get(name) == component["type"] for name, component in components.items()),
            "same_name_resized": [name for name in components if name in mapped and
                                  components[name]["type"] != mapped[name]],
            "DEF_only_instances": sorted(set(components) - set(mapped)),
            "Verilog_only_instances": sorted(set(mapped) - set(components)),
            "interpretation": "earlier synthesis Verilog differs from the repaired DEF; not interchangeable netlists"},
        "wire_model": wires,
        "units": {"position": "um", "delay": "ns", "capacitance": "fF", "leakage": "nW",
                  "effective_resistance": "ns/fF", "area": "um^2"},
        "fit": "median per-output Liberty rise/fall load slopes and intercepts at nearest 0.04 ns slew, load indices 1:5",
        "fit_diagnostics": {name: library[name]["outputs"] for name in selected_types},
        "fixed_pin_policy": "fixed input/output pins are separate timing endpoints, breaking sequential feedback",
        "pin_positions": "cell centers; no variant-dependent pin offsets",
        "primary_input_driver": "BUF_X1/Z fitted RC", "primary_output_load": "NAND2_X1/A1 capacitance",
        "timing_weights": "one per driven stage, fixed for the entire run; not worst-path timing or STA",
        "policy_weights": "declared experiment parameters, not fitted to an industrial sizer",
        "normalizers": "all-small selected variants at the published reference, fixed throughout optimization",
        "displacement_penalty": "0.1 * mean squared displacement / placement_radius^2",
    }
    data["assumptions"] = RCModel(data).certificate()
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("openroad/data/upstream"))
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--placement-radius", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    data = build_instance(arguments.source, arguments.max_cells, arguments.placement_radius)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"selected": data["provenance"]["selected_count"],
                      "eligible": data["provenance"]["eligible_count"],
                      "nets": data["provenance"]["net_count"], "arcs": len(data["arcs"]),
                      "assumptions": data["assumptions"]}, indent=2))


if __name__ == "__main__":
    main()
