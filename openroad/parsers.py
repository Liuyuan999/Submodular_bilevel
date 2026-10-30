"""DEF, LEF, and Liberty parsing helpers."""

import re
import numpy as np


def _braced_body(source, opening):
    depth = 1
    for position in range(opening + 1, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:position]
    raise ValueError("Unbalanced library braces")


def _named_blocks(source, keyword):
    expression = re.compile(r"\b" + re.escape(keyword) +
                            r"\s*\(\s*([A-Za-z0-9_]+)?\s*\)\s*\{")
    for match in expression.finditer(source):
        yield match.group(1), _braced_body(source, match.end() - 1)


def _property(source, name):
    match = re.search(r"\b" + re.escape(name) +
                      r"\s*:\s*([+\-]?[0-9.eE]+)\s*;", source)
    if match is None:
        raise ValueError(f"Missing Liberty property {name}")
    return float(match.group(1))


def _delay_fit(table):
    axes = re.findall(r'index_[12]\s*\(\s*"([^"]+)"\s*\)', table)
    rows_match = re.search(r"\bvalues\s*\((.*?)\)\s*;", table, re.S)
    if len(axes) != 2 or rows_match is None:
        raise ValueError("Unsupported Liberty timing table")
    transitions = np.fromstring(axes[0], sep=",")
    loads = np.fromstring(axes[1], sep=",")
    rows = [np.fromstring(row, sep=",") for row in re.findall(r'"([^"]+)"', rows_match.group(1))]
    values = np.asarray(rows)
    if values.shape != (len(transitions), len(loads)) or len(loads) < 4:
        raise ValueError("Unexpected Liberty timing-table shape")
    transition_index = int(np.argmin(np.abs(transitions - 0.04)))
    load_slice = slice(1, min(5, len(loads)))
    slope, intercept = np.polyfit(loads[load_slice], values[transition_index, load_slice], 1)
    return float(slope), float(intercept)


def _lef_cells(source):
    cells = {}
    for match in re.finditer(r"(?m)^MACRO\s+(\S+)\s*$", source):
        name = match.group(1)
        ending = re.search(r"(?m)^END\s+" + re.escape(name) + r"\s*$", source[match.end():])
        if ending is None:
            raise ValueError(f"LEF macro {name} has no END")
        body = source[match.end():match.end() + ending.start()]
        size = re.search(r"\bSIZE\s+([0-9.]+)\s+BY\s+([0-9.]+)\s*;", body)
        if size:
            cells[name] = (float(size.group(1)), float(size.group(2)))
    return cells


def _section(source, name):
    match = re.search(r"\b" + name + r"\s+\d+\s*;(.*?)\bEND\s+" + name,
                      source, re.S)
    if match is None:
        raise ValueError(f"Missing DEF {name} section")
    return match.group(1)


def _def_design(source, lef_cells):
    unit_match = re.search(r"UNITS\s+DISTANCE\s+MICRONS\s+(\d+)\s*;", source)
    die_match = re.search(r"DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*"
                          r"\(\s*(-?\d+)\s+(-?\d+)\s*\)", source)
    if unit_match is None or die_match is None:
        raise ValueError("Missing DEF units or die")
    units = int(unit_match.group(1))
    die_values = np.asarray([int(die_match.group(index)) for index in range(1, 5)]) / units
    origin = die_values[:2]
    die = die_values[2:] - origin
    components = {}
    for match in re.finditer(r"-\s+(\S+)\s+(\S+)(.*?)\s*;", _section(source, "COMPONENTS"), re.S):
        name, cell_type, body = match.groups()
        placement = re.search(r"\+\s+(?:PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\w+)", body)
        if placement is None or cell_type not in lef_cells or cell_type.startswith("FILLCELL"):
            continue
        width, height = lef_cells[cell_type]
        orientation = placement.group(3)
        if orientation in {"E", "W", "FE", "FW"}:
            width, height = height, width
        lower_left = np.asarray([int(placement.group(1)), int(placement.group(2))]) / units - origin
        components[name] = {"type": cell_type, "position": (lower_left +
                             np.asarray([width, height]) / 2).tolist(),
                            "dimensions": [width, height], "orientation": orientation}
    terminals = {}
    for match in re.finditer(r"-\s+(\S+)\s+(.*?)\s*;", _section(source, "PINS"), re.S):
        name, body = match.groups()
        direction = re.search(r"\+\s+DIRECTION\s+(\w+)", body)
        placement = re.search(r"\+\s+(?:PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)", body)
        if direction and placement:
            coordinates = np.asarray([int(placement.group(1)), int(placement.group(2))]) / units - origin
            terminals[name] = {"direction": direction.group(1), "position": coordinates.tolist()}
    nets = []
    for match in re.finditer(r"-\s+(\S+)\s+(.*?)\s*;", _section(source, "NETS"), re.S):
        connections = re.findall(r"\(\s*(\S+)\s+(\S+)\s*\)", match.group(2))
        nets.append((match.group(1), connections))
    return die.tolist(), components, terminals, nets

