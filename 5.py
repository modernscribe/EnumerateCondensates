#!/usr/bin/env python3
import sys
import json
import math
import argparse
from typing import Dict, Any, List, Tuple
from itertools import combinations

# ============== IO ==============

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default

def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default

# ============== ELEMENT EFFECTIVE PROPERTIES ==============

def build_element_effective_properties(isotopes_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    for el in isotopes_data.get("elements", []):
        Z = safe_int(el.get("Z", 0), 0)
        if Z <= 0:
            continue
        symbol = el.get("symbol", "")
        name = el.get("name", "")
        iso_list = el.get("isotopes", [])
        if not iso_list:
            continue
        total_abund = 0.0
        mass_sum = 0.0
        stab_sum = 0.0
        harm_sum = 0.0
        dim_sum = 0.0
        stable_count = 0
        for iso in iso_list:
            A = safe_int(iso.get("A", 0), 0)
            if A <= 0:
                continue
            abund = safe_float(iso.get("abundance", 0.0), 0.0)
            frac = safe_float(iso.get("stability_frac", 0.0), 0.0)
            harm = safe_float(iso.get("harmonic_eff", 0.0), 0.0)
            dim = safe_float(iso.get("dim_mix", 0.0), 0.0)
            total_abund += abund
            mass_sum += abund * A
            stab_sum += abund * frac
            harm_sum += abund * harm
            dim_sum += abund * dim
            if iso.get("stable", False):
                stable_count += 1
        if total_abund <= 0.0:
            total_abund = float(len(iso_list))
            for iso in iso_list:
                A = safe_int(iso.get("A", 0), 0)
                frac = safe_float(iso.get("stability_frac", 0.0), 0.0)
                harm = safe_float(iso.get("harmonic_eff", 0.0), 0.0)
                dim = safe_float(iso.get("dim_mix", 0.0), 0.0)
                mass_sum += A
                stab_sum += frac
                harm_sum += harm
                dim_sum += dim
        eff_mass = mass_sum / total_abund if total_abund > 0.0 else 0.0
        eff_stab = stab_sum / total_abund if total_abund > 0.0 else 0.0
        eff_harm = harm_sum / total_abund if total_abund > 0.0 else 0.0
        eff_dim = dim_sum / total_abund if total_abund > 0.0 else 0.0
        stable_iso_fraction = float(stable_count) / float(len(iso_list)) if iso_list else 0.0
        elements.append({
            "Z": Z,
            "symbol": symbol,
            "name": name,
            "effective_mass": eff_mass,
            "effective_stability": eff_stab,
            "effective_harmonic": eff_harm,
            "effective_dim_mix": eff_dim,
            "stable_iso_fraction": stable_iso_fraction,
            "n_isotopes": len(iso_list),
        })
    elements.sort(key=lambda e: e["Z"])
    return elements

# ============== VALENCE / BOND CAPACITY ==============

VALENCE_CAPACITY: Dict[str, int] = {
    "H": 1, "He": 0,
    "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 3, "O": 2, "F": 1, "Ne": 0,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 3, "S": 2, "Cl": 1, "Ar": 0,
    "K": 1, "Ca": 2,
    "Ti": 4, "V": 3, "Cr": 3, "Mn": 2, "Fe": 2, "Co": 2, "Ni": 2, "Cu": 2, "Zn": 2,
    "Ga": 3, "Ge": 4, "As": 3, "Se": 2, "Br": 1, "Kr": 0,
    "Rb": 1, "Sr": 2,
    "Ag": 1, "Cd": 2, "Sn": 4, "Sb": 3, "Te": 2, "I": 1, "Xe": 0,
    "Cs": 1, "Ba": 2,
    "Pt": 4, "Au": 3, "Hg": 2, "Pb": 4, "Bi": 3, "Po": 2, "At": 1, "Rn": 0,
    "Ts": 1, "Og": 0,
}

def max_bonds_for_symbol(symbol: str) -> int:
    v = VALENCE_CAPACITY.get(symbol)
    if v is None:
        return 4
    return max(0, v)

# ============== GEOMETRIC EMBEDDING (TREE PRISM) ==============

def build_bond_tree(capacities: List[int]) -> Tuple[bool, List[Tuple[int, int]]]:
    n = len(capacities)
    if n == 0:
        return False, []
    if n == 1:
        return True, []
    caps = list(capacities)
    if any(c < 0 for c in caps):
        return False, []
    if sum(caps) < 2 * (n - 1):
        return False, []
    remaining = list(range(n))
    edges: List[Tuple[int, int]] = []
    while len(remaining) > 1:
        remaining.sort(key=lambda i: caps[i], reverse=True)
        i = remaining[0]
        if caps[i] <= 0:
            return False, []
        j = None
        for k in remaining[1:]:
            if caps[k] > 0:
                j = k
                break
        if j is None:
            return False, []
        edges.append((i, j))
        caps[i] -= 1
        caps[j] -= 1
        if caps[i] == 0:
            if i in remaining:
                remaining.remove(i)
        if caps[j] == 0:
            if j in remaining:
                remaining.remove(j)
    return True, edges

def embed_tree_prism(n: int, edges: List[Tuple[int, int]]) -> Dict[str, Any]:
    if n == 0:
        return {
            "dimension": 3,
            "primitive": "layered_prism",
            "atom_positions": [],
            "bonds": [],
            "bounding_box": {
                "min": [0.0, 0.0, 0.0],
                "max": [0.0, 0.0, 0.0],
            },
            "layers": 0,
            "base_vertices": 0,
        }
    adj: List[List[int]] = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    from collections import deque
    depth = [-1] * n
    parent = [-1] * n
    q = deque()
    root = 0
    depth[root] = 0
    q.append(root)
    order: List[int] = [root]
    while q:
        u = q.popleft()
        for v in adj[u]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1
                parent[v] = u
                q.append(v)
                order.append(v)
    max_depth = max(depth)
    layers: List[List[int]] = [[] for _ in range(max_depth + 1)]
    for idx, d in enumerate(depth):
        if d < 0:
            d = max_depth
        layers[d].append(idx)
    positions = [[0.0, 0.0, 0.0] for _ in range(n)]
    for d, layer_nodes in enumerate(layers):
        m = len(layer_nodes)
        if m == 0:
            continue
        radius = 1.0 + 0.5 * float(d)
        z = float(d)
        if m == 1:
            positions[layer_nodes[0]] = [0.0, 0.0, z]
        else:
            for idx_in_layer, node in enumerate(layer_nodes):
                angle = 2.0 * math.pi * float(idx_in_layer) / float(m)
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                positions[node] = [x, y, z]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    bbox_min = [min(xs), min(ys), min(zs)]
    bbox_max = [max(xs), max(ys), max(zs)]
    base_vertices = max(len(layer) for layer in layers) if layers else 0
    return {
        "dimension": 3,
        "primitive": "layered_prism",
        "atom_positions": positions,
        "bonds": edges,
        "bounding_box": {
            "min": bbox_min,
            "max": bbox_max,
        },
        "layers": max_depth + 1,
        "base_vertices": base_vertices,
    }

# ============== MOLECULE ENUMERATION ==============

def generate_compositions(k: int, max_atoms: int) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    def rec(pos: int, remaining: int, current: List[int]) -> None:
        if pos == k:
            if remaining == 0 and sum(current) > 0:
                out.append(tuple(current))
            return
        for c in range(1, remaining - (k - pos - 1) + 1):
            current.append(c)
            rec(pos + 1, remaining - c, current)
            current.pop()
    for total_atoms in range(1, max_atoms + 1):
        if total_atoms < k:
            continue
        rec(0, total_atoms, [])
    return out

def build_formula(elements_subset: List[Dict[str, Any]], counts: Tuple[int, ...]) -> str:
    parts: List[str] = []
    for el, n in zip(elements_subset, counts):
        sym = el["symbol"]
        if n == 1:
            parts.append(sym)
        else:
            parts.append(f"{sym}{n}")
    return "".join(parts)

def enumerate_molecules(
    elements: List[Dict[str, Any]],
    max_elements: int,
    max_atoms: int,
    max_molecules: int
) -> List[Dict[str, Any]]:
    molecules: List[Dict[str, Any]] = []
    elem_count = len(elements)
    mol_id = 0
    for k in range(1, min(max_elements, elem_count) + 1):
        for idxs in combinations(range(elem_count), k):
            subset = [elements[i] for i in idxs]
            comps = generate_compositions(k, max_atoms)
            for counts in comps:
                atoms = sum(counts)
                if atoms <= 0:
                    continue
                atom_caps: List[int] = []
                atom_meta: List[Dict[str, Any]] = []
                mass = 0.0
                stab_sum = 0.0
                harm_sum = 0.0
                dim_sum = 0.0
                stable_iso_weight = 0.0
                for el, c in zip(subset, counts):
                    cap = max_bonds_for_symbol(el["symbol"])
                    for _ in range(c):
                        atom_caps.append(cap)
                        atom_meta.append({
                            "Z": el["Z"],
                            "symbol": el["symbol"],
                            "name": el["name"],
                        })
                    mass += c * el["effective_mass"]
                    stab_sum += c * el["effective_stability"]
                    harm_sum += c * el["effective_harmonic"]
                    dim_sum += c * el["effective_dim_mix"]
                    stable_iso_weight += c * el["stable_iso_fraction"]
                if len(atom_caps) != atoms:
                    continue
                ok, edges = build_bond_tree(atom_caps)
                if not ok:
                    continue
                avg_stability = stab_sum / float(atoms) if atoms > 0 else 0.0
                avg_harm = harm_sum / float(atoms) if atoms > 0 else 0.0
                avg_dim = dim_sum / float(atoms) if atoms > 0 else 0.0
                avg_stable_iso_fraction = stable_iso_weight / float(atoms) if atoms > 0 else 0.0
                formula = build_formula(subset, counts)
                geom = embed_tree_prism(atoms, edges)
                comp_list: List[Dict[str, Any]] = []
                offset = 0
                for el, c in zip(subset, counts):
                    comp_list.append({
                        "Z": el["Z"],
                        "symbol": el["symbol"],
                        "name": el["name"],
                        "count": c,
                    })
                    offset += c
                mol_id += 1
                molecules.append({
                    "id": mol_id,
                    "formula": formula,
                    "atoms": atoms,
                    "unique_elements": k,
                    "composition": comp_list,
                    "mass_estimate": mass,
                    "stability_index": avg_stability,
                    "harmonic_profile": {
                        "effective_harmonic": avg_harm,
                        "effective_dim_mix": avg_dim,
                        "avg_stable_iso_fraction": avg_stable_iso_fraction,
                    },
                    "geometry": {
                        "primitive": geom["primitive"],
                        "dimension": geom["dimension"],
                        "atom_positions": geom["atom_positions"],
                        "bonds": geom["bonds"],
                        "bounding_box": geom["bounding_box"],
                        "layers": geom["layers"],
                        "base_vertices": geom["base_vertices"],
                    },
                    "valence_model": {
                        "atom_capacities": atom_caps,
                        "construction_valid": True,
                    },
                })
                if max_molecules > 0 and len(molecules) >= max_molecules:
                    return molecules
    return molecules

# ============== METAMATERIAL ENUMERATION (GEOMETRIC PRISM) ==============

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)

def embed_metamaterial_prism(a_count: int, b_count: int) -> Dict[str, Any]:
    atoms = a_count + b_count
    if atoms <= 0:
        return {
            "dimension": 3,
            "primitive": "bilayer_prism",
            "atom_positions": [],
            "layers": 0,
            "base_vertices": 0,
            "bounding_box": {
                "min": [0.0, 0.0, 0.0],
                "max": [0.0, 0.0, 0.0],
            },
        }
    positions: List[List[float]] = []
    z0 = 0.0
    z1 = 1.0
    if a_count > 0:
        for i in range(a_count):
            angle = 2.0 * math.pi * float(i) / float(max(1, a_count))
            x = math.cos(angle)
            y = math.sin(angle)
            positions.append([x, y, z0])
    if b_count > 0:
        for i in range(b_count):
            angle = 2.0 * math.pi * float(i) / float(max(1, b_count))
            x = 1.5 * math.cos(angle + math.pi / float(max(1, b_count)))
            y = 1.5 * math.sin(angle + math.pi / float(max(1, b_count)))
            positions.append([x, y, z1])
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    bbox_min = [min(xs), min(ys), min(zs)]
    bbox_max = [max(xs), max(ys), max(zs)]
    base_vertices = max(a_count, b_count)
    return {
        "dimension": 3,
        "primitive": "bilayer_prism",
        "atom_positions": positions,
        "layers": 2,
        "base_vertices": base_vertices,
        "bounding_box": {
            "min": bbox_min,
            "max": bbox_max,
        },
    }

def enumerate_metamaterials(
    elements: List[Dict[str, Any]],
    lattice_atoms: int,
    max_metamaterials: int
) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    elem_count = len(elements)
    meta_id = 0
    for i in range(elem_count):
        for j in range(i + 1, elem_count):
            el_a = elements[i]
            el_b = elements[j]
            for a in range(1, lattice_atoms):
                for b in range(1, lattice_atoms - a + 1):
                    if gcd(a, b) != 1:
                        continue
                    atoms = a + b
                    mass = a * el_a["effective_mass"] + b * el_b["effective_mass"]
                    stab_sum = a * el_a["effective_stability"] + b * el_b["effective_stability"]
                    harm_a = el_a["effective_harmonic"]
                    harm_b = el_b["effective_harmonic"]
                    dim_a = el_a["effective_dim_mix"]
                    dim_b = el_b["effective_dim_mix"]
                    stab_index = stab_sum / float(atoms) if atoms > 0 else 0.0
                    harm_contrast = abs(harm_a - harm_b)
                    dim_contrast = abs(dim_a - dim_b)
                    harm_mean = (a * harm_a + b * harm_b) / float(atoms) if atoms > 0 else 0.0
                    dim_mean = (a * dim_a + b * dim_b) / float(atoms) if atoms > 0 else 0.0
                    pattern = f"{el_a['symbol']}{a}-{el_b['symbol']}{b}"
                    geom = embed_metamaterial_prism(a, b)
                    meta_id += 1
                    metas.append({
                        "id": meta_id,
                        "pattern": pattern,
                        "unit_atoms": atoms,
                        "elements": [
                            {
                                "Z": el_a["Z"],
                                "symbol": el_a["symbol"],
                                "name": el_a["name"],
                                "count": a,
                            },
                            {
                                "Z": el_b["Z"],
                                "symbol": el_b["symbol"],
                                "name": el_b["name"],
                                "count": b,
                            },
                        ],
                        "mass_estimate": mass,
                        "stability_index": stab_index,
                        "harmonic_profile": {
                            "mean_harmonic": harm_mean,
                            "mean_dim_mix": dim_mean,
                            "harmonic_contrast": harm_contrast,
                            "dim_mix_contrast": dim_contrast,
                        },
                        "geometry": {
                            "primitive": geom["primitive"],
                            "dimension": geom["dimension"],
                            "atom_positions": geom["atom_positions"],
                            "layers": geom["layers"],
                            "base_vertices": geom["base_vertices"],
                            "bounding_box": geom["bounding_box"],
                        },
                    })
                    if max_metamaterials > 0 and len(metas) >= max_metamaterials:
                        return metas
    return metas

# ============== MAIN ==============

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate geometrically-embedded molecules and metamaterials from predicted isotope table"
    )
    parser.add_argument(
        "--isotopes",
        default="predicted_isotopes.json",
        help="Path to predicted isotopes JSON (output of lattice isotope tool)"
    )
    parser.add_argument(
        "--max-elements",
        type=int,
        default=3,
        help="Maximum distinct elements per molecule"
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=6,
        help="Maximum total atoms per molecule"
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=0,
        help="Maximum number of molecules to emit (0 = no limit)"
    )
    parser.add_argument(
        "--lattice-atoms",
        type=int,
        default=4,
        help="Maximum atoms in metamaterial primitive cell"
    )
    parser.add_argument(
        "--max-metamaterials",
        type=int,
        default=0,
        help="Maximum number of metamaterials to emit (0 = no limit)"
    )
    parser.add_argument(
        "--out-molecules",
        type=str,
        default="molecules_geom.json",
        help="Output JSON path for molecules with geometry"
    )
    parser.add_argument(
        "--out-metamaterials",
        type=str,
        default="metamaterials_geom.json",
        help="Output JSON path for metamaterials with geometry"
    )
    args = parser.parse_args()
    try:
        isotopes_data = load_json(args.isotopes)
    except FileNotFoundError:
        print(f"Error: isotopes file not found: {args.isotopes}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: failed to load isotopes JSON: {e}")
        sys.exit(1)
    elements = build_element_effective_properties(isotopes_data)
    if not elements:
        print("Error: no valid elements found in isotopes JSON")
        sys.exit(1)
    molecules = enumerate_molecules(
        elements=elements,
        max_elements=max(1, args.max_elements),
        max_atoms=max(1, args.max_atoms),
        max_molecules=max(0, args.max_molecules),
    )
    metamaterials = enumerate_metamaterials(
        elements=elements,
        lattice_atoms=max(2, args.lattice_atoms),
        max_metamaterials=max(0, args.max_metamaterials),
    )
    try:
        with open(args.out_molecules, "w", encoding="utf-8") as f:
            json.dump({"molecules": molecules}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error: failed to write molecules JSON: {e}")
        sys.exit(1)
    try:
        with open(args.out_metamaterials, "w", encoding="utf-8") as f:
            json.dump({"metamaterials": metamaterials}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error: failed to write metamaterials JSON: {e}")
        sys.exit(1)
    print(f"Wrote {len(molecules)} molecules to {args.out_molecules}")
    print(f"Wrote {len(metamaterials)} metamaterials to {args.out_metamaterials}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
