#!/usr/bin/env python3
import sys
import json
import math
import argparse
from typing import Dict, List, Any
from collections import defaultdict

NOTE_NAMES_A_ROOT = [
    "A", "A#", "B", "C", "C#", "D",
    "D#", "E", "F", "F#", "G", "G#"
]

def load_tuning(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

def degree_from_dim(dim_index: int) -> int:
    return (dim_index + 12) % 12

def is_power_of_two(n: int) -> bool:
    if n <= 0:
        return False
    return (n & (n - 1)) == 0

def nearest_ladder_by_freq(freq: float, ladder_sorted: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not ladder_sorted or freq <= 0.0:
        return None
    lo, hi = 0, len(ladder_sorted) - 1
    if freq <= ladder_sorted[0]["frequency_hz"]:
        return ladder_sorted[0]
    if freq >= ladder_sorted[-1]["frequency_hz"]:
        return ladder_sorted[-1]
    while lo <= hi:
        mid = (lo + hi) // 2
        f_mid = ladder_sorted[mid]["frequency_hz"]
        if f_mid < freq:
            lo = mid + 1
        elif f_mid > freq:
            hi = mid - 1
        else:
            return ladder_sorted[mid]
    candidates = []
    if 0 <= hi < len(ladder_sorted):
        candidates.append(ladder_sorted[hi])
    if 0 <= lo < len(ladder_sorted):
        candidates.append(ladder_sorted[lo])
    best = None
    best_d = float("inf")
    for c in candidates:
        d = abs(freq - c["frequency_hz"])
        if d < best_d:
            best_d = d
            best = c
    return best

def build_ladder_sorted(ladder: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in ladder:
        if "n" not in p or "frequency_hz" not in p:
            continue
        out.append({
            "n": int(p["n"]),
            "frequency_hz": float(p["frequency_hz"]),
            "dim_index": int(p.get("dim_index", 0)),
            "harmonic_cycle": int(p.get("harmonic_cycle", 0)),
        })
    out.sort(key=lambda p: p["frequency_hz"])
    return out

def map_elements_to_base_states(
    element_mapping: Dict[str, Any],
    ladder_sorted: List[Dict[str, Any]],
    base_f: float,
    harmonic_tol: float
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for key, e in element_mapping.items():
        if "frequency_hz" not in e:
            continue
        freq = float(e["frequency_hz"])
        if freq <= 0.0:
            continue
        try:
            Z = int(e.get("Z", int(key)))
        except Exception:
            continue
        slot = nearest_ladder_by_freq(freq, ladder_sorted)
        if slot is None:
            continue
        n = int(slot["n"])
        dim = int(slot["dim_index"])
        cyc = int(slot["harmonic_cycle"])
        deg = degree_from_dim(dim)
        ratio = freq / base_f if base_f > 0.0 else float("nan")
        if math.isfinite(ratio):
            h_est = int(round(ratio))
            h_err = abs(ratio - h_est)
            in_family = h_err <= harmonic_tol
        else:
            h_est = None
            h_err = float("inf")
            in_family = False
        records.append({
            "Z": Z,
            "symbol": e.get("symbol", ""),
            "name": e.get("name", ""),
            "freq": freq,
            "ladder_n": n,
            "dim_index": dim,
            "harmonic_cycle": cyc,
            "degree": deg,
            "ratio": ratio,
            "harmonic_index": h_est,
            "harmonic_err": h_err,
            "in_family": in_family,
        })
    records.sort(key=lambda r: r["Z"])
    return records

def build_state_lattice_for_element(
    base: Dict[str, Any],
    max_charge: int,
    max_iso_offset: int,
    dyadic_weight: float,
    charge_weight: float,
    iso_weight: float,
    polarity_weight: float,
    dim_coupling: float,
    dyadic_indices: List[int]
) -> Dict[str, Any]:
    Z = base["Z"]
    h0 = base["harmonic_index"] if base["harmonic_index"] is not None else 0
    deg0 = base["degree"]
    dim0 = base["dim_index"]
    cyc0 = base["harmonic_cycle"]
    freq0 = base["freq"]
    states: List[Dict[str, Any]] = []
    dyadic_set = set(dyadic_indices)
    for q in range(-max_charge, max_charge + 1):
        for iso in range(-max_iso_offset, max_iso_offset + 1):
            for pol in (-1, 0, 1):
                h_eff = h0 + charge_weight * q + iso_weight * iso + polarity_weight * pol
                nearest_int = int(round(h_eff)) if math.isfinite(h_eff) else 0
                dyadic_nearest = None
                dyadic_dist = float("inf")
                if dyadic_set:
                    for d in dyadic_set:
                        d_dist = abs(h_eff - d)
                        if d_dist < dyadic_dist:
                            dyadic_dist = d_dist
                            dyadic_nearest = d
                deg_eff = (deg0 + pol * (1 if dim0 < 0 else -1 if dim0 > 0 else 0)) % 12
                dim_mix = dim0 + pol * dim_coupling
                dim_score = math.exp(-abs(dim_mix))
                dyadic_score = math.exp(-dyadic_weight * dyadic_dist) if math.isfinite(dyadic_dist) else 0.0
                charge_score = math.exp(-abs(q))
                iso_score = math.exp(-abs(iso))
                stability_score = dyadic_score * dim_score * charge_score * iso_score
                state = {
                    "charge": q,
                    "iso_offset": iso,
                    "polarity": pol,
                    "harmonic_eff": h_eff,
                    "harmonic_nearest_int": nearest_int,
                    "harmonic_nearest_dyadic": dyadic_nearest,
                    "harmonic_dyadic_distance": dyadic_dist,
                    "degree_eff": int(deg_eff),
                    "note_eff": NOTE_NAMES_A_ROOT[int(deg_eff)],
                    "dim_mix": dim_mix,
                    "cycle_base": cyc0,
                    "stability_score": stability_score,
                }
                states.append(state)
    states.sort(key=lambda s: s["stability_score"], reverse=True)
    return {
        "Z": Z,
        "symbol": base["symbol"],
        "name": base["name"],
        "freq": freq0,
        "base_dim_index": dim0,
        "base_degree": deg0,
        "base_note": NOTE_NAMES_A_ROOT[deg0],
        "base_harmonic_index": h0,
        "states": states,
    }

def collect_dyadic_indices(base_records: List[Dict[str, Any]]) -> List[int]:
    hs = set()
    for r in base_records:
        h = r["harmonic_index"]
        if h is None:
            continue
        if h > 0 and is_power_of_two(h):
            hs.add(h)
    if not hs:
        return [1, 2, 4, 8, 16, 32, 64, 128]
    return sorted(hs)

def export_state_lattice(path: str, elements_states: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"elements": elements_states}, f, indent=2)

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict multi-dimensional element, isotope, ion states from harmonic lattice")
    parser.add_argument("--input", default="tuning_solution_12d.json", help="Path to tuning_solution_12d.json")
    parser.add_argument("--output", default="state_lattice_prediction.json", help="Path to export extended state lattice JSON")
    parser.add_argument("--harmonic-tol", type=float, default=0.05, help="Tolerance to assign integer harmonic index to base state")
    parser.add_argument("--max-charge", type=int, default=3, help="Max ionization charge magnitude to consider")
    parser.add_argument("--max-iso-offset", type=int, default=3, help="Max isotope offset index to consider")
    parser.add_argument("--dyadic-weight", type=float, default=0.25, help="Weight for dyadic distance in stability scoring")
    parser.add_argument("--charge-weight", type=float, default=0.5, help="Linear coupling of charge into effective harmonic index")
    parser.add_argument("--iso-weight", type=float, default=0.25, help="Linear coupling of isotope offset into effective harmonic index")
    parser.add_argument("--polarity-weight", type=float, default=0.5, help="Linear coupling of polarity into effective harmonic index")
    parser.add_argument("--dim-coupling", type=float, default=0.5, help="Coupling of polarity into dim_index mixing")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top states per element to show in summary")
    args = parser.parse_args()

    data = load_tuning(args.input)
    base_f = float(data.get("base_frequency", float("nan")))
    element_mapping = data.get("element_mapping", {})
    ladder = data.get("ladder_12d", [])

    if not math.isfinite(base_f):
        print("Base frequency f0* is missing or non-finite in input JSON.")
        sys.exit(1)
    if not element_mapping or not ladder:
        print("Missing element_mapping or ladder_12d in input JSON.")
        sys.exit(1)

    ladder_sorted = build_ladder_sorted(ladder)
    base_records = map_elements_to_base_states(
        element_mapping=element_mapping,
        ladder_sorted=ladder_sorted,
        base_f=base_f,
        harmonic_tol=args.harmonic_tol,
    )
    dyadic_indices = collect_dyadic_indices(base_records)

    print_section("BASE LATTICE SUMMARY")
    print(f"Base frequency f0* (Hz)           : {base_f:.9f}")
    print(f"Total elements mapped             : {len(base_records)}")
    print(f"Dyadic harmonic spine indices     : {dyadic_indices}")

    elements_states: List[Dict[str, Any]] = []
    for base in base_records:
        lattice = build_state_lattice_for_element(
            base=base,
            max_charge=args.max_charge,
            max_iso_offset=args.max_iso_offset,
            dyadic_weight=args.dyadic_weight,
            charge_weight=args.charge_weight,
            iso_weight=args.iso_weight,
            polarity_weight=args.polarity_weight,
            dim_coupling=args.dim_coupling,
            dyadic_indices=dyadic_indices,
        )
        elements_states.append(lattice)

    print_section("SAMPLE STABLE STATES PER ELEMENT")
    for lattice in elements_states[:16]:
        print(f"\nZ={lattice['Z']:3d} {lattice['symbol']:3s} ({lattice['name']}): base h={lattice['base_harmonic_index']}, note={lattice['base_note']}, dim={lattice['base_dim_index']}")
        for s in lattice["states"][:args.top_k]:
            print(
                f"  q={s['charge']:2d} iso={s['iso_offset']:2d} pol={s['polarity']:2d} "
                f"h_eff={s['harmonic_eff']:7.3f} h_dyad={s['harmonic_nearest_dyadic']!s:>3} "
                f"Δh_dyad={s['harmonic_dyadic_distance']:6.3f} "
                f"note={s['note_eff']:2s} dim_mix={s['dim_mix']:6.3f} "
                f"S={s['stability_score']:7.3e}"
            )

    if args.output:
        export_state_lattice(args.output, elements_states)
        print_section("EXPORT")
        print(f"Exported extended state lattice to {args.output}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
