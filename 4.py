#!/usr/bin/env python3
import sys
import json
import math
import argparse
from typing import Dict, Any, List, Optional, Tuple

# ============== IO ==============

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default

def quantiles(values: List[float], qs: List[float]) -> Dict[float, float]:
    if not values:
        return {q: float("nan") for q in qs}
    vs = sorted(values)
    n = len(vs)
    out: Dict[float, float] = {}
    for q in qs:
        if n == 1:
            out[q] = vs[0]
            continue
        idx = q * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            out[q] = vs[lo]
        else:
            frac = idx - lo
            out[q] = vs[lo] * (1.0 - frac) + vs[hi] * frac
    return out

# ============== A_BASE INFERENCE (NO NEGATIVE MASS) ==============

def infer_A_base(el: Dict[str, Any]) -> int:
    raw = el.get("A_base", None)
    try:
        v = int(raw)
        if v > 0:
            return v
    except Exception:
        pass
    try:
        Z = int(el.get("Z", 0))
    except Exception:
        Z = 0
    if Z <= 0:
        raise ValueError(f"Element missing valid Z/A_base: {el}")
    if Z <= 20:
        approx = int(round(2.0 * Z))
    elif Z <= 40:
        approx = int(round(2.1 * Z))
    elif Z <= 82:
        approx = int(round(2.3 * Z))
    else:
        approx = int(round(2.5 * Z))
    if approx < 1:
        approx = Z
    return approx

# ============== REFERENCE ISOTOPE MAPPING ==============

def build_isotope_index(isotopes_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    idx: Dict[int, Dict[str, Any]] = {}
    for el in isotopes_data.get("elements", []):
        try:
            Z = int(el.get("Z"))
        except Exception:
            continue
        iso_list = el.get("isotopes", [])
        if not iso_list:
            continue
        base_iso = None
        best_abund = -1.0
        for iso in iso_list:
            if iso.get("base", False):
                base_iso = iso
                break
        if base_iso is None:
            for iso in iso_list:
                abund = safe_float(iso.get("abundance", 0.0), 0.0)
                if abund > best_abund:
                    best_abund = abund
                    base_iso = iso
        if base_iso is None:
            base_iso = iso_list[0]
        try:
            A_base = int(base_iso.get("A"))
        except Exception:
            continue
        mapped_isos: List[Dict[str, Any]] = []
        for iso in iso_list:
            try:
                A = int(iso.get("A"))
            except Exception:
                continue
            iso_offset = A - A_base
            mapped_isos.append({
                "A": A,
                "iso_offset": iso_offset,
                "symbol": iso.get("symbol", f"{el.get('symbol','')}-{A}"),
                "stable": bool(iso.get("stable", False)),
                "abundance": safe_float(iso.get("abundance", 0.0), 0.0),
                "meta": iso,
            })
        idx[Z] = {
            "Z": Z,
            "symbol": el.get("symbol", ""),
            "name": el.get("name", ""),
            "A_base": A_base,
            "base_iso_symbol": base_iso.get("symbol", f"{el.get('symbol','')}-{A_base}"),
            "isotopes": mapped_isos,
        }
    return idx

# ============== LATTICE UTIL ==============

def extract_element_lattice(elements_states: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    idx: Dict[int, Dict[str, Any]] = {}
    for el in elements_states:
        try:
            Z = int(el.get("Z"))
        except Exception:
            continue
        idx[Z] = el
    return idx

def get_max_score(states: List[Dict[str, Any]]) -> float:
    best = 0.0
    for s in states:
        sc = safe_float(s.get("stability_score", 0.0), 0.0)
        if sc > best:
            best = sc
    return best

def best_state_for_offset(states: List[Dict[str, Any]], iso_offset: int, charge: int = 0) -> Tuple[Optional[Dict[str, Any]], float, int]:
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    rank = -1
    candidates: List[Dict[str, Any]] = []
    for s in states:
        if int(s.get("charge", 0)) != charge:
            continue
        if int(s.get("iso_offset", 0)) != iso_offset:
            continue
        candidates.append(s)
    if not candidates:
        return None, 0.0, -1
    sorted_states = sorted(states, key=lambda s: safe_float(s.get("stability_score", 0.0), 0.0), reverse=True)
    for s in candidates:
        sc = safe_float(s.get("stability_score", 0.0), 0.0)
        if sc > best_score:
            best_score = sc
            best = s
    if best is not None:
        for i, s in enumerate(sorted_states):
            if s is best:
                rank = i + 1
                break
    return best, best_score, rank

# ============== PER-ELEMENT ANALYSIS (REFERENCE MODE) ==============

def analyze_isotopes_for_element(
    Z: int,
    el_lattice: Dict[str, Any],
    iso_info: Dict[str, Any],
    strict_frac: float,
    loose_frac: float
) -> Dict[str, Any]:
    states = el_lattice.get("states", [])
    if not states:
        return {
            "Z": Z,
            "symbol": el_lattice.get("symbol", ""),
            "name": el_lattice.get("name", ""),
            "has_states": False,
            "has_isotopes": True,
            "n_isotopes": len(iso_info.get("isotopes", [])),
        }
    max_score = get_max_score(states)
    iso_results: List[Dict[str, Any]] = []
    n_stable = 0
    n_unstable = 0
    for iso in iso_info.get("isotopes", []):
        iso_offset = int(iso["iso_offset"])
        stable = bool(iso.get("stable", False))
        if stable:
            n_stable += 1
        else:
            n_unstable += 1
        best, best_score, rank = best_state_for_offset(states, iso_offset, charge=0)
        frac = best_score / max_score if max_score > 0.0 else 0.0
        preds_present = best is not None
        iso_results.append({
            "A": iso["A"],
            "iso_offset": iso_offset,
            "symbol": iso["symbol"],
            "stable": stable,
            "abundance": safe_float(iso.get("abundance", 0.0), 0.0),
            "predicted": preds_present,
            "best_score": best_score,
            "stability_frac": frac,
            "rank": rank,
            "pred_state": best,
            "base_note": el_lattice.get("base_note", ""),
            "base_h": el_lattice.get("base_harmonic_index", None),
        })
    n_predicted = sum(1 for r in iso_results if r["predicted"])
    n_stable_predicted = sum(1 for r in iso_results if r["predicted"] and r["stable"])
    n_unstable_predicted = sum(1 for r in iso_results if r["predicted"] and not r["stable"])
    n_stable_strict = sum(1 for r in iso_results if r["stable"] and r["stability_frac"] >= strict_frac)
    n_stable_loose = sum(1 for r in iso_results if r["stable"] and r["stability_frac"] >= loose_frac)
    n_unstable_strict = sum(1 for r in iso_results if (not r["stable"]) and r["stability_frac"] >= strict_frac)
    n_unstable_loose = sum(1 for r in iso_results if (not r["stable"]) and r["stability_frac"] >= loose_frac)
    return {
        "Z": Z,
        "symbol": el_lattice.get("symbol", ""),
        "name": el_lattice.get("name", ""),
        "A_base": iso_info.get("A_base"),
        "base_iso_symbol": iso_info.get("base_iso_symbol", ""),
        "has_states": True,
        "has_isotopes": True,
        "n_isotopes": len(iso_results),
        "n_stable": n_stable,
        "n_unstable": n_unstable,
        "n_predicted": n_predicted,
        "n_stable_predicted": n_stable_predicted,
        "n_unstable_predicted": n_unstable_predicted,
        "n_stable_strict": n_stable_strict,
        "n_stable_loose": n_stable_loose,
        "n_unstable_strict": n_unstable_strict,
        "n_unstable_loose": n_unstable_loose,
        "isotopes": iso_results,
    }

# ============== SYSTEM-LEVEL ANALYSIS (REFERENCE MODE) ==============

def analyze_isotope_system(
    lattice_data: Dict[str, Any],
    isotopes_data: Dict[str, Any],
    strict_frac: float,
    loose_frac: float
) -> Dict[str, Any]:
    elements_states = lattice_data.get("elements", [])
    el_idx = extract_element_lattice(elements_states)
    iso_idx = build_isotope_index(isotopes_data)
    per_element: List[Dict[str, Any]] = []
    for Z, iso_info in sorted(iso_idx.items()):
        if Z not in el_idx:
            per_element.append({
                "Z": Z,
                "symbol": iso_info.get("symbol", ""),
                "name": iso_info.get("name", ""),
                "A_base": iso_info.get("A_base"),
                "base_iso_symbol": iso_info.get("base_iso_symbol", ""),
                "has_states": False,
                "has_isotopes": True,
                "n_isotopes": len(iso_info.get("isotopes", [])),
                "isotopes": [],
            })
            continue
        el_lattice = el_idx[Z]
        per_element.append(
            analyze_isotopes_for_element(
                Z, el_lattice, iso_info,
                strict_frac=strict_frac,
                loose_frac=loose_frac,
            )
        )
    stable_fracs: List[float] = []
    unstable_fracs: List[float] = []
    stable_ranks: List[int] = []
    unstable_ranks: List[int] = []
    stable_top1 = 0
    stable_top3 = 0
    stable_top5 = 0
    total_stable = 0
    for el in per_element:
        if not el.get("has_states", False) or not el.get("has_isotopes", False):
            continue
        for iso in el.get("isotopes", []):
            if not iso["predicted"]:
                continue
            rank = iso["rank"]
            frac = iso["stability_frac"]
            if iso["stable"]:
                total_stable += 1
                stable_fracs.append(frac)
                if rank > 0:
                    stable_ranks.append(rank)
                if rank == 1:
                    stable_top1 += 1
                if 1 <= rank <= 3:
                    stable_top3 += 1
                if 1 <= rank <= 5:
                    stable_top5 += 1
            else:
                unstable_fracs.append(frac)
                if rank > 0:
                    unstable_ranks.append(rank)
    q_stable_frac = quantiles(stable_fracs, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]) if stable_fracs else {}
    q_unstable_frac = quantiles(unstable_fracs, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]) if unstable_fracs else {}
    q_stable_rank = quantiles([float(r) for r in stable_ranks], [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]) if stable_ranks else {}
    q_unstable_rank = quantiles([float(r) for r in unstable_ranks], [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]) if unstable_ranks else {}
    return {
        "per_element": per_element,
        "summary": {
            "n_elements_with_isotopes": len(per_element),
            "total_stable_iso": total_stable,
            "stable_frac_quantiles": q_stable_frac,
            "unstable_frac_quantiles": q_unstable_frac,
            "stable_rank_quantiles": q_stable_rank,
            "unstable_rank_quantiles": q_unstable_rank,
            "stable_top1": stable_top1,
            "stable_top3": stable_top3,
            "stable_top5": stable_top5,
            "stable_top1_frac": stable_top1 / total_stable if total_stable > 0 else float("nan"),
            "stable_top3_frac": stable_top3 / total_stable if total_stable > 0 else float("nan"),
            "stable_top5_frac": stable_top5 / total_stable if total_stable > 0 else float("nan"),
        },
    }

# ============== PREDICTED ISOTOPE ENUMERATION (NO REFERENCE) ==============

def build_predicted_isotope_table(
    lattice_data: Dict[str, Any],
    stable_threshold: float,
    min_frac: float
) -> Dict[str, Any]:
    elements_out: List[Dict[str, Any]] = []
    for el in lattice_data.get("elements", []):
        try:
            Z = int(el.get("Z"))
        except Exception:
            continue
        symbol = el.get("symbol", "")
        name = el.get("name", "")
        A_base = infer_A_base(el)
        states_all = el.get("states", [])
        states = [s for s in states_all if int(s.get("charge", 0)) == 0]
        if not states:
            continue
        max_score = get_max_score(states)
        if max_score <= 0.0:
            continue
        sorted_states = sorted(states, key=lambda s: safe_float(s.get("stability_score", 0.0), 0.0), reverse=True)
        by_offset: Dict[int, Dict[str, Any]] = {}
        for i, s in enumerate(sorted_states):
            iso_offset = int(s.get("iso_offset", 0))
            sc = safe_float(s.get("stability_score", 0.0), 0.0)
            if sc <= 0.0:
                continue
            frac = sc / max_score
            if frac < min_frac:
                continue
            rank = i + 1
            prev = by_offset.get(iso_offset)
            if prev is None or sc > prev["best_score"]:
                by_offset[iso_offset] = {
                    "state": s,
                    "best_score": sc,
                    "frac": frac,
                    "rank": rank,
                }
        if not by_offset:
            continue
        sum_scores = sum(v["best_score"] for v in by_offset.values())
        if sum_scores <= 0.0:
            continue
        iso_list: List[Dict[str, Any]] = []
        for iso_offset in sorted(by_offset.keys()):
            entry = by_offset[iso_offset]
            s = entry["state"]
            sc = entry["best_score"]
            frac = entry["frac"]
            rank = entry["rank"]
            abundance = sc / sum_scores
            stable = frac >= stable_threshold
            A = A_base + iso_offset
            if A < 1:
                continue
            iso_symbol = f"{symbol}-{A}"
            iso_list.append({
                "A": A,
                "iso_offset": iso_offset,
                "symbol": iso_symbol,
                "base": bool(iso_offset == 0),
                "stable": stable,
                "abundance": abundance,
                "stability_frac": frac,
                "stability_score": sc,
                "rank": rank,
                "note_eff": s.get("note_eff", ""),
                "harmonic_eff": safe_float(s.get("harmonic_eff", 0.0), 0.0),
                "dim_mix": safe_float(s.get("dim_mix", 0.0), 0.0),
            })
        if not iso_list:
            continue
        elements_out.append({
            "Z": Z,
            "symbol": symbol,
            "name": name,
            "A_base": A_base,
            "base_iso_symbol": f"{symbol}-{A_base}",
            "isotopes": iso_list,
        })
    return {"elements": elements_out}

# ============== PRINTING ==============

def print_global_isotope_summary(summary: Dict[str, Any]) -> None:
    print_section("ISOTOPE COHERENCE SUMMARY")
    print(f"Elements with isotope reference    : {summary.get('n_elements_with_isotopes', 0)}")
    print(f"Total stable isotopes accounted    : {summary.get('total_stable_iso', 0)}")
    print()
    print("Stable isotope stability_frac quantiles (best_score / max_score):")
    for q in sorted(summary.get("stable_frac_quantiles", {}).keys()):
        v = summary["stable_frac_quantiles"][q]
        print(f"  q={q:4.2f}: {v:7.3f}")
    print()
    print("Unstable isotope stability_frac quantiles (if present):")
    for q in sorted(summary.get("unstable_frac_quantiles", {}).keys()):
        v = summary["unstable_frac_quantiles"][q]
        print(f"  q={q:4.2f}: {v:7.3f}")
    print()
    print("Stable isotope rank quantiles (1 = most stable state in lattice):")
    for q in sorted(summary.get("stable_rank_quantiles", {}).keys()):
        v = summary["stable_rank_quantiles"][q]
        print(f"  q={q:4.2f}: {v:7.3f}")
    print()
    print("Unstable isotope rank quantiles (if present):")
    for q in sorted(summary.get("unstable_rank_quantiles", {}).keys()):
        v = summary["unstable_rank_quantiles"][q]
        print(f"  q={q:4.2f}: {v:7.3f}")
    print()
    print("Stable isotopes vs top-k predicted states:")
    print(f"  Fraction of stable isotopes with rank=1     : {summary.get('stable_top1_frac', float('nan')):.3f}")
    print(f"  Fraction of stable isotopes in top-3 states : {summary.get('stable_top3_frac', float('nan')):.3f}")
    print(f"  Fraction of stable isotopes in top-5 states : {summary.get('stable_top5_frac', float('nan')):.3f}")

def print_element_isotope_sample(per_element: List[Dict[str, Any]], limit_elements: int = 8, limit_isos: int = 8) -> None:
    print_section("ELEMENT / ISOTOPE SAMPLE VIEW")
    shown = 0
    for el in per_element:
        if shown >= limit_elements:
            break
        if not el.get("has_isotopes", False):
            continue
        iso_list = el.get("isotopes", [])
        if not iso_list:
            continue
        print()
        print(f"Z={el['Z']:3d} {el['symbol']:3s} ({el.get('name','')}) base={el.get('base_iso_symbol','')} A_base={el.get('A_base')}")
        header = "  A   iso_off  stable  abund    predicted  rank  frac    note_eff  h_eff     dim_mix"
        print(header)
        print("-" * len(header))
        count = 0
        for iso in sorted(iso_list, key=lambda r: (not r["stable"], -r.get("abundance", 0.0))):
            if count >= limit_isos:
                break
            count += 1
            s = iso.get("pred_state", None)
            note_eff = s.get("note_eff", "") if s else ""
            h_eff = safe_float(s.get("harmonic_eff", 0.0), 0.0) if s else 0.0
            dim_mix = safe_float(s.get("dim_mix", 0.0), 0.0) if s else 0.0
            print(
                f"  {iso['A']:3d}  {iso['iso_offset']:7d}  "
                f"{'T' if iso['stable'] else 'F':6s}  "
                f"{iso['abundance']:6.3f}  "
                f"{'T' if iso.get('predicted', True) else 'F':9s}  "
                f"{iso.get('rank', 0):4d}  "
                f"{iso.get('stability_frac', 0.0):5.3f}  "
                f"{note_eff:8s}  "
                f"{h_eff:7.3f}  "
                f"{dim_mix:8.3f}"
            )
        shown += 1

# ============== MAIN ==============

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harmonic lattice isotope tools: compare against reference or export predicted isotopes"
    )
    parser.add_argument(
        "--lattice",
        default="state_lattice_prediction.json",
        help="Path to state_lattice_prediction.json"
    )
    parser.add_argument(
        "--isotopes",
        default="isotopes_reference.json",
        help="Path to isotopes_reference.json (for comparison mode)"
    )
    parser.add_argument(
        "--strict-frac",
        type=float,
        default=0.9,
        help="Strict threshold on stability_frac for a predicted isotope to be considered strongly supported (comparison mode)"
    )
    parser.add_argument(
        "--loose-frac",
        type=float,
        default=0.5,
        help="Loose threshold on stability_frac for a predicted isotope to be considered supported (comparison mode)"
    )
    parser.add_argument(
        "--sample-elements",
        type=int,
        default=8,
        help="Number of elements to show in the sample table (comparison mode)"
    )
    parser.add_argument(
        "--sample-isos",
        type=int,
        default=8,
        help="Number of isotopes per element to show in the sample table (comparison mode)"
    )
    parser.add_argument(
        "--export-predicted",
        type=str,
        default=None,
        help="If set, export predicted isotopes from the lattice to this JSON file and exit"
    )
    parser.add_argument(
        "--stable-threshold",
        type=float,
        default=0.5,
        help="Threshold on stability_frac to mark an isotope as stable in export mode"
    )
    parser.add_argument(
        "--min-frac",
        type=float,
        default=0.0,
        help="Minimum stability_frac for a state to be included as an isotope in export mode"
    )
    args = parser.parse_args()
    try:
        lattice_data = load_json(args.lattice)
    except FileNotFoundError:
        print(f"Error: lattice file not found: {args.lattice}")
        sys.exit(1)
    if args.export_predicted:
        predicted = build_predicted_isotope_table(
            lattice_data=lattice_data,
            stable_threshold=args.stable_threshold,
            min_frac=args.min_frac,
        )
        try:
            with open(args.export_predicted, "w", encoding="utf-8") as f:
                json.dump(predicted, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error: failed to write predicted isotope JSON: {e}")
            sys.exit(1)
        print(f"Exported predicted isotopes to {args.export_predicted}")
        return
    try:
        isotopes_data = load_json(args.isotopes)
    except FileNotFoundError:
        print(f"Error: isotope reference file not found: {args.isotopes}")
        sys.exit(1)
    result = analyze_isotope_system(
        lattice_data=lattice_data,
        isotopes_data=isotopes_data,
        strict_frac=args.strict_frac,
        loose_frac=args.loose_frac,
    )
    print_global_isotope_summary(result["summary"])
    print_element_isotope_sample(
        result["per_element"],
        limit_elements=args.sample_elements,
        limit_isos=args.sample_isos,
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
