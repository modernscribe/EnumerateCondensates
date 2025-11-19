#!/usr/bin/env python3
import json
import math
import os
import sys
from typing import List, Dict, Tuple

import numpy as np

N_DIM = 12
PRINCIPLES = ["Truth", "Purity", "Law", "Love", "Wisdom", "Life", "Glory"]
EPS = 1e-12

MAJOR_DEGREES = [0, 2, 4, 5, 7, 9, 11]
SHARP_DEGREES = [1, 3, 6, 8, 10]

NOTE_NAMES_A_ROOT = [
    "A", "A#", "B", "C", "C#", "D",
    "D#", "E", "F", "F#", "G", "G#"
]

ELEMENTS = [
    ("H", "Hydrogen"),
    ("He", "Helium"),
    ("Li", "Lithium"),
    ("Be", "Beryllium"),
    ("B", "Boron"),
    ("C", "Carbon"),
    ("N", "Nitrogen"),
    ("O", "Oxygen"),
    ("F", "Fluorine"),
    ("Ne", "Neon"),
    ("Na", "Sodium"),
    ("Mg", "Magnesium"),
    ("Al", "Aluminium"),
    ("Si", "Silicon"),
    ("P", "Phosphorus"),
    ("S", "Sulfur"),
    ("Cl", "Chlorine"),
    ("Ar", "Argon"),
    ("K", "Potassium"),
    ("Ca", "Calcium"),
    ("Sc", "Scandium"),
    ("Ti", "Titanium"),
    ("V", "Vanadium"),
    ("Cr", "Chromium"),
    ("Mn", "Manganese"),
    ("Fe", "Iron"),
    ("Co", "Cobalt"),
    ("Ni", "Nickel"),
    ("Cu", "Copper"),
    ("Zn", "Zinc"),
    ("Ga", "Gallium"),
    ("Ge", "Germanium"),
    ("As", "Arsenic"),
    ("Se", "Selenium"),
    ("Br", "Bromine"),
    ("Kr", "Krypton"),
    ("Rb", "Rubidium"),
    ("Sr", "Strontium"),
    ("Y", "Yttrium"),
    ("Zr", "Zirconium"),
    ("Nb", "Niobium"),
    ("Mo", "Molybdenum"),
    ("Tc", "Technetium"),
    ("Ru", "Ruthenium"),
    ("Rh", "Rhodium"),
    ("Pd", "Palladium"),
    ("Ag", "Silver"),
    ("Cd", "Cadmium"),
    ("In", "Indium"),
    ("Sn", "Tin"),
    ("Sb", "Antimony"),
    ("Te", "Tellurium"),
    ("I", "Iodine"),
    ("Xe", "Xenon"),
    ("Cs", "Caesium"),
    ("Ba", "Barium"),
    ("La", "Lanthanum"),
    ("Ce", "Cerium"),
    ("Pr", "Praseodymium"),
    ("Nd", "Neodymium"),
    ("Pm", "Promethium"),
    ("Sm", "Samarium"),
    ("Eu", "Europium"),
    ("Gd", "Gadolinium"),
    ("Tb", "Terbium"),
    ("Dy", "Dysprosium"),
    ("Ho", "Holmium"),
    ("Er", "Erbium"),
    ("Tm", "Thulium"),
    ("Yb", "Ytterbium"),
    ("Lu", "Lutetium"),
    ("Hf", "Hafnium"),
    ("Ta", "Tantalum"),
    ("W", "Tungsten"),
    ("Re", "Rhenium"),
    ("Os", "Osmium"),
    ("Ir", "Iridium"),
    ("Pt", "Platinum"),
    ("Au", "Gold"),
    ("Hg", "Mercury"),
    ("Tl", "Thallium"),
    ("Pb", "Lead"),
    ("Bi", "Bismuth"),
    ("Po", "Polonium"),
    ("At", "Astatine"),
    ("Rn", "Radon"),
    ("Fr", "Francium"),
    ("Ra", "Radium"),
    ("Ac", "Actinium"),
    ("Th", "Thorium"),
    ("Pa", "Protactinium"),
    ("U", "Uranium"),
    ("Np", "Neptunium"),
    ("Pu", "Plutonium"),
    ("Am", "Americium"),
    ("Cm", "Curium"),
    ("Bk", "Berkelium"),
    ("Cf", "Californium"),
    ("Es", "Einsteinium"),
    ("Fm", "Fermium"),
    ("Md", "Mendelevium"),
    ("No", "Nobelium"),
    ("Lr", "Lawrencium"),
    ("Rf", "Rutherfordium"),
    ("Db", "Dubnium"),
    ("Sg", "Seaborgium"),
    ("Bh", "Bohrium"),
    ("Hs", "Hassium"),
    ("Mt", "Meitnerium"),
    ("Ds", "Darmstadtium"),
    ("Rg", "Roentgenium"),
    ("Cn", "Copernicium"),
    ("Nh", "Nihonium"),
    ("Fl", "Flerovium"),
    ("Mc", "Moscovium"),
    ("Lv", "Livermorium"),
    ("Ts", "Tennessine"),
    ("Og", "Oganesson"),
]

C_LIGHT = 2.99792458e8
H_PLANCK = 6.62607015e-34
E_CHARGE = 1.602176634e-19

def Null_point() -> float:
    return 0.0

def InverseZero_operator(p: float) -> float:
    return -0.0 if p == 0.0 else p

def H_inverse_zero_null() -> float:
    return 1.0

def sigma_selector(omega: float) -> float:
    return 1.0

def f_truth(x: np.ndarray) -> np.ndarray:
    return x.copy()

def f_purity(x: np.ndarray) -> np.ndarray:
    abs_x = np.abs(x)
    s = np.sum(abs_x)
    if s < EPS:
        return np.full_like(x, Null_point())
    return abs_x / s

def f_law(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0)

def f_love(x: np.ndarray) -> np.ndarray:
    m = np.mean(x)
    result = (x + m) * 0.5
    mask = np.abs(result) < EPS
    result[mask] = InverseZero_operator(0.0)
    return result

def f_wisdom(x: np.ndarray) -> np.ndarray:
    left = np.roll(x, 1)
    right = np.roll(x, -1)
    return (x + 0.5 * (left + right)) * 0.5

def f_life(x: np.ndarray) -> np.ndarray:
    base = np.tanh(x)
    zeta = H_inverse_zero_null()
    return base * zeta

def f_glory(x: np.ndarray) -> np.ndarray:
    v = np.where(x >= 0, x * x, -x * x)
    s = np.sum(np.abs(v))
    if s < EPS:
        return np.full_like(x, Null_point())
    return v / s

PRINCIPLE_MAP = {
    "Truth": f_truth,
    "Purity": f_purity,
    "Law": f_law,
    "Love": f_love,
    "Wisdom": f_wisdom,
    "Life": f_life,
    "Glory": f_glory,
}

def apply_principle(name: str, x: np.ndarray) -> np.ndarray:
    return PRINCIPLE_MAP[name](x)

def steps_for_temperament(temperament: str) -> int:
    if temperament in ("equal", "12tet"):
        return 12
    if temperament == "11tet":
        return 11
    return 12

def omega_k(k: int, f0: float, temperament: str = "equal") -> float:
    steps = steps_for_temperament(temperament)
    ratio = 2.0 ** (k / float(steps))
    return 2.0 * math.pi * f0 * ratio

def compute_holonomy_for_f0(fp: np.ndarray, enabled: np.ndarray, f0: float, temperament: str = "equal") -> Dict:
    holonomies: Dict[str, Dict] = {}
    for name in PRINCIPLES:
        vec = apply_principle(name, fp)
        phases: List[float] = []
        for k in range(N_DIM):
            if enabled[k]:
                omega = omega_k(k, f0, temperament)
                phase = omega * float(vec[k])
                phases.append(phase)
        if phases:
            total_phase = float(np.sum(phases))
            hol = complex(math.cos(total_phase), math.sin(total_phase))
        else:
            total_phase = 0.0
            hol = complex(1.0, 0.0)
        holonomies[name] = {
            "phase": float(total_phase),
            "holonomy": [float(hol.real), float(hol.imag)],
            "magnitude": float(abs(hol)),
        }
    crown = complex(1.0, 0.0)
    for h in holonomies.values():
        c = complex(h["holonomy"][0], h["holonomy"][1])
        crown *= c
    crown_mag = float(abs(crown))
    return {
        "per_principle": holonomies,
        "crown": {
            "real": float(crown.real),
            "imag": float(crown.imag),
            "magnitude": crown_mag,
            "closure": float(abs(crown_mag - 1.0)),
        },
    }

def load_samples(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    samples: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fp_vec = np.array(obj["fixed_point"]["vector"], dtype=float)
            is_null = bool(obj["fixed_point"].get("is_null", False))
            enabled_raw = obj.get("enabled_tones")
            if enabled_raw is None:
                enabled_raw = [True] * N_DIM
            enabled = np.array([bool(b) for b in enabled_raw], dtype=bool)
            if fp_vec.shape[0] != N_DIM or enabled.shape[0] != N_DIM:
                continue
            if is_null:
                continue
            samples.append({
                "id": obj.get("id", ""),
                "fp": fp_vec,
                "enabled": enabled,
            })
    if not samples:
        raise RuntimeError("No usable non-null samples found in input file.")
    return samples

def score_f0(
    f0: float,
    samples: List[Dict],
    temperament: str = "equal",
    closure_weight: float = 10.0
) -> Tuple[float, float, float]:
    phases: List[float] = []
    closures: List[float] = []
    for s in samples:
        fp = s["fp"]
        enabled = s["enabled"]
        hol = compute_holonomy_for_f0(fp, enabled, f0, temperament)
        crown = hol["crown"]
        mag = float(crown["magnitude"])
        closure = float(abs(mag - 1.0))
        phase = math.atan2(crown["imag"], crown["real"])
        phases.append(phase)
        closures.append(closure)
    if not phases:
        return -math.inf, math.nan, math.nan
    phase_arr = np.unwrap(np.array(phases, dtype=float))
    phase_var = float(np.var(phase_arr))
    closure_mean = float(np.mean(np.array(closures, dtype=float)))
    score = -phase_var - closure_weight * closure_mean
    return score, phase_var, closure_mean

def sweep_f0(
    samples: List[Dict],
    f0_min: float,
    f0_max: float,
    steps_sweep: int,
    temperament: str,
    closure_weight: float
) -> Dict:
    f0_values = np.linspace(f0_min, f0_max, steps_sweep, dtype=float)
    best = {
        "f0": None,
        "score": -math.inf,
        "phase_var": None,
        "closure_mean": None,
    }
    for idx, f0 in enumerate(f0_values):
        score, phase_var, closure_mean = score_f0(float(f0), samples, temperament, closure_weight)
        if score > best["score"]:
            best["f0"] = float(f0)
            best["score"] = float(score)
            best["phase_var"] = float(phase_var)
            best["closure_mean"] = float(closure_mean)
        if steps_sweep >= 20 and (idx + 1) % max(1, steps_sweep // 20) == 0:
            pct = 100.0 * (idx + 1) / steps_sweep
            print(f"[SWEEP] {idx+1}/{steps_sweep} ({pct:5.1f}%) f0={f0:8.3f} Hz score={score: .6e}")
    return best

def refine_f0(
    samples: List[Dict],
    f0_min: float,
    f0_max: float,
    steps_sweep: int,
    temperament: str,
    closure_weight: float,
    target_step: float = 0.1,
    max_rounds: int = 6
) -> Dict:
    current_min = f0_min
    current_max = f0_max
    current_steps = steps_sweep
    best_overall: Dict = {
        "f0": None,
        "score": -math.inf,
        "phase_var": None,
        "closure_mean": None,
    }
    for rnd in range(max_rounds):
        span = current_max - current_min
        if span <= 0.0:
            break
        step_size = span / max(1, current_steps - 1)
        print(
            f"\n[REFINE] round {rnd+1}: "
            f"range={current_min:.6f}-{current_max:.6f} Hz "
            f"(step≈{step_size:.4f} Hz, steps={current_steps})"
        )
        best = sweep_f0(samples, current_min, current_max, current_steps, temperament, closure_weight)
        if best["f0"] is None:
            break
        if best["score"] > best_overall["score"]:
            best_overall = {
                "f0": best["f0"],
                "score": best["score"],
                "phase_var": best["phase_var"],
                "closure_mean": best["closure_mean"],
            }
        span = current_max - current_min
        step_size = span / max(1, current_steps - 1)
        if step_size <= target_step:
            print(f"[REFINE] target resolution reached (step≈{step_size:.4f} Hz)")
            break
        new_span = max(span * 0.2, target_step * 5.0)
        half = new_span / 2.0
        center = best["f0"]
        new_min = max(f0_min, center - half)
        new_max = min(f0_max, center + half)
        if new_max <= new_min:
            new_min = max(f0_min, center - target_step / 2.0)
            new_max = min(f0_max, center + target_step / 2.0)
        current_min = new_min
        current_max = new_max
        span = current_max - current_min
        current_steps = max(11, int(span / target_step) + 1)
    return best_overall

def evaluate_references(
    samples: List[Dict],
    refs: List[float],
    temperament: str,
    closure_weight: float
) -> List[Dict]:
    rows: List[Dict] = []
    for f0 in refs:
        score, phase_var, closure_mean = score_f0(f0, samples, temperament, closure_weight)
        rows.append({
            "f0": float(f0),
            "score": float(score),
            "phase_var": float(phase_var),
            "closure_mean": float(closure_mean),
        })
    return rows

def classify_em_band(freq: float) -> str:
    if freq <= 0.0:
        return "static"
    if freq < 3e3:
        return "ELF/ULF"
    if freq < 3e6:
        return "radio_LF_MF"
    if freq < 3e8:
        return "radio_HF_VHF"
    if freq < 3e11:
        return "microwave"
    if freq < 4e14:
        return "infrared"
    if freq < 7.5e14:
        return "visible"
    if freq < 3e16:
        return "ultraviolet"
    if freq < 3e19:
        return "xray"
    return "gamma"

def build_octave_frequencies(f0: float, temperament: str = "equal") -> Dict:
    steps = steps_for_temperament(temperament)
    chromatic = [float(f0 * (2.0 ** (k / float(steps)))) for k in range(12)]
    majors: Dict[str, float] = {}
    for name, deg in zip(PRINCIPLES, MAJOR_DEGREES):
        majors[name] = chromatic[deg]
    harmonics: Dict[str, float] = {}
    for i, deg in enumerate(SHARP_DEGREES):
        harmonics[f"H{i+1}"] = chromatic[deg]
    return {
        "chromatic": chromatic,
        "principles_major": majors,
        "harmonics_sharp": harmonics,
    }

def build_multi_octave_table(f0: float, octaves: Tuple[int, int] = (-1, 1), temperament: str = "equal") -> Dict:
    steps = steps_for_temperament(temperament)
    lo, hi = octaves
    table: Dict[str, Dict[str, float]] = {}
    for o in range(lo, hi + 1):
        base = f0 * (2.0 ** o)
        octave_key = f"oct_{o:+d}"
        chrom = [float(base * (2.0 ** (k / float(steps)))) for k in range(12)]
        octave_data: Dict[str, float] = {}
        for idx, freq in enumerate(chrom):
            name = NOTE_NAMES_A_ROOT[idx]
            label = f"{name}{4+o}"
            octave_data[label] = freq
        table[octave_key] = octave_data
    return table

def build_element_mapping(f0: float, max_Z: int = 118, temperament: str = "equal") -> Dict[str, Dict[str, object]]:
    steps = steps_for_temperament(temperament)
    mapping: Dict[str, Dict[str, object]] = {}
    max_Z = min(max_Z, len(ELEMENTS))
    for Z in range(1, max_Z + 1):
        idx = Z - 1
        symbol, name = ELEMENTS[idx]
        shifted = idx - 5
        degree = int(shifted % steps)
        cycle = math.floor(shifted / float(steps))
        note_oct = 4 + cycle
        oct_shift = note_oct - 4
        ratio_oct = 2.0 ** oct_shift
        ratio_degree = 2.0 ** (degree / float(steps))
        freq = float(f0 * ratio_oct * ratio_degree)
        note_name = NOTE_NAMES_A_ROOT[degree]
        label = f"{note_name}{note_oct}"
        wavelength = float(C_LIGHT / freq) if freq > 0.0 else float("inf")
        energy_eV = float(H_PLANCK * freq / E_CHARGE) if freq > 0.0 else 0.0
        band = classify_em_band(freq)
        mapping[str(Z)] = {
            "Z": Z,
            "symbol": symbol,
            "name": name,
            "degree": degree,
            "note": note_name,
            "octave": note_oct,
            "label": label,
            "frequency_hz": freq,
            "wavelength_m": wavelength,
            "energy_eV": energy_eV,
            "band": band,
        }
    group14 = [6, 14, 32, 50, 82, 114]
    for i, Z in enumerate(group14):
        if 1 <= Z <= max_Z:
            key = str(Z)
            base = mapping[key]
            symbol = base["symbol"]
            name = base["name"]
            octave = 4 + i
            freq = float(f0 * (2.0 ** i))
            wavelength = float(C_LIGHT / freq) if freq > 0.0 else float("inf")
            energy_eV = float(H_PLANCK * freq / E_CHARGE) if freq > 0.0 else 0.0
            band = classify_em_band(freq)
            mapping[key] = {
                "Z": Z,
                "symbol": symbol,
                "name": name,
                "degree": 0,
                "note": "A",
                "octave": octave,
                "label": f"A{octave}",
                "frequency_hz": freq,
                "wavelength_m": wavelength,
                "energy_eV": energy_eV,
                "band": band,
            }
    return mapping

def build_12d_ladder(
    f0: float,
    n_min: int = -240,
    n_max: int = 336,
    temperament: str = "equal"
) -> List[Dict[str, object]]:
    steps = steps_for_temperament(temperament)
    ladder: List[Dict[str, object]] = []
    for n in range(n_min, n_max + 1):
        ratio = 2.0 ** (n / float(steps))
        freq = float(f0 * ratio)
        band = classify_em_band(freq)
        wavelength = float(C_LIGHT / freq) if freq > 0.0 else float("inf")
        energy_eV = float(H_PLANCK * freq / E_CHARGE) if freq > 0.0 else 0.0
        note_degree = int(n % 12)
        octave_index = int(n // 12)
        note_name = NOTE_NAMES_A_ROOT[note_degree]
        label = f"{note_name}_{octave_index}"
        ladder.append({
            "n": int(n),
            "ratio": ratio,
            "frequency_hz": freq,
            "log10_freq": float(math.log10(freq)) if freq > 0.0 else float("-inf"),
            "wavelength_m": wavelength,
            "energy_eV": energy_eV,
            "band": band,
            "degree": int(note_degree),
            "octave_index": int(octave_index),
            "label": label,
        })
    return ladder

def find_visible_center(ladder: List[Dict[str, object]]) -> Dict[str, object]:
    visible_points = [p for p in ladder if p["band"] == "visible"]
    if not visible_points:
        return {}
    f_min = min(p["frequency_hz"] for p in visible_points)
    f_max = max(p["frequency_hz"] for p in visible_points)
    f_mid = math.sqrt(f_min * f_max)
    center_point = min(visible_points, key=lambda p: abs(p["frequency_hz"] - f_mid))
    center = dict(center_point)
    center["dim_index"] = 0
    center["harmonic_cycle"] = 0
    return center

def reindex_ladder_dimensions(
    ladder: List[Dict[str, object]],
    n_center: int
) -> None:
    for p in ladder:
        offset = p["n"] - n_center
        base_dim = ((offset + 6) % 12) - 6
        cycle = int(round((offset - base_dim) / 12.0))
        p["dim_index"] = base_dim
        p["harmonic_cycle"] = cycle

def scan_coalescence_over_ladder(
    samples: List[Dict],
    ladder: List[Dict[str, object]],
    temperament: str,
    closure_weight: float,
    f_min: float = 1.0,
    top_k: int = 32
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    sorted_ladder = sorted(ladder, key=lambda p: p["frequency_hz"], reverse=True)
    for p in sorted_ladder:
        f0 = p["frequency_hz"]
        if f0 < f_min:
            break
        score, phase_var, closure_mean = score_f0(f0, samples, temperament, closure_weight)
        results.append({
            "n": p["n"],
            "dim_index": p.get("dim_index", 0),
            "harmonic_cycle": p.get("harmonic_cycle", 0),
            "label": p["label"],
            "frequency_hz": f0,
            "score": score,
            "phase_var": phase_var,
            "closure_mean": closure_mean,
        })
    results_sorted = sorted(results, key=lambda r: r["score"], reverse=True)
    if top_k > 0 and len(results_sorted) > top_k:
        results_sorted = results_sorted[:top_k]
    return results_sorted

def export_solution_json(
    path: str,
    best: Dict,
    octave: Dict,
    multi_oct: Dict,
    elem_map: Dict[str, Dict[str, object]],
    ladder: List[Dict[str, object]],
    visible_center: Dict[str, object],
    coalescence_points: List[Dict[str, object]]
) -> None:
    data = {
        "base_frequency": float(best["f0"]),
        "score": float(best["score"]),
        "phase_var": float(best["phase_var"]),
        "closure_mean": float(best["closure_mean"]),
        "chromatic_1oct": octave["chromatic"],
        "principles_major": octave["principles_major"],
        "harmonics_sharp": octave["harmonics_sharp"],
        "multi_octave": multi_oct,
        "element_mapping": elem_map,
        "ladder_12d": ladder,
        "visible_center": visible_center,
        "coalescence_points": coalescence_points,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Harmonic tuning solver from holonomy atlas with elemental + 12D ladder mapping"
    )
    parser.add_argument("--input", default="harmonic_exploration_enhanced.jsonl", help="Path to exploration JSONL file")
    parser.add_argument("--f0-min", type=float, default=200.0, help="Minimum base frequency (Hz)")
    parser.add_argument("--f0-max", type=float, default=600.0, help="Maximum base frequency (Hz)")
    parser.add_argument("--steps", type=int, default=401, help="Number of f0 samples in initial sweep")
    parser.add_argument("--temperament", choices=["equal", "11tet"], default="equal", help="Temperament model")
    parser.add_argument("--closure-weight", type=float, default=10.0, help="Weight on crown magnitude closure in score")
    parser.add_argument("--target-step", type=float, default=0.1, help="Target frequency resolution (Hz) for refinement")
    parser.add_argument("--max-rounds", type=int, default=6, help="Maximum refinement rounds")
    parser.add_argument("--no-refs", action="store_true", help="Disable evaluation of reference base frequencies")
    parser.add_argument("--export-json", default="tuning_solution_12d.json", help="Path to export JSON tuning table")
    parser.add_argument("--max-Z", type=int, default=118, help="Maximum atomic number to map")
    parser.add_argument("--n-min", type=int, default=-600, help="Minimum ladder index n")
    parser.add_argument("--n-max", type=int, default=1600, help="Maximum ladder index n")
    parser.add_argument("--coalesce-fmin", type=float, default=1.0, help="Minimum f0 for global coalescence scan (Hz)")
    parser.add_argument("--coalesce-topk", type=int, default=32, help="Top K coalescence points to retain")
    args = parser.parse_args()

    print("Loading samples...")
    samples = load_samples(args.input)
    print(f"Loaded {len(samples)} non-null samples")

    print("\nRunning adaptive sweep for base frequency f0...")
    best = refine_f0(
        samples=samples,
        f0_min=args.f0_min,
        f0_max=args.f0_max,
        steps_sweep=args.steps,
        temperament=args.temperament,
        closure_weight=args.closure_weight,
        target_step=args.target_step,
        max_rounds=args.max_rounds,
    )

    print("\nOPTIMAL BASE FREQUENCY (under current model)")
    print(f"  f0*          : {best['f0']:.6f} Hz")
    print(f"  score        : {best['score']:.6e}")
    print(f"  phase_var    : {best['phase_var']:.6e}")
    print(f"  closure_mean : {best['closure_mean']:.6e}")

    octave = build_octave_frequencies(best["f0"], args.temperament)
    chrom = octave["chromatic"]
    majors = octave["principles_major"]
    harms = octave["harmonics_sharp"]

    print("\nFULL OCTAVE (chromatic 12-tone set from f0*)")
    for i, f in enumerate(chrom):
        print(f"  index {i:2d} ({NOTE_NAMES_A_ROOT[i]:2s}): {f:9.4f} Hz")

    print("\nMAJOR PRINCIPLES (7 diatonic tones mapped to principles)")
    for name in PRINCIPLES:
        f = majors[name]
        deg = MAJOR_DEGREES[PRINCIPLES.index(name)]
        print(f"  {name:7s} (degree {deg:2d}, note {NOTE_NAMES_A_ROOT[deg]:2s}): {f:9.4f} Hz")

    print("\nHARMONIC SHARPS (5 chromatic tones between principles)")
    for i, deg in enumerate(SHARP_DEGREES):
        label = f"H{i+1}"
        f = harms[label]
        print(f"  {label:3s} (degree {deg:2d}, note {NOTE_NAMES_A_ROOT[deg]:2s}): {f:9.4f} Hz")

    multi_oct = build_multi_octave_table(best["f0"], octaves=(-1, 1), temperament=args.temperament)

    print("\nMULTI-OCTAVE TABLE (A-root, -1..+1 around f0*)")
    for oct_key, notes in multi_oct.items():
        print(f"  {oct_key}:")
        for label, freq in sorted(notes.items(), key=lambda kv: kv[0]):
            print(f"    {label:4s}: {freq:9.4f} Hz")

    elem_map = build_element_mapping(best["f0"], max_Z=args.max_Z, temperament=args.temperament)

    print("\nELEMENTAL MAPPING (selected elements)")
    interesting_Z = [1, 6, 7, 8, 26, 29, 47, 79, 82, 92]
    for Z in interesting_Z:
        key = str(Z)
        if key not in elem_map:
            continue
        e = elem_map[key]
        print(
            f"  Z={e['Z']:3d} {e['symbol']:2s} ({e['name']:<11s}) -> "
            f"{e['label']:5s} @ {e['frequency_hz']:11.4f} Hz, "
            f"λ={e['wavelength_m']: .3e} m, E={e['energy_eV']: .3e} eV, band={e['band']}"
        )

    ladder = build_12d_ladder(best["f0"], n_min=args.n_min, n_max=args.n_max, temperament=args.temperament)
    print(f"\n12D LADDER SIZE: {len(ladder)} points")

    visible_center = find_visible_center(ladder)
    if visible_center:
        n_center = visible_center["n"]
        reindex_ladder_dimensions(ladder, n_center)
        print("\nVISIBLE CENTER (dim_index = 0 slice)")
        print(
            f"  n_center={visible_center['n']:5d} label={visible_center['label']:8s} "
            f"f={visible_center['frequency_hz']:11.4e} Hz λ={visible_center['wavelength_m']: .3e} m "
            f"E={visible_center['energy_eV']: .3e} eV"
        )
        print("\nNEIGHBORHOOD AROUND VISIBLE CENTER (dim_index -6..+5)")
        for p in ladder:
            dim = p.get("dim_index", 0)
            if -6 <= dim <= 5:
                print(
                    f"  dim={dim:4d} cyc={p.get('harmonic_cycle',0):4d} "
                    f"n={p['n']:5d} {p['label']:8s} "
                    f"f={p['frequency_hz']:11.4e} Hz band={p['band']}"
                )
    else:
        print("\nNo visible band center found; dim_index/harmonic_cycle remain default.")

    print("\nLADDER SEGMENT (AUDIO BAND ~20–20000 Hz)")
    for p in ladder:
        if 20.0 <= p["frequency_hz"] <= 20000.0:
            dim = p.get("dim_index", 0)
            cyc = p.get("harmonic_cycle", 0)
            print(
                f"  n={p['n']:4d} dim={dim:4d} cyc={cyc:4d} {p['label']:8s} "
                f"f={p['frequency_hz']:11.4f} Hz band={p['band']}"
            )

    print("\nLADDER SEGMENT (VISIBLE BAND ~4e14–7.5e14 Hz)")
    for p in ladder:
        if 4.0e14 <= p["frequency_hz"] <= 7.5e14:
            dim = p.get("dim_index", 0)
            cyc = p.get("harmonic_cycle", 0)
            print(
                f"  dim={dim:4d} cyc={cyc:4d} n={p['n']:5d} {p['label']:8s} "
                f"f={p['frequency_hz']:11.4e} Hz λ={p['wavelength_m']: .3e} m E={p['energy_eV']: .3e} eV"
            )

    print("\nGLOBAL COALESCENCE SCAN (full ladder top→1 Hz)")
    coalesce_points = scan_coalescence_over_ladder(
        samples=samples,
        ladder=ladder,
        temperament=args.temperament,
        closure_weight=args.closure_weight,
        f_min=args.coalesce_fmin,
        top_k=args.coalesce_topk,
    )
    for idx, cp in enumerate(coalesce_points, 1):
        print(
            f"  #{idx:2d} dim={cp['dim_index']:4d} cyc={cp['harmonic_cycle']:4d} "
            f"n={cp['n']:5d} {cp['label']:8s} "
            f"f0={cp['frequency_hz']:11.4f} Hz "
            f"score={cp['score']: .6e} phase_var={cp['phase_var']: .6e} "
            f"closure_mean={cp['closure_mean']: .6e}"
        )

    if not args.no_refs:
        refs = [432.0, 435.0, 440.0, 444.0]
        print("\nREFERENCE BASE FREQUENCIES")
        ref_rows = evaluate_references(samples, refs, args.temperament, args.closure_weight)
        print("  f0 (Hz)    score            phase_var        closure_mean")
        for row in ref_rows:
            print(
                f"  {row['f0']:7.2f}  {row['score']: .6e}  "
                f"{row['phase_var']: .6e}  {row['closure_mean']: .6e}"
            )

    if args.export_json:
        export_solution_json(
            args.export_json,
            best,
            octave,
            multi_oct,
            elem_map,
            ladder,
            visible_center,
            coalesce_points,
        )
        print(f"\nExported tuning solution to {args.export_json}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
