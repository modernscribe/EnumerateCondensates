#!/usr/bin/env python3
import json
import numpy as np
import math
from typing import List, Dict, Tuple
import os

N_DIM = 12
PRINCIPLES = ["Truth", "Purity", "Law", "Love", "Wisdom", "Life", "Glory"]
ZODIAC = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
EPS = 1e-12

def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def Null_point():
    return 0.0

def InverseZero_operator(p):
    return -0.0 if p == 0.0 else p

def H_inverse_zero_null():
    return 1.0

def sigma_selector(omega):
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

def step_vector(x: np.ndarray, enabled: np.ndarray) -> np.ndarray:
    x_masked = x * enabled
    for name in PRINCIPLES:
        x_masked = apply_principle(name, x_masked)
    return x_masked

def omega_k(k):
    return 2.0 * math.pi * (k / 12.0)

def project_to_torus(vec: np.ndarray, R: float = 3.0, r: float = 1.0) -> Tuple[float, float]:
    u = np.arctan2(np.sum(vec[6:]), np.sum(vec[:6]))
    v = np.arctan2(
        np.sum(vec[3:9]),
        np.sum(np.concatenate([vec[:3], vec[9:]]))
    )
    return float(u), float(v)

def embed_on_torus(u: float, v: float, amplitude: np.ndarray,
                   R: float = 3.0, r: float = 1.0) -> Tuple[float, float, float]:
    X0 = (R + r * np.cos(v)) * np.cos(u)
    Y0 = (R + r * np.cos(v)) * np.sin(u)
    Z0 = r * np.sin(v)
    amp = float(np.mean(np.abs(amplitude)))
    scale = 1.0 + 0.3 * amp
    return X0 * scale, Y0 * scale, Z0 * scale

def compute_holonomy(fp: np.ndarray, enabled: np.ndarray) -> Dict:
    holonomies: Dict[str, Dict] = {}
    for name in PRINCIPLES:
        base = fp * enabled
        vec = apply_principle(name, base)
        phases: List[float] = []
        for k in range(N_DIM):
            if enabled[k]:
                omega = omega_k(k)
                phase = omega * vec[k]
                phases.append(float(phase))
        if phases:
            total_phase = float(np.sum(phases))
            holonomy = complex(math.cos(total_phase), math.sin(total_phase))
        else:
            total_phase = 0.0
            holonomy = complex(1.0, 0.0)
        holonomies[name] = {
            "phase": float(total_phase),
            "holonomy": [float(holonomy.real), float(holonomy.imag)],
            "magnitude": float(abs(holonomy)),
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

def compute_tonal_regime(fp: np.ndarray, entropy: float) -> Dict:
    amp = np.abs(fp.astype(float))
    s = float(np.sum(amp))
    if s < EPS:
        return {
            "label": "null",
            "dominant_indices": [],
            "dominant_weights": [],
        }
    amp_norm = amp / s
    sorted_idx = np.argsort(amp_norm)[::-1]
    idx1 = int(sorted_idx[0])
    w1 = float(amp_norm[idx1])
    idx2 = int(sorted_idx[1]) if N_DIM > 1 else idx1
    w2 = float(amp_norm[idx2])
    idx3 = int(sorted_idx[2]) if N_DIM > 2 else idx2
    w3 = float(amp_norm[idx3])
    if w1 > 0.7:
        label = "monochord"
    elif w1 > 0.45 and w2 > 0.25:
        label = "dyad"
    elif w1 > 0.3 and w3 > 0.15:
        label = "triad"
    elif entropy > 3.0:
        label = "distributed"
    else:
        label = "complex"
    return {
        "label": label,
        "dominant_indices": [idx1, idx2, idx3],
        "dominant_weights": [w1, w2, w3],
    }

def classify_torus_sector(u: float, v: float) -> Dict:
    two_pi = 2.0 * math.pi
    def wrap(a: float) -> float:
        return float(a % two_pi)
    u_w = wrap(u)
    v_w = wrap(v)
    if abs(v_w - math.pi / 2.0) <= math.pi / 4.0:
        radial_band = "crown_band"
    elif abs(v_w - 3.0 * math.pi / 2.0) <= math.pi / 4.0:
        radial_band = "root_band"
    elif v_w <= math.pi / 4.0 or v_w >= two_pi - math.pi / 4.0:
        radial_band = "equator_band"
    else:
        radial_band = "mid_band"
    quadrant = int(u_w // (math.pi / 2.0)) % 4
    if quadrant == 0:
        longitudinal_sector = "east"
    elif quadrant == 1:
        longitudinal_sector = "north"
    elif quadrant == 2:
        longitudinal_sector = "west"
    else:
        longitudinal_sector = "south"
    diff = abs(u_w - v_w)
    if diff > math.pi:
        diff = two_pi - diff
    diagonal_aligned = bool(diff < (math.pi / 8.0))
    return {
        "radial_band": radial_band,
        "longitudinal_sector": longitudinal_sector,
        "diagonal_aligned": diagonal_aligned,
    }

def compute_symmetry_profile(fp: np.ndarray, enabled: np.ndarray, rotational_symmetry: float) -> Dict:
    vec = fp.astype(float)
    norm = float(np.linalg.norm(vec))
    if norm < EPS:
        return {
            "rotational_class": "none",
            "alt_correlation": 0.0,
            "mirror_correlation": 0.0,
            "enabled_pattern": "silent",
        }
    mirror_vec = vec[::-1]
    mirror_corr = float(np.dot(vec, mirror_vec) / (norm * norm))
    idx = np.arange(N_DIM, dtype=float)
    alt_mask = np.power(-1.0, idx)
    alt_vec = vec * alt_mask
    alt_corr = float(np.dot(vec, alt_vec) / (norm * norm))
    if rotational_symmetry > 0.8:
        rotational_class = "high_cyclic"
    elif rotational_symmetry > 0.4:
        rotational_class = "partial_cyclic"
    elif rotational_symmetry > 0.1:
        rotational_class = "weak_cyclic"
    else:
        rotational_class = "aperiodic"
    positions = np.where(enabled)[0]
    if positions.size == 0:
        pattern = "silent"
    else:
        span = int(positions[-1] - positions[0] + 1)
        count = int(positions.size)
        density = float(count) / float(span) if span > 0 else 1.0
        if span == count:
            pattern = "clustered"
        elif density < 0.5:
            pattern = "spread"
        else:
            pattern = "multi_cluster"
    return {
        "rotational_class": rotational_class,
        "alt_correlation": alt_corr,
        "mirror_correlation": mirror_corr,
        "enabled_pattern": pattern,
    }

def compute_jacobian(fp: np.ndarray, enabled: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    J = np.zeros((N_DIM, N_DIM), dtype=float)
    f0 = step_vector(fp, enabled)
    for i in range(N_DIM):
        if not enabled[i]:
            continue
        x_plus = fp.copy()
        x_plus[i] += eps
        f_plus = step_vector(x_plus, enabled)
        J[:, i] = (f_plus - f0) / eps
    return J

def compute_modes(jacobian: np.ndarray, top_k: int = 4) -> List[Dict]:
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    indices = np.argsort(np.abs(eigenvalues))[::-1]
    modes: List[Dict] = []
    for i, idx in enumerate(indices[:top_k]):
        lam = eigenvalues[idx]
        vec = eigenvectors[:, idx]
        vec_norm = vec / (np.linalg.norm(vec) + EPS)
        modes.append({
            "index": int(i),
            "lambda_real": float(np.real(lam)),
            "lambda_imag": float(np.imag(lam)),
            "lambda_magnitude": float(np.abs(lam)),
            "vector": np.real(vec_norm).astype(float).tolist(),
        })
    return modes

def compute_fixed_point_enhanced(
    address: np.ndarray,
    enabled: np.ndarray,
    tol: float = 1e-9,
    max_steps: int = 4096
) -> Dict:
    x = address * enabled
    trajectory = [x.copy()]
    converged = False
    step = 0
    delta = float("inf")
    for step in range(max_steps):
        y = step_vector(x, enabled)
        delta = float(np.linalg.norm(y - x))
        x = y
        trajectory.append(x.copy())
        if delta < tol:
            converged = True
            break
    fp = x
    is_null = bool(np.all(np.abs(fp) < EPS))
    sum_abs = float(np.sum(np.abs(fp)))
    l2 = float(np.linalg.norm(fp))
    if sum_abs > EPS:
        normalized = np.abs(fp) / sum_abs
        entropy = 0.0
        for p in normalized:
            if p > EPS:
                entropy += float(p * math.log2(p))
        entropy = float(-entropy)
    else:
        entropy = 0.0
    max_corr = 0.0
    for shift in range(1, N_DIM):
        corr = float(np.dot(fp, np.roll(fp, shift)))
        max_corr = max(max_corr, abs(corr))
    u, v = project_to_torus(fp)
    X, Y, Z = embed_on_torus(u, v, fp)
    holonomy = compute_holonomy(fp, enabled)
    tonal_regime = compute_tonal_regime(fp, entropy)
    space_sector = classify_torus_sector(u, v)
    symmetry_profile = compute_symmetry_profile(fp, enabled, max_corr)
    modes = None
    stability = "unknown"
    if converged and not is_null:
        try:
            J = compute_jacobian(fp, enabled)
            modes = compute_modes(J, top_k=4)
            if all(m["lambda_magnitude"] < 1.0 for m in modes):
                stability = "stable"
            else:
                stability = "unstable"
        except Exception:
            modes = None
            stability = "unknown"
    relaxation_steps = int(step + 1)
    relaxation_rate = float(delta / relaxation_steps) if relaxation_steps > 0 else float("inf")
    return {
        "fixed_point": {
            "vector": fp.astype(float).tolist(),
            "converged": bool(converged),
            "steps": relaxation_steps,
            "last_delta": float(delta),
            "is_null": bool(is_null),
        },
        "invariants": {
            "sum_abs": sum_abs,
            "l2_norm": l2,
            "entropy": float(entropy),
            "rotational_symmetry": float(max_corr),
        },
        "geometry": {
            "toroidal_coords": {"u": float(u), "v": float(v)},
            "embedding": {"X": float(X), "Y": float(Y), "Z": float(Z)},
        },
        "holonomy": holonomy,
        "modes": modes,
        "stability": stability,
        "trajectory_length": int(len(trajectory)),
        "regime": tonal_regime,
        "space_sector": space_sector,
        "symmetry_profile": symmetry_profile,
        "dynamics": {
            "relaxation_steps": relaxation_steps,
            "relaxation_rate": relaxation_rate,
        },
    }

def generate_structured_addresses(n: int) -> List[np.ndarray]:
    samples: List[np.ndarray] = []
    samples.append(np.ones(N_DIM, dtype=float) / float(N_DIM))
    for i in range(N_DIM):
        samples.append(np.eye(N_DIM, dtype=float)[i])
    samples.append(np.array([1.0 / float(i + 1) for i in range(N_DIM)], dtype=float))
    samples.append(np.array(
        [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
        dtype=float
    ))
    normalized: List[np.ndarray] = []
    for s in samples:
        s_abs = np.sum(np.abs(s))
        if s_abs < EPS:
            normalized.append(np.zeros_like(s))
        else:
            normalized.append(s / s_abs)
    samples = normalized
    rng = np.random.default_rng()
    while len(samples) < n:
        addr = rng.standard_normal(N_DIM, dtype=float)
        s_abs = np.sum(np.abs(addr))
        if s_abs < EPS:
            continue
        addr = addr / s_abs
        samples.append(addr)
    return samples[:n]

def run_exploration_enhanced(
    n_samples: int = 100,
    output_file: str = "harmonic_exploration_enhanced.jsonl"
):
    print("Starting enhanced harmonic exploration with toroidal embedding...")
    print(f"Samples: {n_samples}")
    print()
    addresses = generate_structured_addresses(n_samples)
    enabled_patterns = [
        np.ones(N_DIM, dtype=bool),
        np.array([i < 6 for i in range(N_DIM)], dtype=bool),
        np.array([i >= 6 for i in range(N_DIM)], dtype=bool),
        np.array([i % 2 == 0 for i in range(N_DIM)], dtype=bool),
        np.array([i in [0, 4, 7] for i in range(N_DIM)], dtype=bool),
    ]
    results: List[Dict] = []
    total = len(addresses) * len(enabled_patterns)
    for i, addr in enumerate(addresses):
        for j, enabled in enumerate(enabled_patterns):
            sample_id = i * len(enabled_patterns) + j
            if sample_id % 50 == 0:
                print(f"Progress: {sample_id}/{total} ({100.0 * sample_id / total:.1f}%)")
            result = compute_fixed_point_enhanced(addr, enabled)
            result["id"] = f"sample_{sample_id:05d}"
            result["address"] = addr.astype(float).tolist()
            result["enabled_tones"] = [bool(b) for b in enabled.tolist()]
            results.append(result)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=_json_default) + "\n")
    print(f"\nEnhanced exploration complete! Saved {len(results)} samples")
    null_count = sum(1 for r in results if r["fixed_point"]["is_null"])
    stable_count = sum(1 for r in results if r["stability"] == "stable")
    unstable_count = sum(1 for r in results if r["stability"] == "unstable")
    print("\nSUMMARY:")
    print(f"Null states: {null_count}/{len(results)} ({100.0 * null_count / len(results):.1f}%)")
    print(f"Stable: {stable_count}")
    print(f"Unstable: {unstable_count}")
    closures = [
        r["holonomy"]["crown"]["closure"]
        for r in results
        if not r["fixed_point"]["is_null"]
    ]
    if closures:
        closures_arr = np.array(closures, dtype=float)
        print(f"Avg holonomy closure: {float(np.mean(closures_arr)):.6e}")
        print(f"Max holonomy closure: {float(np.max(closures_arr)):.6e}")
    return results

def analyze_zero_topology(results: List[Dict]):
    print("\n" + "=" * 70)
    print("ZERO TOPOLOGY ANALYSIS")
    print("=" * 70)
    null_samples = [r for r in results if r["fixed_point"]["is_null"]]
    non_null = [r for r in results if not r["fixed_point"]["is_null"]]
    print(f"Null samples: {len(null_samples)}/{len(results)}")
    if null_samples:
        print("\nNull state characteristics:")
        for r in null_samples[:5]:
            enabled_count = sum(1 for b in r["enabled_tones"] if b)
            print(f"  {r['id']}: {enabled_count}/12 tones enabled")
            closure = r["holonomy"]["crown"]["closure"]
            print(f"    Holonomy closure: {closure:.6e}")
    if non_null:
        print(f"\nNon-null samples: {len(non_null)}")
        entropies = [r["invariants"]["entropy"] for r in non_null]
        arr = np.array(entropies, dtype=float)
        print(f"  Entropy range: {float(np.min(arr)):.3f} - {float(np.max(arr)):.3f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified Harmonic Explorer")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", default="harmonic_exploration_enhanced.jsonl")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    results = run_exploration_enhanced(args.samples, args.output)
    if args.analyze:
        analyze_zero_topology(results)
    print("\nComplete!")

if __name__ == "__main__":
    main()
