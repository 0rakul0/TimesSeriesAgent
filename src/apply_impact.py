# apply_impact.py
import json
import numpy as np
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LIB_PATH = os.path.join(DATA_DIR, "impact_library.json")

def load_library():
    if not os.path.exists(LIB_PATH):
        raise FileNotFoundError("impact_library.json not found; run build first")
    return json.load(open(LIB_PATH, "r", encoding="utf-8"))

def compute_scale_factor(r_new0, ref0, hist_std=None, method="zscore", alpha_cap=(0.2,5.0)):
    # r_new0, ref0 are returns (e.g. -0.2 for -20%)
    if method == "none" or ref0 == 0:
        alpha = 1.0
    elif method == "ratio":
        alpha = (r_new0) / (ref0)
    elif method == "zscore":
        if hist_std is None or hist_std == 0:
            alpha = (r_new0) / (ref0) if ref0 != 0 else 1.0
        else:
            z_new = r_new0 / hist_std
            z_ref = ref0 / hist_std
            if z_ref == 0:
                alpha = 1.0
            else:
                alpha = z_new / z_ref
    else:
        raise ValueError("unknown method")
    # cap
    amin, amax = alpha_cap
    alpha = max(amin, min(amax, alpha))
    return float(alpha)

def apply_reference_sequence(cluster, r_new0, hist_std=None, scale_method="zscore"):
    lib = load_library()
    if cluster not in lib:
        # fallback: no library -> zero sequence
        return None
    ref_seq = np.array(lib[cluster]["ref_seq"], dtype=float)
    ref_r0 = ref_seq[0]
    alpha = compute_scale_factor(r_new0, ref_r0, hist_std=hist_std, method=scale_method)
    applied = (alpha * ref_seq).tolist()
    return {
        "cluster": cluster,
        "alpha": alpha,
        "ref_seq": ref_seq.tolist(),
        "applied_seq": applied
    }
