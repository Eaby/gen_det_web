from __future__ import annotations
from typing import Dict, List, Tuple

# Weighted verdict computation shared by controllers

def weighted_verdict(folder_natural_frac: Dict[str, float], folder_counts: Dict[str, int], weights: Dict[str, float], threshold_natural: float = 0.60) -> Tuple[str, float, str]:
    used = [(s, folder_natural_frac.get(s, 0.0), folder_counts.get(s, 0), w) for s, w in weights.items() if folder_counts.get(s, 0) > 0 and w > 0]
    denom = sum(w for *_, w in used)
    if denom <= 0:
        return ("Insufficient Data", 0.0, "")
    weighted_score = sum(frac * w for _, frac, _, w in used) / denom
    verdict = "Likely Natural" if weighted_score >= threshold_natural else ("Mixed" if 0.40 < weighted_score < threshold_natural else "Likely AI")
    rows = "".join([f"<tr><td>{s}</td><td>{n}</td><td>{f*100:.1f}%</td><td>{w:.2f}</td></tr>" for s, f, n, w in used])
    html = f"<table class='results-table'><thead><tr><th>Set</th><th>N</th><th>Natural%</th><th>Weight</th></tr></thead><tbody>{rows}</tbody></table>"
    return verdict, weighted_score, html
