#!/usr/bin/env python3
"""
Generate LaTeX/Markdown tables and figures from experiment results.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def table1_clean():
    """Table 1: Clean video results"""
    r = load_json(Path(__file__).parent.parent / "results" / "exp1_clean" / "results.json")
    if not r:
        return "# Run exp1_clean.py first"
    raw = r.get("raw", r)
    methods = ["simp_v", "pde_find", "pinn", "deepmod"]
    pdes = list(raw.keys())
    rows = ["| PDE | " + " | ".join(m.title() for m in methods) + " |"]
    rows.append("|" + "|".join(["---"] * (len(methods) + 1)) + "|")
    for pde in pdes:
        vals = []
        for m in methods:
            lst = raw[pde].get(m, [])
            if lst:
                import numpy as np
                vals.append(f"{np.mean(lst):.2f} ± {np.std(lst):.2f}")
            else:
                vals.append("—")
        rows.append(f"| {pde} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def table2_noisy():
    """Table 2: Noisy (5% Gaussian)"""
    r = load_json(Path(__file__).parent.parent / "results" / "exp2_noisy" / "results.json")
    if not r:
        return "# Run exp2_noisy.py first"
    rows = ["| PDE | SIMP-V (%) | PDE-FIND (%) | Improvement |", "|-----|------------|--------------|-------------|"]
    for pde, data in r.items():
        key = "gaussian_0.05" if "gaussian_0.05" in data else list(data.keys())[0]
        d = data.get(key, data)
        sv = d.get("simp_v", [])
        pf = d.get("pde_find", [])
        if sv and pf:
            import numpy as np
            m_sv, m_pf = np.mean(sv), np.mean(pf)
            imp = f"{(m_pf / max(m_sv, 0.01)):.1f}x" if m_sv > 0 else "—"
            rows.append(f"| {pde} | {m_sv:.2f} | {m_pf:.2f} | {imp} |")
    return "\n".join(rows)


def table3_implicit():
    """Table 3: Implicit dynamics (coverage)"""
    r = load_json(Path(__file__).parent.parent / "results" / "exp3_implicit" / "results.json")
    if not r:
        return "# Run exp3_implicit.py first"
    coverages = ["20%", "40%", "60%", "80%", "100%"]
    rows = ["| PDE | " + " | ".join(coverages) + " |", "|-----|" + "|".join(["---"] * len(coverages)) + "|"]
    for pde, data in r.items():
        vals = []
        for c in coverages:
            d = data.get(c, {})
            lst = d.get("simp_v", [])
            if lst:
                import numpy as np
                vals.append(f"{np.mean(lst):.1f}")
            else:
                vals.append("—")
        rows.append(f"| {pde} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main():
    out = Path(__file__).parent.parent / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)

    t1 = table1_clean()
    t2 = table2_noisy()
    t3 = table3_implicit()

    with open(out / "table1_clean.md", "w") as f:
        f.write("# Table 1: Clean Video Results\n\n")
        f.write(t1)
    with open(out / "table2_noisy.md", "w") as f:
        f.write("# Table 2: Noisy Video (5% Gaussian)\n\n")
        f.write(t2)
    with open(out / "table3_implicit.md", "w") as f:
        f.write("# Table 3: Implicit Dynamics (Spatial Coverage)\n\n")
        f.write(t3)

    print("Tables written to results/tables/")
    print("\n--- Table 1 ---\n", t1)
    print("\n--- Table 2 ---\n", t2)
    print("\n--- Table 3 ---\n", t3)


if __name__ == "__main__":
    main()
