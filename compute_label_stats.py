#!/usr/bin/env python3
"""
compute_label_stats.py

Produces the numbers needed for the Label Construction table and for the
prevalence / kappa discussion, straight from your labelled data files.

Outputs, per dataset:
    n            total labelled units
    n_pos        positive units
    pos_rate     positive rate (%)
    p_e          chance-agreement term used by Cohen's kappa,
                 p_e = p_neg^2 + p_pos^2  (matched-marginals case)
    1 - p_e      the denominator of kappa -- this is the number that
                 determines whether the Proposition's bound is usable

It also emits ready-to-paste LaTeX table cells and a short prose sentence
with the real ranges, so nothing has to be transcribed by hand.

USAGE
-----
Edit DATASETS below to point at your four labelled files, then:

    python3 compute_label_stats.py

Supports .csv, .tsv, .parquet, .feather, .json (lines), .npy, .npz.
If your labels live somewhere unusual, the only thing that matters is that
load_labels() returns a 1-D array of 0/1 -- adapt that one function.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required:  pip install pandas")


# ---------------------------------------------------------------------
# EDIT THIS BLOCK
# ---------------------------------------------------------------------
# (display_name, path_to_labelled_file, label_column_name, unit_noun)
#
# `unit_noun` is only used for the printed table and should match the
# "Unit" column of Table: trip / trajectory / check-in / review.

DATASETS = [
    ("NYC Taxi",   "nyc_fraud_real_15_p_100000.csv",    "label", "trip"),
    ("Geolife",    "data/divergence-ds/geolife_real100000.csv",     "label", "trajectory"),
    ("Foursquare", "data/divergence-ds/foursquare_real100000.csv",  "label", "check-in"),
    ("Yelp",       "data/divergence-ds/yelp_real100000.csv",        "label", "review"),
]

# If you also want per-client positive rates (useful for explaining why
# some clients see no positives at all under non-IID partitioning), set
# this to the column holding the client/partition id, else leave as None.
CLIENT_COLUMN = None          # e.g. "client_id"
# ---------------------------------------------------------------------


def load_labels(path: str, label_col: str, client_col: str | None = None):
    """Return (labels, client_ids_or_None) as numpy arrays."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    suffix = p.suffix.lower()

    if suffix in {".csv", ".txt"}:
        cols = [label_col] + ([client_col] if client_col else [])
        df = pd.read_csv(p, usecols=cols)
    elif suffix == ".tsv":
        cols = [label_col] + ([client_col] if client_col else [])
        df = pd.read_csv(p, sep="\t", usecols=cols)
    elif suffix == ".parquet":
        cols = [label_col] + ([client_col] if client_col else [])
        df = pd.read_parquet(p, columns=cols)
    elif suffix == ".feather":
        df = pd.read_feather(p)
    elif suffix == ".json":
        df = pd.read_json(p, lines=True)
    elif suffix == ".npy":
        arr = np.load(p)
        return np.asarray(arr).ravel(), None
    elif suffix == ".npz":
        z = np.load(p)
        return np.asarray(z[label_col]).ravel(), None
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    y = df[label_col].to_numpy().ravel()
    c = df[client_col].to_numpy().ravel() if client_col else None
    return y, c


def summarise(name: str, y: np.ndarray, unit: str, clients=None) -> dict:
    y = np.asarray(y)

    # Coerce to 0/1 and complain loudly if it isn't binary.
    uniq = np.unique(y[~pd.isna(y)])
    if not set(uniq.tolist()) <= {0, 1, 0.0, 1.0, True, False}:
        raise ValueError(
            f"{name}: labels are not binary 0/1 -- found {uniq[:10]}"
        )
    y = y.astype(int)

    n = int(y.size)
    n_pos = int(y.sum())
    p_pos = n_pos / n if n else float("nan")
    p_neg = 1.0 - p_pos

    # Chance agreement under matched marginals -- the p_e of Cohen's kappa.
    p_e = p_neg**2 + p_pos**2

    out = {
        "name": name,
        "unit": unit,
        "n": n,
        "n_pos": n_pos,
        "pos_rate": 100.0 * p_pos,
        "p_e": p_e,
        "one_minus_pe": 1.0 - p_e,
    }

    if clients is not None:
        df = pd.DataFrame({"c": clients, "y": y})
        per = df.groupby("c")["y"].agg(["size", "sum"])
        per["rate"] = per["sum"] / per["size"]
        out["n_clients"] = int(per.shape[0])
        out["clients_zero_pos"] = int((per["sum"] == 0).sum())
        out["frac_clients_zero_pos"] = 100.0 * (per["sum"] == 0).mean()
        out["client_rate_median"] = 100.0 * float(per["rate"].median())
        out["client_rate_p90"] = 100.0 * float(per["rate"].quantile(0.90))

    return out


def main() -> None:
    rows = []
    for name, path, label_col, unit in DATASETS:
        try:
            y, c = load_labels(path, label_col, CLIENT_COLUMN)
        except Exception as exc:                      # noqa: BLE001
            print(f"!! {name}: {exc}", file=sys.stderr)
            continue
        rows.append(summarise(name, y, unit, c))

    if not rows:
        sys.exit("No datasets could be read. Check the DATASETS block.")

    # ---------------- human-readable ----------------
    print("\n" + "=" * 78)
    print("LABEL STATISTICS")
    print("=" * 78)
    hdr = f"{'Dataset':<12} {'unit':<11} {'n':>10} {'n_pos':>9} {'pos %':>7} {'p_e':>7} {'1-p_e':>7}"
    print(hdr)
    print("-" * 78)
    for r in rows:
        print(
            f"{r['name']:<12} {r['unit']:<11} {r['n']:>10,} {r['n_pos']:>9,} "
            f"{r['pos_rate']:>6.2f}% {r['p_e']:>7.4f} {r['one_minus_pe']:>7.4f}"
        )

    if CLIENT_COLUMN:
        print("\n" + "-" * 78)
        print("PER-CLIENT (relevant to why some clients never see a positive)")
        print("-" * 78)
        print(f"{'Dataset':<12} {'clients':>9} {'0-pos':>8} {'0-pos %':>9} {'med %':>8} {'p90 %':>8}")
        for r in rows:
            if "n_clients" in r:
                print(
                    f"{r['name']:<12} {r['n_clients']:>9,} "
                    f"{r['clients_zero_pos']:>8,} {r['frac_clients_zero_pos']:>8.1f}% "
                    f"{r['client_rate_median']:>7.2f}% {r['client_rate_p90']:>7.2f}%"
                )

    # ---------------- LaTeX cells ----------------
    print("\n" + "=" * 78)
    print("PASTE INTO Table~\\ref{tab:labels}  (Pos. rate and n columns)")
    print("=" * 78)
    for r in rows:
        print(f"  {r['name']:<12} ->  & {r['pos_rate']:.1f}\\% & {r['n']:,} \\\\".replace(",", "{,}"))

    # ---------------- prose ----------------
    lo = min(r["pos_rate"] for r in rows)
    hi = max(r["pos_rate"] for r in rows)
    pe_lo = min(r["p_e"] for r in rows)
    pe_hi = max(r["p_e"] for r in rows)

    print("\n" + "=" * 78)
    print("PASTE INTO THE PROSE (replaces the bracketed sentence)")
    print("=" * 78)
    print(
        f"\nPositive rates range from {lo:.1f}\\% to {hi:.1f}\\%. Because $\\kappa$ is\n"
        f"prevalence-sensitive, absolute values are not comparable across datasets,\n"
        f"and at these rates the chance-agreement term satisfies\n"
        f"$p_e \\in [{pe_lo:.3f}, {pe_hi:.3f}]$ on every dataset.\n"
    )

    # ---------------- the check that matters ----------------
    print("=" * 78)
    print("CONSISTENCY CHECK against statements already in the manuscript")
    print("=" * 78)
    claims = {
        "Geolife": 3.5,   # "approximately 3.5% fraud rate", heterogeneity section
    }
    for r in rows:
        if r["name"] in claims:
            stated, actual = claims[r["name"]], r["pos_rate"]
            flag = "OK" if abs(stated - actual) < 0.3 else "MISMATCH"
            print(f"  {r['name']}: manuscript says {stated}%, data says {actual:.2f}%  [{flag}]")

    print(
        f"\n  Manuscript sensitivity section says the range is 1.8%-12.6%.\n"
        f"  Data says {lo:.1f}%-{hi:.1f}%.  "
        f"{'OK' if (abs(lo-1.8) < 0.3 and abs(hi-12.6) < 0.3) else 'MISMATCH - sweep it'}"
    )

    print(
        f"\n  Remark 2 worked example assumes 85/15, giving 1-p_e = 0.255.\n"
        f"  Smallest actual 1-p_e is {min(r['one_minus_pe'] for r in rows):.4f}.\n"
        f"  The bound kappa <= 1 - eps/(1-p_e) is vacuous for any eps > "
        f"{min(r['one_minus_pe'] for r in rows):.3f}.\n"
    )


if __name__ == "__main__":
    main()