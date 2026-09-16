#!/usr/bin/env python3
"""
run_ablation.py

Harness for the ablation study of Section~\\ref{sec:ablation}.

Runs the 2^3 layer factorial plus the Byzantine-fraction sweep, aggregates
over seeds, runs the paired Wilcoxon tests with Holm correction and Cliff's
delta, and emits the LaTeX table bodies ready to paste.

WHAT IS WIRED UP
----------------
`run_one()` invokes the repo's federated trainer (build_federation /
run_federation + the toggle-aware MultiLayerAggregator "Ours") with the given
configuration and returns the metrics dict. Everything downstream of it -- the
experimental grid, seed handling, aggregation, statistics, multiplicity
correction, effect sizes, LaTeX emission -- was already written.

Results are checkpointed to disk after every run, so the script is safe to
interrupt and resume. Delete ablation_results.jsonl to start over.

    python3 run_ablation.py --stage factorial
    python3 run_ablation.py --stage sweep
    python3 run_ablation.py --stage report

Dataset files (100k-record CSVs, feature space matching the paper) are resolved
per dataset from --data-dir; override any single path with, e.g.,
    --data nyc_taxi=nyc_real100000.csv geolife=geolife_real.csv
Any dataset whose file is missing is skipped with a warning, so you can run a
subset without editing the grid.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    sys.exit("scipy is required:  pip install scipy")


RESULTS = Path("ablation_results.jsonl")

DATASETS = ["nyc_taxi", "geolife", "foursquare", "yelp"]
ATTACKS = ["grad_p", "lfp", "spf", "ooa"]
SEEDS = [0, 1, 2, 3, 4]

# Layer factorial: (L1 outlier, L2 trust, L3 kappa)
FACTORIAL = list(itertools.product([False, True], repeat=3))

# Fraction sweep uses a subset of configurations.
SWEEP_CONFIGS = [
    (True, True, True),     # full
    (True, True, False),    # no kappa
    (False, True, True),    # no outlier
    (False, False, False),  # no defence
]
SWEEP_FRACTIONS = [0.1, 0.2, 0.3, 0.4]

MAIN_FRACTION = 0.3

# ── Fixed federated protocol (identical across every configuration) ──────────
N_CLIENTS = 100
PER_ROUND = 20
ROUNDS = 50
LOCAL_EPOCHS = 5
LR = 0.01
NONIID_ALPHA = 0.5
VAL_SIZE = 2000

# Ablation attack code -> trainer attack code.
ATTACK_CODE = {"grad_p": "grad-p", "lfp": "lfp", "spf": "spf", "ooa": "ooa"}

# Candidate filenames per dataset, resolved against --data-dir (first hit wins).
DATA_CANDIDATES = {
    "nyc_taxi":   ["nyc_real100000.csv", "nyc_real.csv", "nyc_taxi.csv"],
    "geolife":    ["geolife_real100000.csv", "geolife_real.csv", "geolife.csv"],
    "foursquare": ["foursquare_real100000.csv", "foursquare_real.csv",
                   "foursquare.csv"],
    "yelp":       ["yelp_real100000.csv", "yelp_real.csv", "yelp.csv"],
}

# Filled from --data / --data-dir in main(); dataset -> resolved path (or None).
DATA_PATHS: dict = {}
# Cache of loaded (X, y) per dataset so a 100k CSV is read once, not 80x.
_DATA_CACHE: dict = {}


@dataclass(frozen=True)
class Config:
    dataset: str
    attack: str
    seed: int
    byz_fraction: float
    l1_outlier: bool
    l2_trust: bool
    l3_kappa: bool

    def key(self) -> str:
        return (
            f"{self.dataset}|{self.attack}|{self.seed}|{self.byz_fraction}"
            f"|{int(self.l1_outlier)}{int(self.l2_trust)}{int(self.l3_kappa)}"
        )

    def layer_code(self) -> str:
        return f"{int(self.l1_outlier)}{int(self.l2_trust)}{int(self.l3_kappa)}"


# ---------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------

def _load_dataset(name: str):
    """Load and cache (X, y) for a dataset. Returns None if no file resolved."""
    if name in _DATA_CACHE:
        return _DATA_CACHE[name]
    path = DATA_PATHS.get(name)
    if not path:
        _DATA_CACHE[name] = None
        return None
    import pandas as pd
    if str(path).endswith(".npz"):
        d = np.load(path)
        X, y = d["X"].astype(float), d["y"].astype(float)
    else:
        df = pd.read_csv(path)
        if "label" in df.columns:
            y = df["label"].values.astype(float)
            X = df.drop(columns=["label"]).values.astype(float)
        else:
            y = df.iloc[:, -1].values.astype(float)
            X = df.iloc[:, :-1].values.astype(float)
    _DATA_CACHE[name] = (X, y)
    print(f"    loaded {name}: X={X.shape} fraud={y.mean():.1%}  ({path})")
    return _DATA_CACHE[name]


# =====================================================================
#  Wired to the repo trainer
# =====================================================================
def run_one(cfg: Config) -> dict:
    """
    Run one federated training job and return its metrics.

    Disabling a layer sets that layer's multiplicative contribution to the
    aggregation weight to 1.0 (Layer 2 trust) or removes its flag (Layers 1/3)
    -- never a blanket zero. Sampling, partition, optimiser, rounds and local
    epochs are identical across configurations (constants above); only the three
    layer toggles and the Byzantine fraction change.

    Returns a dict with:
        f1_minority   float %   minority-class (fraud) F1, mean of last 5 rounds
        detect_f1     float %   Byzantine-detection F1 (per client-round)
        detect_prec   float %
        detect_recall float %
        fpr_honest    float %   honest client-rounds excluded
        round_at_90   float     first round reaching 90% of own final F1
        ms_per_round  float     mean aggregation wall-clock, ms
    """
    from sklearn.preprocessing import StandardScaler
    from fl_base import make_validation_set
    from fl_runner import build_federation, run_federation
    from aggregators import MultiLayerAggregator

    data = _load_dataset(cfg.dataset)
    if data is None:
        raise FileNotFoundError(
            f"No data file resolved for dataset '{cfg.dataset}'. "
            f"Pass --data {cfg.dataset}=<path.csv> or place one of "
            f"{DATA_CANDIDATES[cfg.dataset]} under --data-dir.")
    X, y = data

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xv, yv = make_validation_set(Xs, y, min(VAL_SIZE, len(Xs)), seed=42)

    # Same global-RNG reset trick used in convergence_experiment: every config
    # starts from the same model init + partition for the given seed, so the
    # only thing that varies is the toggles / fraction we are ablating.
    np.random.seed(cfg.seed)
    server, clients, byz_ids = build_federation(
        Xs, y, scaler, Xv, yv,
        n_clients=N_CLIENTS,
        clients_per_round=PER_ROUND,
        byzantine_ratio=cfg.byz_fraction,
        attack_type=ATTACK_CODE[cfg.attack],
        noniid_alpha=NONIID_ALPHA,
        lr=LR,
        local_epochs=LOCAL_EPOCHS,
        seed=cfg.seed,
    )
    # Swap in the toggle-aware aggregator for this configuration.
    server.aggregator = MultiLayerAggregator(
        n_clients=N_CLIENTS,
        use_outlier=cfg.l1_outlier,
        use_trust=cfg.l2_trust,
        use_kappa=cfg.l3_kappa,
    )

    byz_set = set(byz_ids)
    history = run_federation(
        server, clients, byz_set,
        n_rounds=ROUNDS, p_churn=0.0, verbose=False, seed=cfg.seed)

    return _metrics_from_run(history, server.aggregator, byz_set)


def _metrics_from_run(history, aggregator, byz_set) -> dict:
    """Turn one run's history + exclusion log into the 7 reported metrics."""
    if not history:
        return {"f1_minority": 0.0, "detect_f1": 0.0, "detect_prec": 0.0,
                "detect_recall": 0.0, "fpr_honest": 0.0,
                "round_at_90": float(ROUNDS), "ms_per_round": 0.0}

    f1s = np.array([h["f1"] for h in history], float)
    rounds = np.array([h["round"] for h in history], int)
    final_f1 = float(np.mean(f1s[-5:]))

    # First round reaching 90% of own final F1 (relative convergence speed).
    if final_f1 <= 0:
        round_at_90 = float(rounds[-1])
    else:
        tgt = 0.90 * final_f1
        hit = np.where(f1s >= tgt)[0]
        round_at_90 = float(rounds[hit[0]]) if len(hit) else float(rounds[-1])

    ms_per_round = float(np.mean([h["agg_time_ms"] for h in history]))

    # Detection scored per client-round against the ACTIVE toggles' decision,
    # read from the aggregator's flag_log (excluded == weight driven to 0).
    tp = fp = fn = tn = 0
    for cids, excluded in getattr(aggregator, "flag_log", []):
        for cid, ex in zip(cids, excluded):
            is_byz = cid in byz_set
            if is_byz and ex:
                tp += 1
            elif is_byz and not ex:
                fn += 1
            elif (not is_byz) and ex:
                fp += 1
            else:
                tn += 1

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    det_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    fpr_honest = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "f1_minority":   round(final_f1, 2),
        "detect_f1":     round(det_f1 * 100, 2),
        "detect_prec":   round(prec * 100, 2),
        "detect_recall": round(rec * 100, 2),
        "fpr_honest":    round(fpr_honest * 100, 2),
        "round_at_90":   round(round_at_90, 1),
        "ms_per_round":  round(ms_per_round, 3),
    }
# =====================================================================


# ---------------------------------------------------------------------
#  Execution with checkpointing
# ---------------------------------------------------------------------

def load_done() -> dict:
    if not RESULTS.exists():
        return {}
    done = {}
    with RESULTS.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[rec["key"]] = rec
    return done


def append(rec: dict) -> None:
    with RESULTS.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def execute(configs: list[Config]) -> None:
    done = load_done()
    todo = [c for c in configs if c.key() not in done]
    # Skip datasets whose file could not be resolved (run a subset cleanly).
    missing = {c.dataset for c in todo if not DATA_PATHS.get(c.dataset)}
    if missing:
        for d in sorted(missing):
            print(f"  [skip] no data file for '{d}' "
                  f"(tried {DATA_CANDIDATES.get(d, [])})")
        todo = [c for c in todo if c.dataset not in missing]
    print(f"{len(configs)} configurations, {len(done)} already done, "
          f"{len(todo)} to run.")

    for i, cfg in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {cfg.key()}", flush=True)
        metrics = run_one(cfg)
        append({"key": cfg.key(), **asdict(cfg), **metrics})

    print("done.")


def grid_factorial() -> list[Config]:
    return [
        Config(d, a, s, MAIN_FRACTION, *layers)
        for d in DATASETS
        for a in ATTACKS
        for s in SEEDS
        for layers in FACTORIAL
    ]


def grid_sweep() -> list[Config]:
    return [
        Config(d, a, s, f, *layers)
        for d in DATASETS
        for a in ATTACKS
        for s in SEEDS[:3]          # 3 seeds for the sweep
        for f in SWEEP_FRACTIONS
        for layers in SWEEP_CONFIGS
    ]


# ---------------------------------------------------------------------
#  Statistics
# ---------------------------------------------------------------------

def cliffs_delta(a, b) -> float:
    """Non-parametric effect size in [-1, 1]. Positive means a > b."""
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def cell_means(records: list[dict], metric: str, byz: float) -> dict:
    """
    -> {layer_code: {(dataset, attack): mean over seeds}}
    Seeds are averaged within a cell before pairing, per the protocol.
    """
    acc: dict = {}
    for r in records:
        if r["byz_fraction"] != byz:
            continue
        code = f"{int(r['l1_outlier'])}{int(r['l2_trust'])}{int(r['l3_kappa'])}"
        cell = (r["dataset"], r["attack"])
        acc.setdefault(code, {}).setdefault(cell, []).append(r[metric])
    return {
        code: {cell: float(np.mean(vals)) for cell, vals in cells.items()}
        for code, cells in acc.items()
    }


def compare_to_full(records: list[dict], metric: str = "f1_minority") -> list[dict]:
    means = cell_means(records, metric, MAIN_FRACTION)
    if "111" not in means:
        sys.exit("No results for the full configuration (111). Run --stage factorial.")

    full = means["111"]
    cells = sorted(full.keys())
    rows, pvals = [], []

    for code in sorted(means):
        if code == "111":
            continue
        other = means[code]
        shared = [c for c in cells if c in other]
        if len(shared) < 6:
            continue
        x = [other[c] for c in shared]
        y = [full[c] for c in shared]
        try:
            _, p = wilcoxon(x, y)
        except ValueError:      # all differences zero
            p = 1.0
        pvals.append(float(p))
        rows.append(
            {
                "code": code,
                "n_cells": len(shared),
                "mean": float(np.mean(x)),
                "delta_vs_full": float(np.mean(x) - np.mean(y)),
                "cliffs_delta": cliffs_delta(x, y),
                "p_raw": float(p),
            }
        )

    for row, p_adj in zip(rows, holm(pvals)):
        row["p_holm"] = p_adj
    return rows


def marginal_contributions(records: list[dict], metric: str = "f1_minority") -> list[dict]:
    """
    For each layer and attack: mean F1 change from enabling that layer,
    both with the other two absent and with both present.
    """
    by = {}
    for r in records:
        if r["byz_fraction"] != MAIN_FRACTION:
            continue
        code = f"{int(r['l1_outlier'])}{int(r['l2_trust'])}{int(r['l3_kappa'])}"
        by.setdefault((r["attack"], code), []).append(r[metric])
    mean = {k: float(np.mean(v)) for k, v in by.items()}

    # position: 0 = L1, 1 = L2, 2 = L3
    def flip(code: str, pos: int, on: bool) -> str:
        c = list(code)
        c[pos] = "1" if on else "0"
        return "".join(c)

    out = []
    for pos, name in enumerate(["L1", "L2", "L3"]):
        others = "".join("0" for _ in range(3))
        for attack in ATTACKS:
            alone_on = mean.get((attack, flip(others, pos, True)))
            alone_off = mean.get((attack, flip(others, pos, False)))

            all_on = "111"
            with_on = mean.get((attack, flip(all_on, pos, True)))
            with_off = mean.get((attack, flip(all_on, pos, False)))

            out.append(
                {
                    "layer": name,
                    "attack": attack,
                    "delta_alone": (
                        None if alone_on is None or alone_off is None
                        else alone_on - alone_off
                    ),
                    "delta_with_others": (
                        None if with_on is None or with_off is None
                        else with_on - with_off
                    ),
                }
            )
    return out


# ---------------------------------------------------------------------
#  LaTeX emission
# ---------------------------------------------------------------------

def fmt(v, nd=1):
    return "---" if v is None else f"{v:.{nd}f}"


def emit(records: list[dict]) -> None:
    print("\n" + "=" * 74)
    print("TABLE: ablation_main  --  paste the row bodies")
    print("=" * 74)

    stats = {r["code"]: r for r in compare_to_full(records)}

    by_attack = {}
    for r in records:
        if r["byz_fraction"] != MAIN_FRACTION:
            continue
        code = f"{int(r['l1_outlier'])}{int(r['l2_trust'])}{int(r['l3_kappa'])}"
        by_attack.setdefault(code, {}).setdefault(r["attack"], []).append(r)

    for code in ["000", "100", "010", "001", "110", "101", "011", "111"]:
        if code not in by_attack:
            continue
        marks = " & ".join(
            r"\checkmark" if c == "1" else r"\ding{55}" for c in code
        )
        cells = []
        for a in ATTACKS:
            rs = by_attack[code].get(a, [])
            if rs:
                vals = [x["f1_minority"] for x in rs]
                cells.append(f"{np.mean(vals):.1f}$\\pm${np.std(vals):.1f}")
            else:
                cells.append("---")

        allr = [x for a in ATTACKS for x in by_attack[code].get(a, [])]
        det = fmt(np.mean([x["detect_f1"] for x in allr])) if allr else "---"
        fpr = fmt(np.mean([x["fpr_honest"] for x in allr])) if allr else "---"
        r90 = fmt(np.mean([x["round_at_90"] for x in allr])) if allr else "---"
        ms = fmt(np.mean([x["ms_per_round"] for x in allr]), 2) if allr else "---"

        if code == "111":
            p_s, d_s = "---", "---"
        else:
            st = stats.get(code)
            p_s = fmt(st["p_holm"], 3) if st else "---"
            d_s = fmt(st["cliffs_delta"], 2) if st else "---"

        print(
            f"{marks} & " + " & ".join(cells) +
            f" & {det} & {fpr} & {r90} & {ms} & {p_s} & {d_s} \\\\"
        )

    print("\n" + "=" * 74)
    print("TABLE: ablation_marginal  --  paste the row bodies")
    print("=" * 74)
    for m in marginal_contributions(records):
        print(
            f"{m['layer']} & {m['attack']} & "
            f"{fmt(m['delta_alone'])} & {fmt(m['delta_with_others'])} &  \\\\"
        )

    print("\n" + "=" * 74)
    print("SANITY CHECKS")
    print("=" * 74)

    full = [r for r in records
            if r["byz_fraction"] == MAIN_FRACTION
            and (r["l1_outlier"], r["l2_trust"], r["l3_kappa"]) == (True, True, True)]
    if full:
        fpr = np.mean([r["fpr_honest"] for r in full])
        print(f"  Full framework honest-client FPR: {fpr:.1f}%")
        if fpr > 10:
            print("    ^ report this prominently; it is a real deployment cost")

    sig = [r for r in compare_to_full(records) if r["p_holm"] < 0.05]
    print(f"  Configurations differing from full after Holm: {len(sig)} of 7")
    if not sig:
        print("    ^ no configuration separates at f=0.3. The fraction sweep")
        print("      is then your result -- see write-up guidance case (d).")

    l3 = [m for m in marginal_contributions(records)
          if m["layer"] == "L3" and m["delta_with_others"] is not None]
    if l3:
        lfp = [m for m in l3 if m["attack"] == "lfp"]
        if lfp and lfp[0]["delta_with_others"] is not None:
            print(f"  L3 marginal on LFP (with others): "
                  f"{lfp[0]['delta_with_others']:+.1f} F1")
            print("    ^ if small, use write-up guidance case (b). Do not bury it.")


# ---------------------------------------------------------------------
#  Dataset path resolution
# ---------------------------------------------------------------------

def resolve_data_paths(data_dir: str, overrides: list[str]) -> None:
    ov = {}
    for spec in overrides or []:
        if "=" not in spec:
            sys.exit(f"--data expects dataset=path, got '{spec}'")
        name, path = spec.split("=", 1)
        if name not in DATASETS:
            sys.exit(f"--data unknown dataset '{name}' (choose from {DATASETS})")
        ov[name] = path
    for name in DATASETS:
        if name in ov:
            DATA_PATHS[name] = ov[name] if os.path.exists(ov[name]) else None
            if DATA_PATHS[name] is None:
                print(f"  [warn] override for '{name}' not found: {ov[name]}")
            continue
        found = None
        for cand in DATA_CANDIDATES[name]:
            p = os.path.join(data_dir, cand) if data_dir else cand
            if os.path.exists(p):
                found = p
                break
        DATA_PATHS[name] = found


def main() -> None:
    global DATASETS, ATTACKS, SEEDS, N_CLIENTS, PER_ROUND, ROUNDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["factorial", "sweep", "report"],
                    required=True)
    ap.add_argument("--data-dir", default=".",
                    help="directory holding the per-dataset CSVs")
    ap.add_argument("--data", nargs="+", default=None,
                    help="per-dataset path overrides, e.g. nyc_taxi=nyc.csv")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="restrict the grid to these datasets "
                         f"(subset of {DATASETS})")
    ap.add_argument("--attacks", nargs="+", default=None,
                    help=f"restrict the grid to these attacks (subset of {ATTACKS})")
    ap.add_argument("--seeds", type=int, default=None,
                    help="number of seeds to use (default 5 factorial / 3 sweep)")
    ap.add_argument("--rounds", type=int, default=None,
                    help=f"FL rounds per run (default {ROUNDS})")
    ap.add_argument("--clients", type=int, default=None,
                    help=f"total clients (default {N_CLIENTS})")
    ap.add_argument("--per-round", type=int, default=None, dest="per_round",
                    help=f"clients sampled per round (default {PER_ROUND})")
    args = ap.parse_args()

    # Apply grid / protocol overrides before building the grid.
    if args.datasets:
        bad = [d for d in args.datasets if d not in DATASETS]
        if bad:
            sys.exit(f"--datasets unknown: {bad} (choose from {DATASETS})")
        DATASETS = list(args.datasets)
    if args.attacks:
        bad = [a for a in args.attacks if a not in ATTACKS]
        if bad:
            sys.exit(f"--attacks unknown: {bad} (choose from {ATTACKS})")
        ATTACKS = list(args.attacks)
    if args.seeds is not None:
        SEEDS = list(range(args.seeds))
    if args.rounds is not None:
        ROUNDS = args.rounds
    if args.clients is not None:
        N_CLIENTS = args.clients
    if args.per_round is not None:
        PER_ROUND = args.per_round

    resolve_data_paths(args.data_dir, args.data)

    if args.stage == "factorial":
        execute(grid_factorial())
    elif args.stage == "sweep":
        execute(grid_sweep())
    else:
        recs = list(load_done().values())
        if not recs:
            sys.exit("No results yet.")
        emit(recs)


if __name__ == "__main__":
    main()
