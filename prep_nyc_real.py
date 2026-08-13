#!/usr/bin/env python3
"""
prep_nyc_real.py — convert real NYC yellow-taxi CSV (2015 schema) into the
10-feature X,y format used by the experiments / divergence_ablation.py.

Real trips have no fraud label, so one is DERIVED heuristically. Two modes:

  --mode quantile (default): compute a continuous "suspicion" severity score
      from the same fraud signatures (fare-per-mile gouging, fare-per-second
      phantom/short trips, short-trip fare inflation), robustly standardise it,
      and label the top --fraud-rate fraction as fraud. This GUARANTEES the
      target prevalence (e.g. 0.17) and keeps the fraud class a coherent,
      learnable feature signature (needed so the global model is not degenerate).

  --mode rule: the original fixed-threshold rule (~9% on 2015 data), kept for
      reproducibility.

NOTE (disclose in the paper): these labels are heuristic proxies — the source
trips are unlabeled. The prevalence is a calibrated design choice, not a
ground-truth fraud rate.

Usage:
  python3 prep_nyc_real.py --csv yellow_2015.csv --out nyc_real --fraud-rate 0.17
  python3 prep_nyc_real.py --csv yellow_2015.csv --out nyc_real --n 100000 --fraud-rate 0.18
  python3 prep_nyc_real.py --csv yellow_2015.csv --out nyc_real --mode rule
  # -> nyc_real.csv (10 features + label) and nyc_real.npz (X, y)
"""
import argparse
import numpy as np
import pandas as pd

FEATURES = ['pickup_lat','pickup_lon','dropoff_lat','dropoff_lon','duration_s',
            'distance','cost_proxy','hour_of_day','cost_per_unit','duration_per_unit']


def robust_z(v):
    """Median/MAD standardisation — outlier-resistant, so a few extreme trips
    don't dominate the score."""
    v = np.asarray(v, float)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-9
    return (v - med) / (1.4826 * mad)


def suspicion_score(fare, dist, dur):
    """Continuous fraud-severity score from the three fraud signatures the
    rule encodes. Higher = more fraud-like."""
    dist_ = np.maximum(dist, 0.01)
    dur_  = np.maximum(dur, 1.0)
    fpm = fare / dist_                       # fare per mile   (gouging)
    fps = fare / dur_                        # fare per second (phantom/short)
    short_inflate = (fare / dist_) * (dist_ < 1.0)   # short-trip inflation
    return robust_z(fpm) + robust_z(fps) + 0.5 * robust_z(short_inflate)


def fraud_label_quantile(fare, dist, dur, rate):
    """Label the top `rate` fraction (by suspicion score) as fraud → exact
    target prevalence."""
    score = suspicion_score(fare, dist, dur)
    thr = np.quantile(score, 1.0 - rate)
    return (score >= thr).astype(int)


def fraud_label_rule(fare, dist, dur, hour):
    """Original fixed-threshold rule (~9% on 2015 data)."""
    fpm = fare / np.maximum(dist, 0.01)
    lab = np.zeros(len(fare), int)
    lab[(fpm > 8.0) & (dist < 1.0)] = 1          # gouging on short trip
    lab[(dur < 60) & (fare > 15)]  = 1           # phantom trip
    lab[(fare > 120) & (dist < 0.5)] = 1         # spoof-inflated fare
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='nyc_real')
    ap.add_argument('--n', type=int, default=None, help='optional row cap after cleaning')
    ap.add_argument('--mode', choices=['quantile', 'rule'], default='quantile')
    ap.add_argument('--fraud-rate', type=float, default=0.17, dest='fraud_rate',
                    help='target fraud prevalence for --mode quantile (e.g. 0.15-0.20)')
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    df.columns = [c.strip() for c in df.columns]

    def col(*names):
        for n in names:
            if n in df.columns: return df[n]
        raise KeyError(f'need one of {names}; have {list(df.columns)[:8]}...')

    pu = pd.to_datetime(col('tpep_pickup_datetime','pickup_datetime'), dayfirst=True, errors='coerce')
    do = pd.to_datetime(col('tpep_dropoff_datetime','dropoff_datetime'), dayfirst=True, errors='coerce')
    dur = (do - pu).dt.total_seconds()

    dist = col('trip_distance').astype(float)
    fare = col('fare_amount','total_amount').astype(float)
    plat = col('pickup_latitude').astype(float);  plon = col('pickup_longitude').astype(float)
    dlat = col('dropoff_latitude').astype(float); dlon = col('dropoff_longitude').astype(float)
    hour = pu.dt.hour.astype(float)

    X = pd.DataFrame({
        'pickup_lat': plat, 'pickup_lon': plon,
        'dropoff_lat': dlat, 'dropoff_lon': dlon,
        'duration_s': dur, 'distance': dist, 'cost_proxy': fare,
        'hour_of_day': hour,
        'cost_per_unit': fare / np.maximum(dist, 0.1),
        'duration_per_unit': dur / np.maximum(dist, 0.1),
    })

    # ── Clean FIRST, so the labelled prevalence is measured on the kept rows ──
    m = (
        X['duration_s'].between(30, 4*3600) &
        X['distance'].between(0.1, 100) &
        X['cost_proxy'].between(2.0, 500) &
        X['pickup_lat'].between(40.4, 41.0) & X['pickup_lon'].between(-74.3, -73.6) &
        X['dropoff_lat'].between(40.4, 41.0) & X['dropoff_lon'].between(-74.3, -73.6) &
        X[FEATURES].notna().all(axis=1)
    )
    X = X[m].reset_index(drop=True)

    # ── Derive label on the cleaned set ──
    f = X['cost_proxy'].values          # fare
    d = X['distance'].values
    u = X['duration_s'].values
    if a.mode == 'quantile':
        X['label'] = fraud_label_quantile(f, d, u, a.fraud_rate)
    else:
        X['label'] = fraud_label_rule(f, d, u, X['hour_of_day'].values)

    if a.n and len(X) > a.n:
        X = X.sample(a.n, random_state=42).reset_index(drop=True)

    X.to_csv(f'{a.out}.csv', index=False)
    np.savez_compressed(f'{a.out}.npz',
                        X=X[FEATURES].values.astype(float),
                        y=X['label'].values.astype(float))
    print(f'mode={a.mode}'
          + (f'  target={a.fraud_rate:.0%}' if a.mode == 'quantile' else ''))
    print(f'kept {len(X):,} rows  fraud_rate={X["label"].mean():.2%}')
    print(f'wrote {a.out}.csv and {a.out}.npz')


if __name__ == '__main__':
    main()
