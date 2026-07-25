#!/usr/bin/env python3
"""
prep_nyc_real.py — convert real NYC yellow-taxi CSV (2015 schema) into the
10-feature X,y format used by the experiments / divergence_ablation.py.

Real trips have no fraud label, so one is derived with the repo's heuristic
(same as nyc_taxi_loader._fraud_label).

Usage:
  python3 prep_nyc_real.py --csv yellow_2015.csv --out nyc_real
  # -> nyc_real.csv (10 features + label) and nyc_real.npz (X, y)
  python3 prep_nyc_real.py --csv yellow_2015.csv --out nyc_real --n 100000
"""
import argparse
import numpy as np
import pandas as pd

FEATURES = ['pickup_lat','pickup_lon','dropoff_lat','dropoff_lon','duration_s',
            'distance','cost_proxy','hour_of_day','cost_per_unit','duration_per_unit']

def fraud_label(fare, dist, dur, hour):
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
    X['label'] = fraud_label(fare.values, dist.values, dur.values, hour.values)

    # ── Clean: valid NYC bbox, positive duration/distance, drop NaN ──
    m = (
        X['duration_s'].between(30, 4*3600) &
        X['distance'].between(0.1, 100) &
        X['cost_proxy'].between(2.0, 500) &
        X['pickup_lat'].between(40.4, 41.0) & X['pickup_lon'].between(-74.3, -73.6) &
        X['dropoff_lat'].between(40.4, 41.0) & X['dropoff_lon'].between(-74.3, -73.6) &
        X[FEATURES].notna().all(axis=1)
    )
    X = X[m].reset_index(drop=True)
    if a.n and len(X) > a.n:
        X = X.sample(a.n, random_state=42).reset_index(drop=True)

    X.to_csv(f'{a.out}.csv', index=False)
    np.savez_compressed(f'{a.out}.npz',
                        X=X[FEATURES].values.astype(float),
                        y=X['label'].values.astype(float))
    print(f'kept {len(X):,} rows  fraud_rate={X["label"].mean():.2%}')
    print(f'wrote {a.out}.csv and {a.out}.npz')

if __name__ == '__main__':
    main()
