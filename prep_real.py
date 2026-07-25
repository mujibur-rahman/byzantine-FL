#!/usr/bin/env python3
"""
prep_real.py — convert real Geolife / Foursquare / Yelp data into the
10-feature X,y format used by the experiments / divergence_ablation.py.

Reuses the repo's tested real-data loaders (they handle the raw formats:
Geolife .plt trajectories, Foursquare TSMC .txt check-ins, Yelp JSON), then
exports npz + csv. RUN FROM THE REPO DIRECTORY (imports the loaders).

Fraud labels are heuristic (real data is unlabelled) — state this in the paper.

Usage:
  python3 prep_real.py --dataset geolife    --path ~/data/geolife/Data          --out geolife_real
  python3 prep_real.py --dataset foursquare --path ~/data/foursquare/dataset_TSMC2014_NYC.txt --out foursquare_real
  python3 prep_real.py --dataset yelp       --path ~/data/yelp/yelp_academic_dataset_business.json \
                       --aux ~/data/yelp/yelp_academic_dataset_checkin.json --out yelp_real
  # optional: --n 80000  cap rows

Then:
  python3 divergence_ablation.py --data geolife_real.npz --seeds 5
"""
import argparse, os
import numpy as np
import pandas as pd

FEATURES = ['pickup_lat','pickup_lon','dropoff_lat','dropoff_lon','duration_s',
            'distance','cost_proxy','hour_of_day','cost_per_unit','duration_per_unit']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['geolife','foursquare','yelp'])
    ap.add_argument('--path', required=True, help='real data file/dir (see --help examples)')
    ap.add_argument('--aux', default=None, help='yelp only: checkin JSON path')
    ap.add_argument('--n', type=int, default=80_000)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or f'{a.dataset}_real'

    # DATASET_NAME is resolved at import time from this env var
    os.environ['DATASET'] = a.dataset

    if a.dataset == 'geolife':
        from geolife_loader import load_geolife
        X, y = load_geolife(data_dir=a.path, n_samples=a.n)
    elif a.dataset == 'foursquare':
        from foursquare_loader import load_foursquare
        X, y = load_foursquare(path=a.path, n_samples=a.n)
    else:  # yelp
        if not a.aux:
            ap.error('yelp needs --aux <checkin JSON path>')
        from yelp_loader import load_yelp
        X, y = load_yelp(business_path=a.path, checkin_path=a.aux, n_samples=a.n)

    X = np.asarray(X, float); y = np.asarray(y, float)
    if X.shape[1] != 10:
        raise SystemExit(f'expected 10 features, got {X.shape[1]}')

    df = pd.DataFrame(X, columns=FEATURES); df['label'] = y.astype(int)
    df.to_csv(f'{out}.csv', index=False)
    np.savez_compressed(f'{out}.npz', X=X, y=y)
    print(f'{a.dataset}: {len(y):,} rows  fraud_rate={y.mean():.2%}')
    print(f'wrote {out}.csv and {out}.npz')

if __name__ == '__main__':
    main()
