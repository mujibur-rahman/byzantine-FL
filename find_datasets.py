"""
find_datasets.py
================
Scans your filesystem for the four dataset files and prints
the exact paths to paste into dataset_config.py.

Run:  python3 find_datasets.py
"""

import os
import sys

# ── Files we are looking for ─────────────────────────────────────────────────
TARGETS = {
    'nyc-taxi': [
        'fhvhv_tripdata_2025-01.parquet',
        'fhvhv_tripdata_2024-12.parquet',
        'fhvhv_tripdata_2024-11.parquet',
        'fhvhv_tripdata_2024-10.parquet',
        'yellow_tripdata_2025-01.parquet',
        'yellow_tripdata_2024-12.parquet',
        'taxi_zone_lookup.csv',
    ],
    'geolife': [
        'Geolife Trajectories 1.3',
        'GeolifeTrajectories1.3',
        'Data',          # the Data/ subfolder inside geolife
    ],
    'foursquare': [
        'dataset_TSMC2014_NYC.txt',
        'dataset_TSMC2014_TKY.txt',
        'checkins.txt',
        'raw_Checkins.txt',
        'raw_POIs.txt',
    ],
    'yelp': [
        'yelp_academic_dataset_business.json',
        #'yelp_academic_dataset_checkin.json',
    ],
}

# ── Search roots (common places people put data) ─────────────────────────────
HOME = os.path.expanduser('~')
SEARCH_ROOTS = [
    HOME,
    os.path.join(HOME, 'data'),
    os.path.join(HOME, 'datasets'),
    os.path.join(HOME, 'Downloads'),
    os.path.join(HOME, 'Desktop'),
    os.path.join(HOME, 'Documents'),
    os.path.join(HOME, 'research'),
    os.path.join(HOME, 'Research'),
    os.path.join(HOME, 'PhD'),
    os.path.join(HOME, 'projects'),
    os.path.join(HOME, 'Projects'),
    '/mnt',
    '/data',
    '/datasets',
    'C:\\Users\\' + os.environ.get('USERNAME', '') + '\\Downloads',
    'C:\\Users\\' + os.environ.get('USERNAME', '') + '\\Desktop',
    'C:\\data',
    'D:\\data',
    'D:\\datasets',
    '.',
    '..',
]

MAX_DEPTH = 6   # how deep to search in each root

# ── Recursive file finder ─────────────────────────────────────────────────────
def find_file(filename, search_roots, max_depth=MAX_DEPTH):
    """Return list of absolute paths where filename was found."""
    found = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Limit depth
            depth = dirpath.replace(root, '').count(os.sep)
            if depth > max_depth:
                dirnames.clear()
                continue
            # Skip hidden dirs and common non-data dirs
            dirnames[:] = [d for d in dirnames
                           if not d.startswith('.')
                           and d not in ('node_modules', '__pycache__',
                                         '.git', 'venv', 'env', '.venv')]
            if filename in filenames:
                found.append(os.path.join(dirpath, filename))
            # Also check if a directory matches (for geolife Data/)
            if filename in dirnames:
                found.append(os.path.join(dirpath, filename))
    return found

# ── Run search ────────────────────────────────────────────────────────────────
print("Searching for dataset files...")
print("(This may take 10-30 seconds depending on disk size)\n")

found_map = {}   # dataset -> {filename -> [paths]}

for dataset, filenames in TARGETS.items():
    found_map[dataset] = {}
    for fname in filenames:
        hits = find_file(fname, SEARCH_ROOTS)
        if hits:
            found_map[dataset][fname] = hits

# ── Report ────────────────────────────────────────────────────────────────────
print("=" * 68)
print("  DATASET FILE SEARCH RESULTS")
print("=" * 68)

config_updates = {}   # what to put in dataset_config.py

for dataset, files in found_map.items():
    print(f"\n[{dataset}]")
    if not files:
        print(f"  ✗  No files found.")
        print(f"     Check SEARCH_ROOTS in this script if data is elsewhere.")
        config_updates[dataset] = None
        continue

    for fname, paths in files.items():
        for p in paths:
            print(f"  ✓  {fname}")
            print(f"     {p}")

    # Build config suggestion
    if dataset == 'nyc-taxi':
        parquet = next(
            (paths[0] for f, paths in files.items()
             if f.endswith('.parquet')), None)
        lookup  = next(
            (paths[0] for f, paths in files.items()
             if f == 'taxi_zone_lookup.csv'), None)
        config_updates[dataset] = {'parquet': parquet, 'lookup': lookup}

    elif dataset == 'geolife':
        # Look for the Data/ subfolder
        data_dir = next(
            (paths[0] for f, paths in files.items()
             if f == 'Data'), None)
        if data_dir is None:
            # Maybe the root folder is the geolife dir
            any_path = next(
                (paths[0] for f, paths in files.items()), None)
            if any_path:
                data_dir = os.path.dirname(any_path)
        config_updates[dataset] = {'data_dir': data_dir}

    elif dataset == 'foursquare':
        txt = next(
            (paths[0] for f, paths in files.items()
             if f.endswith('.txt')), None)
        config_updates[dataset] = {'path': txt}

    elif dataset == 'yelp':
        biz = next(
            (paths[0] for f, paths in files.items()
             if 'business' in f), None)
        chk = next(
            (paths[0] for f, paths in files.items()
             if 'checkin' in f), None)
        config_updates[dataset] = {'business': biz, 'checkin': chk}

# ── Print dataset_config.py update block ─────────────────────────────────────
print("\n" + "=" * 68)
print("  PASTE THIS INTO dataset_config.py  (replace DATASET_REGISTRY)")
print("=" * 68)
print()
print("DATASET_REGISTRY = {")

nyc  = config_updates.get('nyc-taxi')   or {}
geo  = config_updates.get('geolife')    or {}
fsq  = config_updates.get('foursquare') or {}
yelp = config_updates.get('yelp')       or {}

def q(v):
    return f"'{v}'" if v else "None  # NOT FOUND — check path manually"

print(f"""
    'nyc-taxi': {{
        'description':  'NYC TLC High-Volume FHV trips (2025)',
        'real_path':    {q(nyc.get('parquet'))},
        'aux_path':     {q(nyc.get('lookup'))},
        'n_samples':    100_000,
        'download_url': 'https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page',
        'fraud_rate':   0.15,
    }},

    'geolife': {{
        'description':  'Microsoft Geolife GPS Trajectories',
        'real_path':    {q(geo.get('data_dir'))},
        'aux_path':     None,
        'n_samples':    80_000,
        'download_url': 'https://www.microsoft.com/en-us/download/details.aspx?id=52367',
        'fraud_rate':   0.12,
    }},

    'foursquare': {{
        'description':  'Foursquare NYC check-in dataset (Yang et al. 2014)',
        'real_path':    {q(fsq.get('path'))},
        'aux_path':     None,
        'n_samples':    80_000,
        'download_url': 'https://sites.google.com/site/yangdingqi/home/foursquare-dataset',
        'fraud_rate':   0.13,
    }},

    'yelp': {{
        'description':  'Yelp Open Dataset',
        'real_path':    {q(yelp.get('business'))},
        'aux_path':     {q(yelp.get('checkin'))},
        'n_samples':    80_000,
        'download_url': 'https://business.yelp.com/data/resources/open-dataset/',
        'fraud_rate':   0.13,
    }},
}}""")

print()
print("=" * 68)
print("  AFTER UPDATING, run:  python3 dataset_config.py")
print("  to verify all paths are found.")
print("=" * 68)
