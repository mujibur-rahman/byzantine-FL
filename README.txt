Sync bundle — overwrite your local D:\experiment\byzantine-FL with these 16
files (preserve paths). They carry every fix/feature from this session; your
local repo was several commits behind (that's why --data was ignored and
experiment2 still had the old bug).

Key fixes included:
  dataset_config.py    --data / DATA_FILE real-data override (was missing:
                       your --data runs silently used SYNTHETIC data)
  experiment2/3/5/6    the training-API / noise / accumulator bug fixes
  aggregators.py, fl_server.py, fl_runner.py  async + staleness-corrected kappa
  divergence_*, prep_*, robustness_sweep.py   the analysis/figure tools

After copying, verify real data loads (look for "(real data override)" +
your row count):
  python3 experiment2_sensitivity.py --dataset nyc-taxi --data nyc_real100000.csv


###baseline_figures.py
python3 baseline_figures.py --csv NYC-Taxi=results/nyc-taxi/nyc-taxi_table_baselines.csv Foursquare=results/foursquare/foursquare_table_baselines.csv Yelp=results/yelp/yelp_table_baselines.csv Geolife=results/geolife/geolife_table_baselines.csv --out fig/baselines

####
python3 layer_figures.py --csv NYC-Taxi=results/nyc-taxi/nyc-taxi_layer_confusion.csv Foursquare=results/foursquare/foursquare_layer_confusion.csv Yelp=results/yelp/yelp_layer_confusion.csv Geolife=results/geolife/geolife_geolife_layer_confusion.csv --out fig/layers --byz 20

####
python3 experiment4_overhead.py

####
python3 experiment5_adaptive.py  --dataset yelp --data .\data\divergence-ds\yelp_real100000.csv
