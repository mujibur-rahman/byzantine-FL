#!/usr/bin/env python3
"""Build the combined convergence LaTeX table (method x dataset, mean±std)
from the per-dataset *_summary.csv written by convergence_experiment.py."""
import pandas as pd

DS = [('NYC-Taxi', 'conv_nyc_summary.csv'), ('Foursquare', 'conv_4sq_summary.csv'),
      ('Yelp', 'conv_yelp_summary.csv'), ('Geolife', 'conv_geo_summary.csv')]
METHODS = ['FedAvg', 'TrimmedMean', 'RFVIR', 'Multi-Krum', 'Bulyan', 'FLAME', 'Ours']

d = {name: pd.read_csv(p).set_index('method') for name, p in DS}
NL = r' \\'


def cell(name, m, col):
    try:
        return str(d[name].loc[m, col]).replace('±', r'$\pm$')
    except Exception:
        return '--'


L = [r'\begin{table*}[t]', r'\centering',
     r'\caption{Convergence speed across the four datasets (clean setting, 100 '
     r'clients, 20 sampled/round, 50 rounds; mean$\pm$s.d.\ over 3 seeds). '
     r'\textbf{Acc.}: final accuracy (last-5-round mean). \textbf{R@90}: first '
     r'round reaching 90\% of its own final accuracy. \textbf{AUC}: '
     r'mean accuracy over all rounds. Large R@90 spreads reflect '
     r'random-initialisation variance, not method instability; final accuracy '
     r'and AUC are the stable metrics. Geolife accuracy equals the '
     r'majority-class baseline (minority-class F1$\to$0).}',
     r'\label{tab:convergence}', r'\small',
     r'\begin{tabular}{l' + 'ccc' * 4 + r'}', r'\toprule',
     ' & ' + ' & '.join(r'\multicolumn{3}{c}{\textbf{' + n + r'}}' for n, _ in DS) + NL,
     r'\textbf{Method} & ' + ' & '.join(['Acc. & R@90 & AUC'] * 4) + NL, r'\midrule']

for m in METHODS:
    if m == 'Ours':
        L.append(r'\midrule')
    cells = []
    for name, _ in DS:
        cells += [cell(name, m, 'final_acc'), cell(name, m, 'round@90%'),
                  cell(name, m, 'auc')]
    if m == 'Ours':
        cells = [r'\textbf{' + c + '}' for c in cells]
        label = r'\textbf{Ours}'
    else:
        label = m
    L.append(label + ' & ' + ' & '.join(cells) + NL)

L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
tex = '\n'.join(L) + '\n'
open('convergence_table.tex', 'w').write(tex)
print(tex)
