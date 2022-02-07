import os
import sys
import shutil
from glob import glob
import pandas as pd

data_dir = sys.argv[1]

if not os.path.exists('results'):
    os.makedirs('results')

nn_sizes = {'xsmall': 0, 'small': 1, 'medium': 2}

metric_files = glob(data_dir + '/**/metrics.csv', recursive=True)
dfs = []

for f in metric_files:
    df = pd.read_csv(f)
    dfs.append(df)

df_all = pd.concat(dfs)

df_sdr = df_all['path'].str.split('/', expand=True)[6].str.split('_', expand=True)
df_nnp = df_all['path'].str.split('/', expand=True)[8].str.split('_', expand=True)

df_sdr.columns = ['dataset_sdr', 'projection', 'sdr_n_iter', 'sdr_lr']
df_nnp = df_nnp.loc[:,[2, 6, 5]]
df_nnp.columns = ['nnp_size', 'nnp_epochs', 'nnp_mode']

df_nnp['nnp_epochs'] = df_nnp['nnp_epochs'].str.replace('000early.txt', '0').astype('int')

df_all.reset_index(drop=True, inplace=True)
df_sdr.reset_index(drop=True, inplace=True)
df_nnp.reset_index(drop=True, inplace=True)

df_x = pd.concat([df_all, df_sdr, df_nnp], axis=1)
df_x = df_x.loc[:,['dataset', 'projection', 'sdr_n_iter', 'sdr_lr', 'nnp_size', 'nnp_epochs', 'nnp_mode', 'T', 'C', 'R', 'S', 'N']]

df_x.to_csv('results/all_metrics.csv', index=None)



png_files = glob(data_dir + '/**/*.png', recursive=True)

for p in png_files:
    algof, _, nnpf = p.split('/')[-3:]
    dataset, algo, n_iter, lr = algof.split('_')
    nns = nnpf.split('_')
    nn_size = nns[2]
    nn_epochs = int(nns[6].replace('early.txt', ''))
    lr = float(lr)
    n_iter = int(n_iter)

    fmt = {'algo': algo, 'dataset': dataset, 'nn_size_int': nn_sizes[nn_size], 'nn_size': nn_size, 'nn_epochs': str(nn_epochs).zfill(4), 'n_iter': str(n_iter).zfill(2), 'lr': '{:.2f}'.format(lr)}

    path = 'results/{0}'.format(dataset)
    if not os.path.exists(path):
        os.makedirs(path)

    new_png = '{dataset}_{algo}_{n_iter}_{lr}_{nn_size_int}-{nn_size}_{nn_epochs}.png'.format(**fmt)
    print(p, new_png)
    shutil.copy(p, os.path.join(path, new_png))


