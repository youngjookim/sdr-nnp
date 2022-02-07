import metrics
from glob import glob
import sys
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

def save_plot(figname, X_2d_train, X_2d_nnp_train, X_2d_nnp_test, y_train=None, y_test=None, point_size=5):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.tight_layout()

    ax[0].scatter(X_2d_train[:,0], X_2d_train[:,1], s=point_size, c=y_train)
    ax[1].scatter(X_2d_nnp_train[:,0], X_2d_nnp_train[:,1], s=point_size, c=y_train)
    ax[2].scatter(X_2d_nnp_test[:,0], X_2d_nnp_test[:,1], s=point_size, c=y_test)

    for j in range(3):
        ax[j].axis('off')

    ax[0].set_title('2D orig train')
    ax[1].set_title('2D NNP train')
    ax[2].set_title('2D NNP test')

    fig.savefig(figname)

def read_data(this_path):
    f1 = open(this_path)
    data=np.genfromtxt(f1.readlines(),dtype='float32')
    return data

def read_header(this_path):
    fp = open(this_path)
    str_header = fp.readline().split()
    return int(str_header[0]), int(str_header[1])

def compute_all_metrics(X, X_2d, D_high, D_low, y=None):
    T = metrics.metric_trustworthiness(X, X_2d, D_high, D_low)
    C = metrics.metric_continuity(X, X_2d, D_high, D_low)
    R = metrics.metric_shepard_diagram_correlation(D_high, D_low)
    S = metrics.metric_normalized_stress(D_high, D_low)

    if y is not None:
        N = metrics.metric_neighborhood_hit(X_2d, y, k=3)
    else:
        N = -1.0

    return T, C, R, S, N

def compute_distance_memoized(X, file_name):
    if os.path.exists(file_name):
        print('Loading {0}'.format(file_name))
        D = np.load(file_name)
    else:
        D = metrics.compute_distance_list(X)
        np.save(file_name, D)
    
    return D

if __name__ == "__main__":
    data_dir = sys.argv[1]#'datasets'#
    datasets = sys.argv[2].split(',')#['GALAHDR2']#

    ## PARAMETERS
    random_state=420
    test_size=0.2
    point_size = 5

    ## input files
    str_i_data='data_.txt'              # original nd data
    str_i_header='data__header.txt'     # header with #of observations and the #of dimensions
    str_i_labels='data__labels.txt'     # labels of the data
    str_i_sdr='data__s_dr.txt'          # the sharpened dr results

    for dataset in datasets:
        print(dataset)
        dirs = sorted(glob(data_dir + '/{0}_*'.format(dataset)))#'/_sdr/{0}_*'.format(dataset)))

        D_high_train = None
        D_high_test = None

        src_dataset_path = os.path.join(data_dir, dataset)

        for d in dirs:
            this_dir = os.path.join(d, '')#'data'

            assert os.path.exists(this_dir+ '/' + str_i_data), 'main(): No input data file.'
            data_orig=read_data(this_dir+'/' + str_i_data) #data_orig is original nD

            if os.path.exists(this_dir+'/' + str_i_header):
                n_obs, n_dim = read_header(this_dir+'/' + str_i_header)
                assert ((data_orig.shape[0]==n_obs) & (data_orig.shape[1]==n_dim)), 'main(): the number of observations and dimensions are different in data and header files.'
            else:
                n_obs, n_dim = data_orig.shape[0], data_orig.shape[1]

            assert os.path.exists(this_dir+'/' + str_i_sdr), 'main(): No input data file.'
            data_sdr_2d=read_data(this_dir+'/' + str_i_sdr)   #data_sdr_2d --> 2D sdr
            
            if os.path.exists(this_dir+ '/' + str_i_labels):
                labels = read_data(this_dir+'/' + str_i_labels)
            else:
                labels = None 

            scaler=MinMaxScaler()
             
            data_orig=scaler.fit_transform(data_orig)
            data_sdr_2d=scaler.fit_transform(data_sdr_2d)
                
            X_train_orig_sdr, X_test_orig_sdr, X_train_sdr, X_test_sdr = train_test_split(data_orig, data_sdr_2d, test_size=test_size, random_state=random_state)

            if labels is not None:
                _, _, y_train, y_test = train_test_split(data_orig, labels, test_size=test_size, random_state=random_state)
            else:
                print('No labels')
                y_train = None
                y_test = None
            
            if D_high_train is None:
                print('Computing D_high_train')
                D_high_train = compute_distance_memoized(X_train_orig_sdr, os.path.join(this_dir, 'D_high_train.npy'))

            if D_high_test is None:
                print('Computing D_high_test')
                D_high_test = compute_distance_memoized(X_test_orig_sdr, os.path.join(this_dir, 'D_high_test.npy'))

            train_data = sorted(glob(this_dir + '/NNP_*_train_*.txt'))
            test_data = sorted(glob(this_dir + '/NNP_*_test_*.txt'))

            results = []

            for tr, te in zip(train_data, test_data):
                X_train = read_data(tr)
                print('Computing D_low_train {0}'.format(tr))
                D_low_train = compute_distance_memoized(X_train, tr + '_D_low_train.npy')

                print('Computing D_low_test {0}'.format(te))
                X_test = read_data(te)
                D_low_test = compute_distance_memoized(X_test, te + '_D_low_test.npy')

                results.append((dataset, tr, 'train', ) + compute_all_metrics(X_train_orig_sdr, X_train, D_high_train, D_low_train, y_train))
                results.append((dataset, te, 'test', ) + compute_all_metrics(X_test_orig_sdr, X_test, D_high_test, D_low_test, y_test))

                save_plot(tr + '_plot.png', X_train_sdr, X_train, X_test, y_train=y_train, y_test=y_test, point_size=point_size)

            df = pd.DataFrame(results, columns=[    'dataset',
                                                    'path',
                                                    'type',
                                                    'T',
                                                    'C',
                                                    'R',
                                                    'S',
                                                    'N'])
            df.to_csv(os.path.join(this_dir, 'metrics.csv'), header=True, index=None)

