from glob import glob
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import os
from nnp import NNP

def read_data(this_path):
    f1 = open(this_path)
    data=np.genfromtxt(f1.readlines(),dtype='float32')
    return data

def read_header(this_path):
    fp = open(this_path)
    str_header = fp.readline().split()
    return int(str_header[0]), int(str_header[1])

if __name__ == "__main__":
    input_dir = "datasets_202109" #sys.argv[1] #
    datasets = ["galah"] #sys.argv[2].split(',') #

    ## PARAMETERS
    random_state=420
    test_size=0.8689#0.2
    point_size = 5

    ## input files
    str_i_data='data_.txt'              # original nd data
    str_i_header='data__header.txt'     # header with #of observations and the #of dimensions
    #str_i_labels='data__labels.txt'     # labels of the data
    str_i_sdr='data__result_DR.txt'          # the sharpened dr results
    str_i_all_data='dataall_.txt'

    for dataset in datasets:
        dirs = sorted(glob(input_dir + '/{0}_*'.format(dataset)))

        for d in dirs:
            this_dir = os.path.join(d, '')

            for size in ['medium']:#, 'small', 'xsmall']:
                print(this_dir+ '/' + str_i_data)
                assert os.path.exists(this_dir+ '/' + str_i_data), 'main(): No input data file {0}.'.format(this_dir + '/' + str_i_data)
                data_orig=read_data(this_dir+'/' + str_i_data) #data_orig is original nD

                assert os.path.exists(this_dir+ '/' + str_i_all_data), 'main(): No input data file {0}.'.format(this_dir + '/' + str_i_data)
                all_data=read_data(this_dir+'/' + str_i_all_data) #data_orig is original nD

                # if os.path.exists(this_dir+'/' + str_i_header):
                #     n_obs, n_dim = read_header(this_dir+'/' + str_i_header)
                #     assert ((data_orig.shape[0]==n_obs) & (data_orig.shape[1]==n_dim)), 'main(): the number of observations and dimensions are different in data and header files.'
                # else:
                #     n_obs, n_dim = data_orig.shape[0], data_orig.shape[1]

                assert os.path.exists(this_dir+'/' + str_i_sdr), 'main(): No input data file.'
                data_sdr_2d=read_data(this_dir+'/' + str_i_sdr)   #data_sdr_2d --> 2D sdr
                
                scaler=MinMaxScaler()
                
                data_orig=scaler.fit_transform(data_orig)
                all_data=scaler.fit_transform(all_data)
                data_sdr_2d=scaler.fit_transform(data_sdr_2d)
                

                X_train_orig_sdr = data_orig
                X_test_orig_sdr = all_data
                X_train_sdr = data_sdr_2d
                #X_test_sdr

                print(this_dir, 'early stopping')

                if os.path.exists(os.path.join(this_dir, 'NNP_size_{0}_X_2d_train_000early.txt'.format(size))):
                    print(this_dir, 'skipping')
                else:
                    p = NNP(seed_data='precomputed', init=X_train_sdr, size=size, opt='adam', loss='mae', verbose=0, scale_data=False, dropout=True)
                    p.fit(X_train_orig_sdr)
                    X_2d_train = p.transform(X_train_orig_sdr)
                    X_2d_test  = p.transform(X_test_orig_sdr)

                    np.savetxt(os.path.join(this_dir, 'NNP_size_{0}_X_2d_train_000early.txt'.format(size)), X_2d_train, delimiter='\t')
                    np.savetxt(os.path.join(this_dir, 'NNP_size_{0}_X_2d_test_000early.txt'.format(size)), X_2d_test, delimiter='\t')

                for epochs in [300]: #300, 1000, 3000
                    print(this_dir, '{0} epochs'.format(epochs))

                    if os.path.exists(os.path.join(this_dir, 'NNP_size_{0}_X_2d_train_{1}_epochs.txt'.format(size, epochs))):
                        print(this_dir, 'skipping')
                    else:
                        p = NNP(seed_data='precomputed', init=X_train_sdr, size=size, opt='adam', loss='mae', epochs=epochs, early_stop=False, verbose=0, scale_data=False, dropout=False)
                        p.fit(X_train_orig_sdr)
                        X_2d_train = p.transform(X_train_orig_sdr)
                        X_2d_test  = p.transform(X_test_orig_sdr)
                        
                        np.savetxt(os.path.join(this_dir, 'NNP_size_{0}_X_2d_train_{1}_epochs.txt'.format(size, epochs)), X_2d_train, delimiter='\t')
                        np.savetxt(os.path.join(this_dir, 'NNP_size_{0}_X_2d_test_{1}_epochs.txt'.format(size, epochs)), X_2d_test, delimiter='\t')
