import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import os

def read_data(this_path):
    f1 = open(this_path)
    data=np.genfromtxt(f1.readlines(),dtype='float32')
    return data

if __name__ == "__main__":
    ## PARAMETERS
    random_state=420
    test_size=0.2
    point_size = 5

    this_dir='/home/yk/Documents/sdr-nnp-linux/Code/python-nnp/datasets/GALAHDR2_LMDS_10_0.17/'
    str_i_data_original='data_.txt'
 
    assert os.path.exists(this_dir + str_i_data_original), 'main(): No input data file.'
    data_orig=read_data(this_dir + str_i_data_original)   

    arr=list(range(1, 10001))
    print(arr)
    X_train_orig, X_test_orig , Y_train_orig, Y_test_orig = train_test_split(arr,arr, test_size=test_size, random_state=random_state)

    np.savetxt(os.path.join(this_dir, 'idx_train.txt'), X_train_orig, delimiter='\t')
    np.savetxt(os.path.join(this_dir, 'idx_test.txt'), X_test_orig, delimiter='\t')