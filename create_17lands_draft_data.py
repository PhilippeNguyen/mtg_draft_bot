# Preprocessing imports
import numpy as np
import os
import pandas as pd
import pickle
from sklearn import preprocessing
from os.path import join as pjoin
from functools import partial
import multiprocessing
import draftsimtools as ds
from providers import seventeen_lands
import argparse

import dill as pickle
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_csv", action="store", dest="data_csv",
        required=True,
        help=("path to the data_folder (as downloaded from 17lands)")
    )
    parser.add_argument(
        "--train_split", action="store", dest="train_split",
        default=0.8,type=float,
        help=("fraction of data to use for training ")
    )
    args = parser.parse_args()

    # card_name_df = pd.read_csv(args.card_name_csv)

    print('parsing data...')
    data,target,card_name_df = seventeen_lands.parse_data_csv(args.data_csv)
    
    num_samples = data.shape[0]
    train_samples = int(args.train_split*num_samples)
    train_data,train_target = data[:train_samples],target[:train_samples]
    test_data,test_target = data[train_samples:],target[train_samples:]

    print('saving pkls...')
    output_prefix = os.path.splitext(args.data_csv)[0]
    pickle.dump({'x':train_data,'y':train_target,'data_format':'sparse'},
                open(output_prefix+'_train_data.pkl','wb'))
    pickle.dump({'x':test_data,'y':test_target,'data_format':'sparse'},
                open(output_prefix+'_test_data.pkl','wb'))
    card_name_df.to_csv(output_prefix+'_card_names.csv',index=False)
