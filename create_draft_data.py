
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

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_folder", action="store", dest="data_folder",
        required=True,
        help=("path to the data_folder ")
    )
    parser.add_argument(
        "--train_split", action="store", dest="train_split",
        default=0.8,type=float,
        help=("fraction of data to use for training ")
    )
    args = parser.parse_args()
    data_folder = args.data_folder
    train_split = args.train_split
    
    pool = multiprocessing.Pool()
        
    rating_path1 = pjoin(data_folder,"m19_rating.tsv")
    rating_path2 = pjoin(data_folder, "m19_land_rating.tsv")
    # drafts_path = pjoin(data_folder, "m19_drafts.csv")
    train_path = pjoin(data_folder, "train.csv")
    test_path = pjoin(data_folder, "test.csv")
    train_output = pjoin(data_folder,'train_drafts.pkl')
    test_output = pjoin(data_folder,'test_drafts.pkl')
    standardized_output = pjoin(data_folder, "standardized_m19_rating.tsv")
    cur_set = ds.create_set(rating_path1, rating_path2)

    
    #need to combine csvs in order to process data
    print('loading data')
    raw_drafts = ''
    for data_path in (train_path,test_path):
        cur_draft = ds.load_drafts(data_path)
        raw_drafts = raw_drafts + cur_draft
    
    cur_set, raw_drafts = ds.fix_commas(cur_set, raw_drafts)
    drafts = ds.process_drafts(raw_drafts)
    drafts = [d for d in drafts if len(d)==45] # Remove imcomplete drafts.

    cur_set.to_csv(standardized_output,
                   sep="\t", index=False)
    le = ds.create_le(cur_set["Name"].values)
    draft_to_tensor_func = partial(ds.draft_to_matrix,le=le)

    print('converting to tensor, may take awhile')
    draft_tensor = pool.map(draft_to_tensor_func,drafts)
    pick_tensor = np.int16(draft_tensor)
    
    num_drafts = np.shape(pick_tensor)[0]
    
    num_train = int(train_split*num_drafts)
    train_data = pick_tensor[:num_train]
    test_data = pick_tensor[num_train:]
    pickle.dump(train_data,open(train_output,'wb'))
    pickle.dump(test_data,open(test_output,'wb'))


    
