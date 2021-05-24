import os
import sys
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(parent_dir_name)
sys.path.append(parent_dir_name)
import pandas as pd
import pickle
import utils
import argparse
import keras
import sys
from distutils.util import strtobool

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_hdf5", action="store", dest="model_hdf5",
        required=True,
        help=("path to the model_hdf5 ")
    )
    parser.add_argument(
        "--test_pkl", action="store", dest="test_data_pkl",
        required=True,
        help=("path to the test_data_pkl ")
    )
    parser.add_argument(
        "--card_name_csv", action="store", dest="card_name_csv",
        required=True,
        help=("path to the card_name_csv")
    )
    parser.add_argument(
        "--shuffle", action="store", dest="shuffle",
        default=False,type=lambda x: bool(strtobool(x)),
        help=("path to the card_name_csv")
    )

    args = parser.parse_args()
    model = keras.models.load_model(args.model_hdf5)

    test_data = pickle.load(open(args.test_pkl,'rb'))
    test_processor = utils.get_processor(test_data)
    iterator = test_processor.get_iter(shuffle=args.shuffle)

    
    
