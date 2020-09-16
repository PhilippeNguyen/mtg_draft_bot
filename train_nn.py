
import pandas as pd
import pickle
from os.path import join as pjoin
import draftsimtools as ds
import argparse
import utils
import tensorflow as tf
import nn_utils
import numpy as np
import keras


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--train_pkl", action="store", dest="train_pkl",
        required=True,
        help=("path to the train_pkl ")
    )
    parser.add_argument(
        "--test_pkl", action="store", dest="test_pkl",
        required=True,
        help=("path to the test_pkl ")
    )
    parser.add_argument(
        "--standard_rating_tsv", action="store", dest="standard_rating_tsv",
        required=True,
        help=("path to the standard_rating_tsv (standardized_m19_rating.tsv) ")
    )
    parser.add_argument(
        "--output_name", action="store", dest="output_name",
        required=True,
        help=("name of the output hdf5 file ")
    )
    parser.add_argument(
        "--batch_size", action="store", dest="batch_size",
        default=32,type=int,
        help=("batch_size ")
    )
    args = parser.parse_args()
    batch_size = args.batch_size
    train_pkl = args.train_pkl
    test_pkl = args.test_pkl
    output_name = args.output_name

    standardized_output = args.standard_rating_tsv
    m19_set = pd.read_csv(standardized_output, delimiter="\t")
    m19_set["Color Vector"] = [eval(s) for s in m19_set["Color Vector"]]
    le = ds.create_le(m19_set["Name"].values)
    
    
    train_data = pickle.load(open(train_pkl,'rb'))
    test_data = pickle.load(open(test_pkl,'rb'))

    num_train = train_data.shape[0]
    num_test = test_data.shape[0]
    train_steps = num_train // batch_size
    test_steps = num_test // batch_size

    train_dataset = tf.data.Dataset.from_generator(utils.DataProcessor(train_data, le).get_iter,
                                                     output_types=(tf.int16,tf.int16))
    test_dataset = tf.data.Dataset.from_generator(utils.DataProcessor(test_data, le).get_iter,
                                                    output_types=(tf.int16, tf.int16))

    st = utils.create_set_tensor(m19_set)
    set_size,feature_size = st.shape
    input_size = set_size*2
    model = nn_utils.build_model(input_size)

    loss = tf.keras.losses.CategoricalCrossentropy(
                                            from_logits=True)

    metric = tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None)
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=[metric]
                  )

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint(filepath=output_name,
                                           save_best_only=True),
    ]
    model.fit(train_dataset.batch(batch_size),epochs=50,
              steps_per_epoch=train_steps,
              validation_data=test_dataset.batch(batch_size),
              validation_steps=test_steps,
              callbacks=my_callbacks)
