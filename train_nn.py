
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
        "--output_name", action="store", dest="output_name",
        required=True,
        help=("name of the output hdf5 file ")
    )
    parser.add_argument(
        "--batch_size", action="store", dest="batch_size",
        default=100,type=int,
        help=("batch_size ")
    )
    parser.add_argument(
        "--train_fraction", action="store", dest="train_fraction",
        default=0.5,type=float,
        help=("fraction of the training set to go through per epoch")
    )
    args = parser.parse_args()
    batch_size = args.batch_size
    train_pkl = args.train_pkl
    test_pkl = args.test_pkl
    output_name = args.output_name
    train_fraction = args.train_fraction
    if train_fraction >1:
        raise ValueError("train_fraction must be less than 1")
    
    train_data = pickle.load(open(train_pkl,'rb'))
    test_data = pickle.load(open(test_pkl,'rb'))


    train_processor = utils.get_processor(train_data)
    test_processor = utils.get_processor(test_data)

    if train_processor.get_set_size() != test_processor.get_set_size():
        raise Exception("""Computed number of cards in the set is different for train and test data. 
                            Num cards in set (train): {}, 
                            Num cards in set (test): {}, 
                            """.format(train_processor.get_set_size(),
                                        test_processor.get_set_size()))
    
    num_train = len(train_processor)
    num_test = len(test_processor)
    train_steps = (num_train*train_fraction) // batch_size
    test_steps = num_test // batch_size
    train_dataset = tf.data.Dataset.from_generator(train_processor.get_iter,
                                                     output_types=(tf.int16,tf.int16))
    test_dataset = tf.data.Dataset.from_generator(test_processor.get_iter,
                                                    output_types=(tf.int16, tf.int16))


    input_size = train_processor.get_set_size()*2
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
