import argparse

import datetime
import os
import pandas as pd
import h5py
import numpy as np
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from functools import partial, reduce
from collections import deque
from IPython.core.debugger import set_trace
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Activation, Add, Lambda
from tensorflow.keras.layers import Dense, MaxPooling1D, Conv1D, LSTM, GRU
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from functools import reduce
import editdistance
from itertools import groupby

from bc.preprocessing.processinput import DataCollection
import bc.utils.sequence as postprocessing

labelBaseMap = {
    0: "A",
    1: "C",
    2: "G",
    3: "T",
    4: "-"
}

class ChironCopy():
    def predict(self, model_path: str = "models/e00538_dis478.h5", filename: str = "~/basecalling-p10/data-preprocessing/test_dataset.hdf5"):

        data_collection = DataCollection(filename)
        self.try_set_gpu()
        
        model = self.load_model(data_collection.get_max_label_len())
        model.load_weights(model_path)

        for (d, _),_ in data_collection.generator():
            reference = postprocessing.numeric_to_bases_sequence(d['full_reference'], labelBaseMap)
            output = model.predict(d['the_input'])
            predictions = postprocessing.decode(output, labelBaseMap.values())
            
            assembled = postprocessing.assemble(predictions, 300, 5, labelBaseMap)
            
            error = postprocessing.calc_sequence_error_metrics(reference, assembled)
            print(assembled[:100])
            print(reference[:100])

    def train(self, model_path: str, data_path: str = "/mnt/sdb/taiyaki_mapped/mapped_umi16to9.hdf5"):
                 
        data_collection = DataCollection(filename)
        self.try_set_gpu()

        model = self.load_model(data_collection.get_max_label_len())
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
        
        save_cb = SaveCB("models", tf.keras.backend.function(model.inputs, [y_pred]), data_collection)

        for idx in range(len(data_collection)):
            print(f"Epoch {idx}/{len(data_collection)}")
            (a, _), _ = next(data_collection)
            model.fit(a[0], a[1], initial_epoch=idx, epochs=idx+1, callbacks=[save_cb]) 
                                    
    def try_set_gpu(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
          try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
          except RuntimeError as e:
            print(e)



    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, 5:, :]
        return kb.ctc_batch_cost(labels, y_pred, input_length, label_length)


    def make_res_block(self, upper, block):
        res = Conv1D(256, 1,
                      padding="same",
                      name=f"res{block}-r")(upper)
        upper = Conv1D(256, 1,
                      padding="same",
                      activation="relu",
                      use_bias="false",
                      name=f"res{block}-c1")(upper)
        upper = Conv1D(256, 3,
                      padding="same",
                      activation="relu",
                      use_bias="false",
                      name=f"res{block}-c2")(upper)
        upper = Conv1D(256, 1,
                      padding="same",
                      use_bias="false",
                      name=f"res{block}-c3")(upper)
        added = Add(name=f"res{block}-add")([res, upper])
        return Activation('relu', name=f"res{block}-relu")(added)

    def make_bdlstm(self, upper, block):
        lstm_1a = LSTM(200, return_sequences=True, name=f"blstm{block}-fwd")(upper)
        lstm_1b = LSTM(200, return_sequences=True, go_backwards=True, name=f"blstm{block}-rev")(upper)
        return Add(name=f"blstm{block}-add")([lstm_1a, lstm_1b])

    def load_model(self, max_label_length: int):
        input_data = Input(name="the_input", shape=(300,1), dtype="float32")
        inner = self.make_res_block(input_data, 1)
        inner = self.make_res_block(inner, 2)
        inner = self.make_res_block(inner, 3)
        inner = self.make_res_block(inner, 4)
        inner = self.make_res_block(inner, 5)
        inner = self.make_bdlstm(inner, 1)
        inner = self.make_bdlstm(inner, 2)
        inner = self.make_bdlstm(inner, 3)

        inner = Dense(64, name="dense", activation="relu")(inner)
        inner = Dense(5, name="dense_output")(inner)

        y_pred = Activation("softmax", name="softmax")(inner)

        labels = Input(name='the_labels', shape=(max_label_length), dtype='float32')
        input_length = Input(name='input_length', shape=(1), dtype='int64')
        label_length = Input(name='label_length', shape=(1), dtype='int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data], outputs=y_pred, name="chiron")

        return model

class SaveCB(Callback):
    def __init__(self, model_output_dir, test_func, prepper):
        self.model_output_dir=model_output_dir
        self.test_func = test_func
        self.prepper = prepper
        self.best_dist = None

    def calculate_loss(self, X, y, testbatchsize=1000):
        editdis = 0
        for b in range(0, len(X), testbatchsize):
            predicted = decode_batch(self.test_func, X[b:b+testbatchsize])
            mtest_y = ["".join(list(map(lambda x: labelBaseMap[x], ty))) for ty in y[b:b+testbatchsize]]
            for (p,l) in zip(predicted, mtest_y):
                editdis += editdistance.eval(p,l)
        return editdis/len(y)

    def on_epoch_end(self, epoch, logs={}):
        test_X, test_y = next(self.prepper.test_gen())
        train_X, train_y = self.prepper.last_train_gen_data[0]['the_input'], self.prepper.last_train_gen_data[0]['unpadded_labels']

        testloss = self.calculate_loss(train_X, train_y)
        print(f"\nAverage test edit distance is: {testloss}")
        valloss = self.calculate_loss(test_X, test_y)
        print(f"\nAverage validation edit distance is: {valloss}")

        if self.best_dist is None or valloss < self.best_dist:
            self.best_dist = valloss
            self.model.save_weights(os.path.join(self.model_output_dir, f'e{epoch:05d}_dis{round(valloss*100)}.h5'))


parser = argparse.ArgumentParser()
parser.add_argument("--p", help="predict", action="store_true")
parser.add_argument("--t", help="train", action="store_true")
parser.add_argument("-default_data", help="use data on server", action="store_true")
parser.add_argument("-mock_data", help="use mock data", action="store_true")
parser.add_argument("-existing_model", help="use exixting model", action="store_true")
parser.add_argument("model_path", help="path to model")
parser.add_argument("data_path", help="path to data file")

args = parser.parse_args()
if (args.p and args.t) or not (args.p or args.t):
    print("Must either train or predict")
    exit()
data =args.data_path # "/mnt/sdb/taiyaki_mapped/mapped_umi16to9.hdf5" if args.default_data else args.data_path
model = "chiron-copy.h5" if args.existing_model else args.model_path
if args.p:
    ChironCopy().predict(model, data)
elif args.t:
    ChironCopy().train(model, data)
    