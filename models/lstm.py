import datetime
import numpy as np
import os
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers, initializers
from tensorflow.keras.models import Model
import sys
sys.path.append("..")
from data_loader import load_data


class LSTM(object):
    def __init__(self, search_dict):
        self.param_space = search_dict

        # load here so only does it once during the search
        data_dir = '../data/'
        self.xf_raw = np.load(data_dir + 'x_flt.npy', allow_pickle=True)
        self.xs_raw = np.load(data_dir + 'x_str.npy', allow_pickle=True)
        self.y_raw = np.load(data_dir + 'y.npy', allow_pickle=True)

        # 'None' so that history length can be a search variable
        self.xf_train, self.xf_test = None, None
        self.xs_train, self.xs_test = None, None
        self.y_train, self.y_test = None, None
        # Embedding input sizes
        self.vocab = None
        # Balance class sizes for training
        self.class_weights = None

    # runs thru the search space, and logs performance
    def search(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        print(f'\nGPU Available: {tf.test.is_gpu_available()}\n')

        for length in self.param_space['history length']:
            self.onboard_data(length)
            for hidd in self.param_space['hidden size']:
                for embed in self.param_space['embedding size']:
                    for batch in self.param_space['batch size']:
                        for drop in self.param_space['dropout']:
                            inputs, model, log_dir = self.train(hidd, embed, batch, drop)
                            train_acc, test_acc, cm, auc = self.evaluate_training(inputs, model)

                            with open(f'{log_dir}/results.txt', 'w') as f:
                                f.write(f'auc = {auc}\n cm = {cm}\n\n')
                                f.write(f'hist len = {length}\nhidd = {hidd}\nembed = {embed}\nbatch = {batch}\ndrop = {drop}')
                            print(f'\n\n\nhs={hidd}, es={embed}, bs={batch}, dr={drop}')
                            print(f'Train accuracy = {train_acc}')
                            print(f'Test accuracy = {test_acc}')
                            print(f'AUC score = {auc}')
                            print(f'Confusion matrix: \n{cm}\n')

    def onboard_data(self, length):
        use_latest = self.param_space['latest behavior']
        xf, xs, y, vocab_sizes = load_data(self.xf_raw, self.xs_raw, self.y_raw, length, use_latest)
        self.xf_train, self.xf_test = xf
        self.xs_train, self.xs_test = xs
        self.y_train, self.y_test = y
        self.vocab = vocab_sizes
        weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(self.y_train),
                                                    self.y_train)
        self.class_weights = {i: weights[i] for i in range(2)}

    def train(self, hidden_size, embed_size, batch_size, dropout):
        # returns a tensor's column
        def slice_feature(data, feat):
            n_samps, n_strs = data.shape[:2]
            feat_slice = tf.slice(data, [0, 0, feat], [n_samps, n_strs, 1])
            return tf.squeeze(feat_slice, axis=2)

        train_inputs, test_inputs = [self.xf_train], [self.xf_test]
        # includes string features in training
        if self.param_space['use non-numeric']:
            for i in range(self.xs_train.shape[-1]):
                train_inputs.append(slice_feature(self.xs_train, i))
                test_inputs.append(slice_feature(self.xs_test, i))
            model = self.get_fn_model(hidden_size, embed_size, dropout)
        else:
            model = self.get_seq_model(hidden_size, dropout)
        model.summary()

        lr = self.param_space['learning rate']
        adam = tf.keras.optimizers.Adam(lr=lr)
        model.compile(optimizer=adam,
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

        log_dir = "logs/lstm/" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.param_space['patience'])
        model.fit(train_inputs,
                  self.y_train,
                  epochs=self.param_space['epochs'],
                  batch_size=batch_size,
                  validation_split=self.param_space['validation frac'],
                  callbacks=[early_stop, tensorboard_callback],
                  class_weight=self.class_weights,
                  shuffle=True)
        return (train_inputs, test_inputs), model, log_dir

    # returns NN with embeddings for non-numeric data
    def get_fn_model(self, hidden_size, embed_size, dropout):
        h_len, n_str = self.xs_train.shape[1:]
        in_s = [layers.Input(shape=(h_len,), name=f's_in{i}') for i in range(n_str)]
        embeds = [layers.Embedding(self.vocab[i], embed_size, input_length=h_len, name=f's_emb{i}')(in_s[i]) for i in range(n_str)]
        inputs_f = layers.Input(shape=self.xf_train.shape[1:], name='flt_inputs')
        x = layers.concatenate([inputs_f]+embeds)

        x = layers.LSTM(hidden_size,
                        return_sequences=False,
                        kernel_regularizer=regularizers.L1L2(l1=0.01, l2=0.01),
                        kernel_initializer=self.param_space['weight initialization'])(x)
        x = layers.Dropout(dropout)(x)
        pred = layers.Dense(1, activation='sigmoid')(x)
        return Model(inputs=[inputs_f]+in_s, outputs=[pred])

    # returns NN without embeddings
    def get_seq_model(self, hidden_size, dropout):
        model = tf.keras.Sequential()
        model.add(layers.LSTM(hidden_size,
                              return_sequences=False,
                              kernel_initializer=self.param_space['weight initialization'],
                              input_shape=self.xf_train.shape[1:]))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def evaluate_training(self, inputs, model):
        train_inputs, test_inputs = inputs
        _, train_acc = model.evaluate(train_inputs, self.y_train, verbose=0)
        _, test_acc = model.evaluate(test_inputs, self.y_test, verbose=0)

        y_pred = model.predict(test_inputs)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        cm = confusion_matrix(self.y_test, y_pred)
        # area under the curve, since the categories are unbalanced
        auc = roc_auc_score(self.y_test, y_pred, average='weighted')
        return train_acc, test_acc, cm, auc


if __name__ == '__main__':
    search_params = {'hidden size': [30, 50],
                     'embedding size': [4, 7],
                     'batch size': [16, 32, 64],
                     'dropout': [0.2, 0.4],
                     # number of orders to consider for each customer
                     'history length': [30, 50, 75],
                     # True to use most recent orders. False to use customer's first orders
                     'latest behavior': True,
                     # True to train on both numeric and non-numeric features. False to train on numeric only
                     'use non-numeric': True,
                     'epochs': 50,
                     'patience': 3,
                     'learning rate': 0.001,
                     'validation frac': 0.15,
                     'weight initialization': 'glorot_uniform'}

    lstm = LSTM(search_params)
    lstm.search()
