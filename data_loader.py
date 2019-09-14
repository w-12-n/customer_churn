import random
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf


# Normalizes the feature matrix, and returns a random train test split of the data.
# xf = numeric feature matrix
# xs = non-numeric feature matrix
# y = labels
# hist_len = # orders to consider for each customer
# from_end = True to use most recent orders. False to use customer's first orders
# train_frac = fraction of dataset to use for training
# tensor = True to return a tf tensor. False to return numpy array
def load_data(xf, xs, y, hist_len, from_end, train_frac=0.8, tensor=True):
    xy_sized = enforce_history(xf, xs, y, hist_len, from_end)
    xf_pair, xs_pair, y_pair = random_split(xy_sized, train_frac)
    xf_pair_2d = standardize_cols(xf_pair)

    if tensor:
        print_stats(xy_sized)
        xf_pair = get_tensor(xf_pair_2d, xf_pair)
        xs_pair, vocab_sizes = map_to_integer(xs_pair)
        return xf_pair, xs_pair, y_pair, vocab_sizes

    xf_pair = get_array(xf_pair_2d, xf_pair)
    return xf_pair, y_pair


# returns 3D numpy array of h_len orders from each customer
def enforce_history(xf, xs, y, h_len, from_end):
    new_xf, new_xs, new_y = [], [], []
    for i, cust_hist in enumerate(xf):
        if len(cust_hist) >= h_len:
            if from_end:
                new_xf.append(cust_hist[-h_len:])
                new_xs.append(xs[i][-h_len:])
            else:
                new_xf.append(cust_hist[:h_len])
                new_xs.append(xs[i][:h_len])
            new_y.append(y[i])
    return np.array(new_xf), np.array(new_xs), np.array(new_y)


def random_split(xs_y, train_frac):
    def split(arr, pivot):
        return arr[:pivot], arr[pivot:]

    xf, xs, y = xs_y

    n_samples = len(y)
    random_order = np.array(range(n_samples))
    random.shuffle(random_order)
    xf_shuff = xf[random_order]
    xs_shuff = xs[random_order]
    y_shuff = y[random_order]

    i = int(n_samples * train_frac)
    xf_data = split(xf_shuff, i)
    xs_data = split(xs_shuff, i)
    y_data = split(y_shuff, i)
    return xf_data, xs_data, y_data


def standardize_cols(data):
    train, test = data
    train2d = make_2d(train)
    test2d = make_2d(test)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train2d)
    test_scaled = scaler.transform(test2d)
    return train_scaled, test_scaled


def make_2d(arr):
    return arr.reshape((-1, arr.shape[-1]))


def to_tensor(numpy_2d, shape):
    numpy_3d = numpy_2d.reshape(shape)
    return tf.convert_to_tensor(numpy_3d, dtype='float32')


def get_tensor(x_pair_numpy, x_pair_tensor):
    x_train_n, x_test_n = x_pair_numpy
    x_train, x_test = x_pair_tensor

    x_pair = to_tensor(x_train_n, x_train.shape), to_tensor(x_test_n, x_test.shape)
    return x_pair


# returns tensor where non-numeric terms are mapped to integers
def map_to_integer(data):
    train, test = data
    train_2d = make_2d(train)
    test_2d = make_2d(test)
    all_feats = np.vstack((train_2d, test_2d))

    vocab_sizes = dict()
    new_train, new_test = [], []
    for feat in range(train_2d.shape[-1]):
        le = LabelEncoder()
        le.fit(all_feats[:, feat])
        int_train = le.transform(train_2d[:, feat])
        int_test = le.transform(test_2d[:, feat])

        new_train.append(int_train)
        new_test.append(int_test)
        vocab_sizes[feat] = len(set(all_feats[:, feat]))

    new_data = np.array(new_train).T, np.array(new_test).T
    return get_tensor(new_data, data), vocab_sizes


def get_array(xf_pair_2d, xf_pair):
    xf_2d_train, xf_2d_test = xf_pair_2d
    xf_train, xf_test = xf_pair

    new_train = xf_2d_train.reshape(xf_train.shape[0], -1)
    new_test = xf_2d_test.reshape(xf_test.shape[0], -1)
    return new_train, new_test


def print_stats(xy):
    xf, xs, y = xy
    print(f'Fraction customers churned = {sum(y) / len(y)}')
    print(f'Number samples = {len(y)}')
    print(f'Number features = {xf[0].shape[-1] + xs[0].shape[-1]}')
