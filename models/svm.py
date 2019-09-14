import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import class_weight
import sys
sys.path.append("..")
from data_loader import load_data


class SVM(object):
    def __init__(self, search_dict):
        self.search_dict = search_dict

        # load here so only does it once
        data_dir = '../data/'
        self.x = np.load(data_dir + 'x_flt.npy', allow_pickle=True)
        self.y = np.load(data_dir + 'y.npy', allow_pickle=True)

        self.x_train, self.x_test = None, None
        self.y_train, self.y_test = None, None
        self.class_weights = None

    def search(self):
        for C in self.search_dict['Cs']:
            for gamma in self.search_dict['gammas']:
                print(f'C = {C}, gamma = {gamma}')
                for e in range(self.search_dict['trials']):
                    print(f'Trial {e+1}')
                    print(f'{self.run(C, gamma)}\n')
                print('\n')

    def run(self, C, gamma):
        print('Onboarding data')
        weights = self.onboard_data()
        svclassifier = SVC(kernel=search_dict['kernel'],
                           C=C,
                           gamma=gamma,
                           class_weight=weights,
                           cache_size=1000)
        print('Training')
        svclassifier.fit(self.x_train, self.y_train)
        y_pred = svclassifier.predict(self.x_test)
        return confusion_matrix(self.y_test, y_pred)

    def onboard_data(self):
        hl = search_dict['history length']
        lb = search_dict['latest behavior']
        x_pair, y_pair = load_data(self.x, self.x, self.y, hl, from_end=lb, tensor=False)
        self.x_train, self.x_test = x_pair
        self.y_train, self.y_test = y_pair

        weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(self.y_train),
                                                    self.y_train)
        return {i: weights[i] for i in range(2)}


if __name__ == '__main__':
    search_dict = {'Cs': [1, 10],
                   'gammas': [0.01, 0.001],
                   'kernel': 'rbf',
                   # number of orders to consider for each customer
                   'history length': 50,
                   'trials': 3,
                   # True to use most recent orders. False to use customer's first orders
                   'latest behavior': True}

    svm = SVM(search_dict)
    svm.search()
