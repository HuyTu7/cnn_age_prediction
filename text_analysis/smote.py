import math
import numpy as np
from random import randint, random, seed
from sklearn.neighbors import NearestNeighbors


class SMOTE():
    def execute(self, l, samples=[], labels=[]):
        np.random.seed(7)
        seed(7)
        return self.balance(samples, labels, r=int(l[0]), neighbors=int(l[1]))

    def smote(self, data, num, k=3, r=1):
        corpus = []
        if len(data) < k:
            k = len(data) - 1
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', p=r).fit(data)
        distances, indices = nbrs.kneighbors(data)
        for i in range(0, num):
            mid = randint(0, len(data) - 1)
            nn = indices[mid, randint(1, k - 1)]
            datamade = []
            for j in range(0, len(data[mid])):
                gap = random()
                datamade.append((data[nn, j] - data[mid, j]) * gap + data[mid, j])
            corpus.append(datamade)
        corpus = np.array(corpus)
        corpus = np.vstack((corpus, np.array(data)))
        return corpus

    def balance(self, data_train, train_label, r=0, neighbors=0):
        """

        :param data_train:
        :param train_label:
        :param m:
        :param r:
        :param neighbors:
        :return:
        """
        a_train = []
        b_train = []
        c_train = []
        d_train = []
        for j, i in enumerate(train_label):
            label = list(i).index(1)
            if label == 0:
                a_train.append(data_train[j])
            elif label == 1:
                b_train.append(data_train[j])
            elif label == 2:
                c_train.append(data_train[j])
            else:
                d_train.append(data_train[j])
        a_train = np.array(a_train)
        b_train = np.array(b_train)
        c_train = np.array(c_train)
        d_train = np.array(d_train)
        data = [a_train, b_train, c_train, d_train]
        mean_len = int(math.ceil((len(data_train)) / 4))
        print("Mean Length: ", mean_len)
        for i in range(len(data)):
            if len(data[i]) < mean_len:
                m = mean_len - len(data[i])
                data[i] = self.smote(data[i], m, k=neighbors, r=r)
            else:
                m = mean_len
                data[i] = data[i][np.random.choice(len(data[i]), m, replace=False)]
        data_train_smoted = np.vstack((data[0], data[1], data[2], data[3]))

        label_train_smoted = np.asarray([[1, 0, 0, 0]] * len(data[0]) + [[0, 1, 0, 0]] * len(data[1]) \
                                        + [[0, 0, 1, 0]] * len(data[2]) + [[0, 0, 0, 1]] * len(data[3]))
        shuffle_indices = np.random.permutation(np.arange(len(data_train_smoted)))

        data_train_smoted = data_train_smoted[shuffle_indices]
        label_train_smoted = label_train_smoted[shuffle_indices]
        return data_train_smoted, label_train_smoted