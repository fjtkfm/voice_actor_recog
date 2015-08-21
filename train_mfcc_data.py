import random

from pybrain.datasets import SupervisedDataSet
import numpy as np

from NN import NN


def make_data_set(file_path):
    text_file = open(file_path)
    lines = text_file.read().split('\n')
    text_file.close()

    random.shuffle(lines)

    max_label = 0
    X, labels = [], []
    for line in lines:
        if not line:
            continue
        line = line.split(' ')
        datas = line[:-1]
        x = []
        for data in datas:
            x.append(float(data))
        X.append(x)

        label = int(line[-1])
        labels.append(label)
        if max_label < label:
            max_label = label

    data_set = SupervisedDataSet(13, max_label + 1)
    for data, label in zip(X, labels):
        label_data = np.zeros(max_label + 1).astype(np.int32)
        label_data[int(label)] = 1
        data_set.addSample(data, label_data)

    return data_set


print 'read train dataset'
train_data_set = make_data_set('doc/train.txt')

print 'read test dataset'
test_data_set = make_data_set('doc/test.txt')

print 'start train'
network = NN(13, 10, train_data_set.outdim)
network.train(train_data_set, test_data_set)
network.save('NN.xml')
