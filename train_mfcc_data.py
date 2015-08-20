from pybrain.datasets import SupervisedDataSet

from NN import NN
import numpy as np
import random


def make_data_set(file_path):
    text_file = open(file_path)
    lines = text_file.read().split('\n')
    text_file.close()

    random.shuffle(lines)

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
        labels.append(line[-1])

    data_set = SupervisedDataSet(13, 19)
    for data, label in zip(X, labels):
        label_data = np.zeros(19).astype(np.int32)
        label_data[int(label)] = 1
        data_set.addSample(data, label_data)

    return data_set


print 'read train dataset'
train_data_set = make_data_set('doc/train-mfcc.txt')

print 'read test dataset'
test_data_set = make_data_set('doc/test-mfcc.txt')

print 'start train'
result_txt = open('doc/result.txt', 'w')
for n_hidden in range(1, 14):
    print 'hidden num', n_hidden
    network = NN(13, n_hidden, 19)
    aveErr = network.train(train_data_set, test_data_set)
    result = 'n_hidden: %d, average error: %f' % (n_hidden, aveErr)

    print result
    result_txt.write(result + '\n')
    network.save('models/NN%d.xml' % n_hidden)
result_txt.close()
