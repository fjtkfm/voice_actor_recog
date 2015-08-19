from pybrain.datasets import SupervisedDataSet

from NN import NN
import make_mfcc_data


def make_data_set(file_path):
    X, labels = make_mfcc_data.read_text_file(file_path)

    data_set = SupervisedDataSet(13, 3)
    for data, label in zip(X, labels):
        if label == '0':
            data_set.addSample(data, [1, 0, 0])
        elif label == '1':
            data_set.addSample(data, [0, 1, 0])
        else:
            data_set.addSample(data, [0, 0, 1])

    return data_set


print 'make train dataset'
train_data_set = make_data_set('doc/train.txt')

print 'make test dataset'
test_data_set = make_data_set('doc/test.txt')

print 'start train'
network = NN(13, 7, 3)
network.train(train_data_set, test_data_set)
print 'save NN model to NN.xml'
network.save('NN.xml')
