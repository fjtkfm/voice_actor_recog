from pybrain.datasets import UnsupervisedDataSet
import numpy as np
from pybrain.tools.xml import NetworkReader


print 'read dataset'

text_file = open('doc/recog.txt')
lines = text_file.read().split('\n')
text_file.close()

text_file = open('doc/labels.txt')
labels = text_file.read().split('\n')
text_file.close()

network = NetworkReader.readFrom('NN.xml')

for line in lines:
    if not line:
        continue
    line = line.split(' ')
    datas = line[:-1]
    x = []
    for data in datas:
        x.append(float(data))
    label = line[-1]

    data_set = UnsupervisedDataSet(13)
    label_data = np.zeros(19).astype(np.int32)
    label_data[int(label)] = 1
    data_set.addSample(x)

    out = network.activateOnDataset(data_set)
    print labels[np.argmax(out)]
