from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer
from pybrain.tools.xml import NetworkWriter

from progressbar import ProgressBar
import time


class NN:
    def __init__(self, n_input, n_hidden, n_output):
        self.network = buildNetwork(n_input, n_hidden, n_output)

    def train(self, train_data_set, test_data_set, epoch=100):
        trainer = BackpropTrainer(self.network, train_data_set)

        progress_bar = ProgressBar(epoch)

        for i in range(epoch):
            progress_bar.update(i+1)
            time.sleep(0.01)
            trainer.train()

        trainer.testOnData(test_data_set, verbose=True)

    def save(self, file_path):
        NetworkWriter.writeToFile(self.network, file_path)
