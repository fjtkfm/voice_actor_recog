from scikits.talkbox.features import mfcc
import scipy
from scipy.io import wavfile
import numpy as np

from progressbar import ProgressBar
import time
import sys


def convert_to_mfcc(voice_path):
    sample_rate, X = scipy.io.wavfile.read(voice_path)
    ceps, mspec, spec = mfcc(X)

    ave_cept = np.zeros((1, 13))
    count = 0
    for one_ceps in ceps:
        if np.isnan(one_ceps[1]):
            continue
        ave_cept += one_ceps
        count += 1
    if count == 0:
        return None
    ave_cept /= count

    return ave_cept


def read_text_file(txt_file_path):

    X, labels = [], []

    text_file = open(txt_file_path)
    lines = text_file.read().split('\n')
    text_file.close()

    progress_bar = ProgressBar(len(lines))

    for index, line in enumerate(lines):

        progress_bar.update(index+1)
        time.sleep(0.01)

        if not line:
            continue
        voice_path, label = line.split(' ')
        ceps = convert_to_mfcc(voice_path)

        if ceps is None:
            continue

        X.append(ceps)
        labels.append(label)

    return X, labels


def make_mfcc_data_file(input_file_path, output_file_path):

    X, labels = read_text_file(input_file_path)

    data_file = open(output_file_path, 'w')

    for x, label in zip(X, labels):
        for data in x[0]:
            data_file.write('%s ' % data)

        data_file.write(label)
        data_file.write('\n')

    data_file.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print '2 file paths required'
        sys.exit()
    make_mfcc_data_file(sys.argv[1], sys.argv[2])

