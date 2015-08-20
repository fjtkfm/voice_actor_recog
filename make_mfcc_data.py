from scikits.talkbox.features import mfcc
import scipy
from scipy.io import wavfile
import numpy as np


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
