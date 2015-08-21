import subprocess
import time

from progressbar import ProgressBar

import make_mfcc_data


def cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    return stdout.rstrip()


dirs = cmd("ls voice")
labels = dirs.splitlines()

if 'doc' not in labels:
    cmd("mkdir doc")

train = open('doc/train.txt', 'w')
test = open('doc/test.txt', 'w')
labels_txt = open('doc/labels.txt', 'w')

progress_bar = ProgressBar(len(labels))
count = 0
for class_no, label in enumerate(labels):

    progress_bar.update(class_no+1)
    time.sleep(0.01)

    work_dir = 'voice/' + label
    voice_files = cmd('ls ' + work_dir + '/*.wav')
    voices = voice_files.splitlines()

    labels_txt.write(label + '\n')
    start_count = count
    length = len(voices)
    for voice in voices:
        ceps = make_mfcc_data.convert_to_mfcc(voice)

        if count - start_count < length * 0.75:
            for data in ceps:
                train.write('%s ' % data)
            train.write('%s' % class_no)
            train.write('\n')
        else:
            for data in ceps:
                test.write('%s ' % data)
            test.write('%s' % class_no)
            test.write('\n')
        count += 1

train.close()
test.close()
labels_txt.close()
