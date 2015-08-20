import sys
import subprocess
import time

from progressbar import ProgressBar

import make_mfcc_data


def cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    return stdout.rstrip()


dirs = cmd("ls " + sys.argv[1])
labels = dirs.splitlines()

if 'doc' not in labels:
    cmd("mkdir doc")

recog = open('doc/recog.txt', 'w')

progress_bar = ProgressBar(len(labels))
for class_no, label in enumerate(labels):

    progress_bar.update(class_no+1)
    time.sleep(0.01)

    work_dir = sys.argv[1] + '/' + label
    voice_files = cmd('ls ' + work_dir + '/*.wav')
    voices = voice_files.splitlines()

    for index, voice in enumerate(voices):
        ceps = make_mfcc_data.convert_to_mfcc(voice)

        for data in ceps:
            recog.write('%s ' % data)
        recog.write('\n')

recog.close()

