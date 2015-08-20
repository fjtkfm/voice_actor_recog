import sys
import subprocess


def cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    return stdout.rstrip()


dirs = cmd("ls " + sys.argv[1])
labels = dirs.splitlines()

if 'doc' not in labels:
    cmd("mkdir doc")
if 'voices' not in labels:
    cmd("mkdir voices")

train = open('doc/train.txt', 'w')
test = open('doc/test.txt', 'w')
labels_txt = open('doc/labels.txt', 'w')

count = 0
for class_no, label in enumerate(labels):
    work_dir = sys.argv[1] + '/' + label
    voice_files = cmd('ls ' + work_dir + '/*.wav')
    voices = voice_files.splitlines()

    labels_txt.write(label + '\n')
    start_count = count
    length = len(voices)
    for voice in voices:
        voice_path = 'voices/%06d' % count + '.wav'
        cmd("cp " + voice + ' ' + voice_path)
        if count - start_count < length * 0.75:
            train.write(voice_path + ' %d\n' % class_no)
        else:
            test.write(voice_path + ' %d\n' % class_no)
        print voice_path
        count += 1

train.close()
test.close()
labels_txt.close()

