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

recog = open('doc/recog.txt', 'w')

count = 0
for class_no, label in enumerate(labels):
    work_dir = sys.argv[1] + '/' + label
    voice_files = cmd('ls ' + work_dir + '/*.wav')
    voices = voice_files.splitlines()

    start_count = count
    length = len(voices)
    for voice in voices:
        voice_path = 'voices/recog%06d' % count + '.wav'
        cmd("cp " + voice + ' ' + voice_path)
        recog.write(voice_path + '\n')
        print voice_path
        count += 1

recog.close()

