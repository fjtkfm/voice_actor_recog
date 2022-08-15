from pathlib import Path
import csv

# from progressbar import ProgressBar

import make_mfcc_data


file_pathes = Path('./voice/remove_silence/').glob('**/*.wav')
data = []
labels = []

for file_path in file_pathes:
    mfcc = make_mfcc_data.convert_to_mfcc(str(file_path))
    for m in mfcc:
        data.append(m)
        labels.append(file_path.stem)

with open('data.txt', 'w') as data_f:
    writer = csv.writer(data_f)
    for d in data:
        writer.writerow(d)

with open('labels.txt', 'w') as label_f:
    writer = csv.writer(label_f)
    for l in labels:
        writer.writerow([l])
