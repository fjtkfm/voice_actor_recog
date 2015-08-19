# voice-actor-recog
音声データからメル周波数ケプストラム係数(mfcc)を抽出し識別・認識を行う学習機

## Files
* make_data_ref_text.py: 指定したフォルダ内のデータと階層構造から、train.txt, test.txt, labels.txtを作成するスクリプト
* train_network.py: docフォルダ内のテキストファイルとvoicesの音声データから、mfccを抽出しニューラルネットに学習させるスクリプト
* NN.py: ニューラルネットモデル
* make_mfcc_data.py: 指定された音声データからmfccを抽出するスクリプト

## Requirements
* Numpy
* Scipy
* Scikits Talkbox
* PyBrain
* progressbar2
(requirements.txt参照)

## LICENSE
MIT License
