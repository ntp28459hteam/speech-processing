import librosa
import soundfile as sf
import numpy as np

def trim_audio(y, threshold=0.00005):
    for i in range(len(y)):
        if(np.abs(y[i]) > threshold):
            index1 = i
            break
    for i in reversed(range(len(y))):
        if(np.abs(y[i]) > threshold):
            index2 = i
            break
    # print(index1, index2)
    y_trim = y[index1:index2]
    return y_trim

def preprocess():
    y, sr = librosa.load('data/demo/01.wav')
    #yt, index = librosa.effects.trim(y)
    y_trim = trim_audio(y)

    sf.write('data/demo/01_trim.wav', y_trim, sr)
    with open('debug.log', "w+") as debug_file:
        for i in y:
            debug_file.write(str(i) + " ")
preprocess()