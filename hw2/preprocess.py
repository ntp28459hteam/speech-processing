import librosa

def preprocess():
    y, sr = librosa.load('demo/01.wav')
    print(y, sr)