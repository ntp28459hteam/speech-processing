import librosa
import math
from sklearn.cluster import KMeans
import os
import joblib
import numpy as np

class_names = ["toi", "mot", "trong", "thoigian", "chungta"]

def get_mfcc(file_path):
    y, sr = librosa.load(file_path) # read .wav file
    hop_length = math.floor(sr*0.010) # 10ms hop
    win_length = math.floor(sr*0.025) # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    # substract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 36 (transpose of X)
    return X.T # hmmlearn use T x N matrix

def clustering(X, n_clusters=8):
    kmeans_model_filename = 'kmeans.joblib'
    with open (kmeans_model_filename, 'rb') as f_kmeans:
        kmeans = joblib.load(f_kmeans)
    return kmeans

def toObservation(file_path):
    mfcc = get_mfcc(file_path)
    # all_vectors = np.concatenate([np.concatenate(v, axis=0) for k, v in dataset.items()], axis=0)
    all_vectors = [mfcc]
    kmeans = clustering(all_vectors)
    print(np.shape(all_vectors))
    sequence = list([kmeans.predict(v).reshape(-1,1) for v in all_vectors])
    return sequence

def load_models():
    global models
    model_filename = 'finalized_model.joblib'
    with open(model_filename, 'rb') as f_hmm:
        models = joblib.load(f_hmm)
    return models

def predict(sequence, classes=class_names):
    '''classes: specify which class will be a candidate result'''
    models = load_models()
    scores = {}
    for cname in classes:
        print(models[cname])
        scores[cname] = models[cname].score(sequence, [len(sequence)]) 
    # predict = max(score, key = score.get)
    return scores

def print2DArr(arr):
    for row in arr:
        for i in row:
            print(i[0], end=' ')
def print2DArrToFile(arr, dst_file):
    with open(dst_file, 'w+') as f:
        for row in arr:
            for i in row:
                f.write(str(i[0]))
                f.write(' ')
cut_trong = toObservation('data/demo_cut/01_cut_.wav')
train_trong = toObservation('data/trong/untitled-15.wav')

print2DArrToFile(cut_trong, 'cut.log')
print2DArrToFile(train_trong, 'train_trong.log'])

print(predict(cut_trong[0], classes=["trong"]))
print(predict(train_trong[0], classes=["trong"]))
