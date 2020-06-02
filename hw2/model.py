import librosa
import numpy as np
import os
import argparse
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm
import joblib
import random
from shutil import copyfile

trained = False
models = {}

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

def get_class_data(data_dir):
    files = os.listdir(data_dir)
    mfcc = [get_mfcc(os.path.join(data_dir,f)) for f in files if f.endswith(".wav") and not f.endswith("_bad.wav")]
    return mfcc

def clustering(X, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0, verbose=0)
    kmeans.fit(X)
    # print("centers", kmeans.cluster_centers_.shape)
    return kmeans

def train(evaluate=False):
    global trained
    trained = True

    class1 = ["toi", "test_toi"]
    class2 = ["mot", "test_mot"]
    class3 = ["trong", "test_trong"]
    class4 = ["thoigian", "test_thoigian"]
    class5 = ["chungta", "test_chungta"]
    classdemo = ["demo"]

    class_names = []
    class_names.extend(class1)
    class_names.extend(class2)
    class_names.extend(class3)
    class_names.extend(class4)
    class_names.extend(class5)
    class_names.extend(classdemo)

    dataset = {}

    if evaluate is True:
        for cname in class_names:
            if cname[:4] == "test":
                cname_ = cname.split("_")[1]
                data_dir_src = os.path.join("data", cname_)
                data_dir_dst = os.path.join("data", cname)
                samples = random.sample([x for x in os.listdir(data_dir_src) if os.path.isfile(os.path.join(data_dir_src, x))], 10)
                [copyfile(os.path.join(data_dir_src, x), os.path.join(data_dir_dst, x)) for x in samples]
    for cname in class_names:
        print(f"Load {cname} dataset")
        dataset[cname] = get_class_data(os.path.join("data", cname))
    all_vectors = np.concatenate([np.concatenate(v, axis=0) for k, v in dataset.items()], axis=0)
    kmeans = clustering(all_vectors)
    kmeans_model_filename = 'kmeans.joblib'
    with open (kmeans_model_filename, 'wb') as f_kmeans:
        joblib.dump(kmeans, f_kmeans)
    print("centers", kmeans.cluster_centers_.shape)


    config = {
        'toi': {'n_components':5},
        'mot': {'n_components': 5},
        'trong': {'n_components': 5},
        'thoigian': {'n_components': 8}, # 10
        'chungta': {'n_components': 8},
        'test_toi': {'n_components': 5},
        'test_mot': {'n_components': 5},
        'test_trong': {'n_components': 5},
        'test_thoigian': {'n_components': 8}, # 10
        'test_chungta': {'n_components': 8},
        'demo': {'n_components': 3},
    }
    for cname in class_names:
        class_vectors = dataset[cname]
        dataset[cname] = list([kmeans.predict(v).reshape(-1,1) for v in dataset[cname]])
        # [print(np.shape(dataset[cname][i])) for i in range(10) if cname != 'demo']
        n_components = config[cname]['n_components']
        start_prob = np.zeros(n_components)
        start_prob[0] = 1.0
        transmat = np.ndarray(shape=(n_components, n_components), dtype=float)
        for i in range((n_components - 1)):
            transmat[i][i] = 0.7
            transmat[i][i + 1] = 0.3
        transmat[n_components - 1][n_components - 1] = 1.0

        hmm = hmmlearn.hmm.MultinomialHMM(
            n_components=n_components, random_state=0, n_iter=1000, verbose=True,
        )
        hmm.startprob_ = start_prob
        hmm.transmat_prior = transmat

        if cname[:4] != 'test':
            X = np.concatenate(dataset[cname])
            lengths = list([len(x) for x in dataset[cname]])
            print("training class", cname)
            hmm.fit(X, lengths=lengths)
            models[cname] = hmm
    print("Training done")
    model_filename = 'finalized_model.joblib'
    with open (model_filename, 'wb') as f_hmm:
        joblib.dump(models, f_hmm)

    if evaluate is True:
        print("Testing")
        for true_cname in class_names:           
            if true_cname != 'demo':
                count = 0
                for O in dataset[true_cname]:
                    score = {cname : model.score(O, [len(O)]) for cname, model in models.items() if cname[:4] != 'test' and cname != 'demo' }
                    predict = max(score, key = score.get)
                    if predict == true_cname or true_cname[:4] == 'test' and predict == true_cname.split('_')[1]:
                        count+=1
            #         print(true_cname, score, predict)
                print(f"true: {count}/{len(dataset[true_cname])}")

        for cname in class_names:
            if cname[:4] == "test":
                data_dir_dst = os.path.join("data", cname)
                samples = os.listdir(data_dir_dst)
                [os.remove(os.path.join(data_dir_dst, x)) for x in samples]
    return models

def predict(load_from_disk=True):
    global models

    if load_from_disk is False:
        if trained is False:
            models = train()
    else:
        model_filename = 'finalized_model.joblib'
        with open(model_filename, 'rb') as f_hmm:
            models = joblib.load(f_hmm)
        
        kmeans_model_filename = 'kmeans.joblib'
        with open (kmeans_model_filename, 'rb') as f_kmeans:
            kmeans = joblib.load(f_kmeans)


    class1 = ["toi", "test_toi"]
    class2 = ["mot", "test_mot"]
    class3 = ["trong", "test_trong"]
    class4 = ["thoigian", "test_thoigian"]
    class5 = ["chungta", "test_chungta"]
    classdemo = ["demo"]

    class_names = []
    class_names.extend(classdemo)
    class_names.extend(class1)
    class_names.extend(class2)
    class_names.extend(class3)
    class_names.extend(class4)
    class_names.extend(class5)

    dataset = {}

    print("Predicting")
    for true_cname in class_names:
        if true_cname == 'demo':
            dataset[true_cname] = get_class_data(os.path.join("data", true_cname))
            all_vectors = np.concatenate([np.concatenate(v, axis=0) for k, v in dataset.items()], axis=0)
            kmeans = clustering(all_vectors)
            dataset[true_cname] = list([kmeans.predict(v).reshape(-1,1) for v in dataset[true_cname]])
            # print(np.shape(dataset[true_cname][0][:10]))
            for O in dataset[true_cname]:
                # score = {cname : model.score(O, [len(O)]) for cname, model in models.items() if cname[:4] != 'test' and cname != 'demo' }
                score = {}
                for cname in class_names:
                    if cname[:4] == 'test' or cname == 'demo':
                        continue
                    model = models[cname]
                    if cname[:4] != 'test' and cname != 'demo':
                        score[cname] = model.score(O, [len(O)])
                predict = max(score, key = score.get)
            return predict