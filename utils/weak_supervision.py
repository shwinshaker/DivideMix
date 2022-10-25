#!./env python
import pickle
import json
import os
import numpy as np
import torch
from ..utils import print

__all__ = ['get_weak_supervision']

# def get_noise_supervision():
# TODO

def print_pseudo_label_info(tar, train_size):
    print('[WSL] ----------------- Pseudo-labeling Info -----------------')
    print('[WSL] Pseudo-labeled subset size: %i' % len(tar['index']))
    print('[WSL] Class count (True label): ', [(i, c) for i, c in zip(*np.unique(tar['true_label'], return_counts=True))])
    print('[WSL] Class count (Pseudo label): ', [(i, c) for i, c in zip(*np.unique(tar['pseudo_label'], return_counts=True))])
    noise_ratio = 1 - (tar['pseudo_label'] == tar['true_label']).sum() / len(tar['true_label'])
    print('[WSL] Noise ratio: %.2f%%' % (noise_ratio * 100))
    coverage = len(tar['index']) / train_size
    print('[WSL] Coverage: %.2f%%' % (coverage * 100))
    print('[WSL] --------------------------------------------------------')

def get_weak_supervision(trainids, data_dir, dataset, seed_file_name="seedwords.json"):
    data_path = os.path.join(data_dir, dataset)
    pseudo_label_path = os.path.join(data_path, "pseudo_weak.pt")
    if os.path.exists(pseudo_label_path):
        tar = torch.load(pseudo_label_path)
        print_pseudo_label_info(tar, train_size=len(trainids))
        return tar
    
    print('=====> Load raw text..')
    df = pickle.load(open(os.path.join(data_path, "df.pkl"), "rb"))
    df = df.iloc[trainids]
    df = df.reset_index(drop=True)
    
    print('=====> Preprocess text..')
    stored_file = os.path.join(data_path, "df_preprocessed.pkl")
    if os.path.exists(stored_file):
        df_preprocessed = pickle.load(open(stored_file, "rb"))
    else:
        df_preprocessed = preprocess(df)
        pickle.dump(df_preprocessed, open(stored_file, 'wb'))
    
    print('=====> Load seed words..')
    labels = df["label"].unique().tolist()
    with open(os.path.join(data_path, seed_file_name)) as fp:
        label_term_dict = json.load(fp)
    
    print('=====> Init tokensizer..')
    tokenizer = fit_get_tokenizer(df_preprocessed['text'], max_words=150000)
    
    print('=====> Get pseudo-labels..')
    index, pseudo_label, true_label = generate_pseudo_labels(df_preprocessed, labels, label_term_dict, tokenizer)
    
    print('=====> Save pseudo-labels..')
    label_to_index = {label: i for i, label in enumerate(labels)}
    tar = {'index': np.array(index),
           'pseudo_label': np.array([label_to_index[l] for l in pseudo_label]),
           'true_label': np.array([label_to_index[l] for l in true_label]),
          }
    torch.save(tar, pseudo_label_path)
    print_pseudo_label_info(tar, train_size=len(trainids))
    return tar


from nltk.corpus import stopwords
import string
def preprocess(df):
    print("Preprocessing data..", flush=True)
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    for index, row in df.iterrows():
        if index % 100 == 0:
            print("Finished rows: " + str(index) + " out of " + str(len(df)), flush=True)
        line = row["text"]
        words = line.strip().split()
        new_words = []
        for word in words:
            word_clean = word.translate(str.maketrans('', '', string.punctuation))
            if len(word_clean) == 0 or word_clean in stop_words:
                continue
            new_words.append(word_clean)
        df["text"][index] = " ".join(new_words)
    return df

# from tensorflow.keras.preprocessing.text import Tokenizer
def fit_get_tokenizer(data, max_words):
    tokenizer = Tokenizer(num_words=max_words, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data)
    return tokenizer

def generate_pseudo_labels(df, labels, label_term_dict, tokenizer):
    def argmax_label(count_dict):
        maxi = 0
        max_label = None
        keys = sorted(count_dict.keys())
        for l in keys:
            count = 0
            for t in count_dict[l]:
                count += count_dict[l][t]
            if count > maxi:
                maxi = count
                max_label = l
        return max_label

    y = []
    X = []
    y_true = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    for index, row in df.iterrows():
        print("[%i/%i]" % (index, len(df)), end='\r')
        line = row["text"]
        label = row["label"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])
        count_dict = {}
        flag = 0
        for l in labels:
            seed_words = set()
            for w in label_term_dict[l]:
                seed_words.add(w)
            int_labels = list(set(words).intersection(seed_words))
            if len(int_labels) == 0:
                continue
            for word in words:
                if word in int_labels:
                    flag = 1
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    try:
                        count_dict[l][word] += 1
                    except:
                        count_dict[l][word] = 1
        if flag:
            lbl = argmax_label(count_dict)
            if not lbl:
                continue
            y.append(lbl)
            X.append(index)
            y_true.append(label)
    return X, y, y_true

