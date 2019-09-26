import os
import random
import argparse
import pickle
import pandas as pd
import numpy as np
import wordsegment as ws

from tranco import Tranco
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, LeavePGroupsOut, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing import sequence, text
from keras.utils import to_categorical

class DataUtils():
    def __init__(self):
        self.domains = list()
        self.labels = list()
        self.x = list()
        self.yb = list()
        self.ym = list()
        self.idx = list()
        
    def load(self, csv_file, label, max_num, min_num=0):
        x = pd.read_csv(csv_file, header=None)[0]
        if len(x) >= min_num:
            if len(x) > max_num:
                x = x.sample(max_num)
            x = list(x)
            self.domains += x
            self.labels += [label] * len(x)

    def domains_to_x_char(self, maxlen=None, mode="post"):
        domains = self.domains

        valid_chars = {i:idx+1 for idx, i in enumerate(set(''.join(domains)))}
        n_words = len(valid_chars) + 1

        if maxlen is None:
            maxlen = np.max([len(i) for i in domains])

        x = np.array([[valid_chars[j] for j in i] for i in domains])
        x = sequence.pad_sequences(x, maxlen, padding=mode, truncating=mode)

        self.x = x
        self.n_words = n_words
        self.maxlen = maxlen

    def domains_to_x_word(self, maxlen=None, n_words=50000):
        domains = self.domains
        
        if maxlen is None:
            maxlen = np.max([len(i) for i in domains])

        ws.load()
        for i in tqdm(range(len(domains))):
            domain_labels = domains[i].split(".")
            words = list()
            for j in range(len(domain_labels)-1):
                segs = ws.segment(domain_labels[j])
                new_segs = list()
                for s in segs:
                    if s in ws.UNIGRAMS:
                        new_segs.append(s)
                    else:
                        new_segs += list(s)
                words += new_segs
            words.append(domain_labels[-1])
            domains[i] = words

        x = list()
        for domain in domains:
            x.append([text.one_hot(word, n_words, filters=" ")[0] for word in domain])
        x = sequence.pad_sequences(x, padding='post', maxlen=maxlen)

        self.x = x
        self.n_words = n_words
        self.maxlen = maxlen

    def labels_to_yb(self):
        self.yb = np.array([0 if i == 'benign' else 1 for i in self.labels])

    def labels_to_ym(self):
        self.n_classes = len(set(self.labels))
        
        encoder = LabelEncoder().fit(self.labels)
        ym = encoder.transform(self.labels)
        self.ym = to_categorical(ym, num_classes=self.n_classes)

        self.label_encoder = encoder.classes_
    
    def split(self, test_size=0.2):
        s = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for train, test in s.split(self.domains, self.labels):
            self.idx.append([train, test])
            return train, test

def load_data(cache_file, benign_num=1000000, dga_num=100000, min_num=0, 
              maxlen=None, mode="post", test_size=0.2):
    dataUtils = DataUtils()
    
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as fp:
            return pickle.load(fp) 
    
    else:
        benign_csv = "./dataset/benign/Tranco-top-1m.csv"
        dga_dir = './dataset/dgarchive/'

        dataUtils.load(benign_csv, "benign", benign_num)

        dga_csvs = os.listdir(dga_dir)
        for dga_csv in dga_csvs:
            label = dga_csv.split('_')[0]
            dataUtils.load(dga_dir+dga_csv, label, dga_num, min_num)
         
        dataUtils.domains_to_x_char(maxlen, mode)
        dataUtils.labels_to_yb()
        dataUtils.labels_to_ym()
        dataUtils.split(test_size)

        with open(cache_file, "wb") as fp:
            pickle.dump(dataUtils, fp)

    return dataUtils

def convert_txt_to_csv(txt_file, csv_dir):
    """ Covert 360/dga.txt to csv """
    domains = list()
    labels = list()

    with open(txt_file, "r") as f:
        for line in f:
            if(line[0] != '#' and line[0] != '\n'):
                domains.append(line.split('\t')[1])
                labels.append(line.split('\t')[0])
    
    if(not os.path.exists(csv_dir)):
        os.makedirs(csv_dir)

    label_set = set(labels)
    idx = 0
    for label in label_set:
        count = labels.count(label)
        df = pd.DataFrame(domains[idx:count], index=None, columns=None)
        df.to_csv(csv_dir + label + "_dga.csv", header=None, index=None)
        

if __name__ == "__main__":
    # convert_txt_to_csv("dataset/360/dga.txt", "dataset/360/csv/")

    parser = argparse.ArgumentParser(description='DGA_Detection')
    parser.add_argument('--cache_file', type=str, default="./dataset/.cache/data.pkl")
    parser.add_argument('--benign_csv', type=str, default="./dataset/benign/Alexa-top-1m.csv")
    parser.add_argument('--dga_dir', type=str, default='./dataset/360/csv/')
    parser.add_argument('--benign_num', type=int, default=1000000)
    parser.add_argument('--dga_max', type=int, default=500000)
    parser.add_argument('--dga_min', type=int, default=10)
    parser.add_argument('--maxlen', type=int, default=None)
    parser.add_argument('--mode', type=str, choices=("pre", "post"), default="post")
    parser.add_argument('--test_size', type=int, default=0.2)
    
    args = parser.parse_args()

    if(os.path.exists(args.cache_file)):
        print("This file has been created.")

    dataUtils = DataUtils()
    dataUtils.load(args.benign_csv, "benign", args.benign_num)

    dga_csvs = os.listdir(args.dga_dir)
    for dga_csv in dga_csvs:
        label = dga_csv.split('_')[0]
        dataUtils.load(args.dga_dir + dga_csv, label, args.dga_max, args.dga_min)
             
    dataUtils.domains_to_x_char(args.maxlen, args.mode)
    dataUtils.labels_to_yb()
    dataUtils.labels_to_ym()
    dataUtils.split(args.test_size)

    with open(args.cache_file, "wb") as fp:
        pickle.dump(dataUtils, fp)
    print("Done.")

    
        
    
    
    



        
    
    

