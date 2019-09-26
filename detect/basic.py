import os
import math
import collections
import numpy as np
from datetime import datetime


from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder

from keras import metrics
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import *
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class BasicClassifier():
    def __init__(self, model_name):  
        self.model_name = model_name

    def load_data(self, data, data_name):
        """Load data from DataUtils (defined in data_utils.py)"""
        self.domains    = data.domains
        self.labels     = data.labels
        self.x          = data.x
        self.yb         = data.yb
        self.ym         = data.ym
        self.idx        = data.idx
        self.input_size = data.maxlen
        self.n_words    = data.n_words
        self.n_classes  = data.n_classes
        self.encoder    = data.label_encoder
        
        self.data_name  = data_name
    
    def run(self, mode, save_path=None, nfold=1, train_flag=True, batch_size=128, epochs=10, use_class_weights=False):
        """Train and evaluate.
        
        Arguments:
            mode {int} -- 0:binary / 1:multi-class / other:multi-task
        
        Keyword Arguments:
            save_path {str} -- Dir of saved model (default: {None})
            nfold {int} -- K in the k-fold cross validation (default: {1})
            train_flag {bool} -- Whether to train the model or not  (default: {True})
            batch_size {int} -- The number of batch size (default: {128})
            epochs {int} -- The number of train epochs (default: {10})
        """
        
        print("\n**********{0}**********".format(self.model_name))

        self.mode = mode
        if save_path is None:
            t = datetime.now().strftime("%d%H%M%S")
            self.save_path = "./out/{0}{1}-{2}".format(self.model_name, mode, t)
        else:
            self.save_path = save_path
        makedir(self.save_path)

        fold = 1
        for train, test in self.idx:
            model_path = os.path.join(self.save_path, "model")
            makedir(model_path)
            model_path = "{0}/{1}.h5".format(model_path, fold)

            if train_flag:
                self.train(train, batch_size, epochs, model_path, use_class_weights)
            
            #self.model = load_model(model_path)
            self.load_model(model_path)
            self.evaluation(test, fold)

            if fold >= nfold:
                break
            fold += 1
        
    def load_model(self, model_path):
        self.model = load_model(model_path)

    def build_model(self):
        inputs = Input(shape=(self.input_size, ))
        x = self.extract_features(inputs)
        out_b = Dense(1, activation="sigmoid", name="out_b")(x)
        out_m = Dense(self.n_classes, activation="softmax", name="out_m")(x)

        if self.mode == 0:
            model = Model(inputs=inputs, outputs=out_b)
            model.compile(optimizer="adam", loss="binary_crossentropy")
        
        elif self.mode == 1:
            model = Model(inputs=inputs, outputs=out_m)
            model.compile(optimizer="adam", loss="categorical_crossentropy")
        
        else:
            model = Model(inputs=inputs, outputs=[out_b, out_m])
            model.compile(optimizer="adam",
                loss={"out_b": "binary_crossentropy", "out_m": "categorical_crossentropy"},
                loss_weights={'out_b': 1., 'out_m': 0.2})
        
        model.summary()
        self.model = model

    def extract_features(self, x):
        return x

    def train(self, idx, batch_size, epochs, model_path, use_class_weights=False):
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()

        if self.mode == 0:
            y = self.yb[idx]
        elif self.mode == 1:
            y = self.ym[idx]
        else:
            y = {"out_b":self.yb[idx], "out_m":self.ym[idx]}

        if use_class_weights:
            weight_b = self.create_class_weight(self.yb[idx], 0.1)
            ym = np.argmax(self.ym[idx], axis=-1)
            weight_m = self.create_class_weight(ym, 0.3)

            if self.mode == 0:
                weight = weight_b
            elif self.mode == 1:
                weight = weight_m
            else:
                weight = {"oub_b": weight_b, "out_m": weight_m}
        else:
            weight = None    

        log_path = os.path.join(self.save_path, "log")
        makedir(log_path)

        self.model.fit(
            self.x[idx], 
            y,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=weight,
            callbacks=[EarlyStopping(patience=2),
                       TensorBoard(log_path),
                       ModelCheckpoint(model_path, save_best_only=True)])

    def create_class_weight(self, labels, mu):
        """Create weight based on the number of domain name in the dataset
        
           see details in <A LSTM based framework for handling multiclass imbalance in DGA botnet detection> 
        """
        counter = collections.Counter(labels)
        total = 0.0
        for value in counter.values():
            total += value
        
        class_weight = dict()
        for key in counter.keys():
            score = math.pow(total/float(counter[key]), mu)
            class_weight[key] = score

        return class_weight

    def evaluation(self, idx, fold):
        p_score = self.model.predict(self.x[idx])
        if self.mode == 0:
            self.evaluation_binary(p_score, self.yb[idx], fold)
        elif self.mode == 1:
            self.evaluation_multi(p_score, self.ym[idx], fold)
        else:
            self.evaluation_binary(p_score[0], self.yb[idx], fold)
            self.evaluation_multi(p_score[1], self.ym[idx], fold)

    def evaluation_binary(self, y_score, y_true, fold):
        auc = roc_auc_score(y_true, y_score)
        y_pred = [0 if i < 0.5 else 1 for i in y_score]
        prf = precision_recall_fscore_support(y_true, y_pred, average="binary", warn_for=[])
        print("fold{0}, b: \nauc={1}, prf={2}".format(fold, auc, prf))
        
        with open(os.path.join(self.save_path, "result.txt"), "a") as fp:
            t = datetime.now().strftime("%y-%m-%d %H:%M:%S")
            fp.write("Dataset: {0}\n".format(self.data_name))
            fp.write("{0} - Binary, fold{1}\n".format(t, fold))
            fp.write("Precision,\tRecall,\tF1Score,\tAUC\n")
            fp.write("{0},\t{1},\t{2},\t{3},\n\n".format(prf[0], prf[1], prf[2], auc))

    def evaluation_multi(self, y_score, y_true, fold):
        y_score = [i.argmax() for i in y_score]
        y_true = [i.argmax() for i in y_true]

        report = classification_report(y_true, y_score, target_names=self.encoder, digits=3)

        micro = precision_recall_fscore_support(y_true, y_score, average="micro", warn_for=None)
        macro = precision_recall_fscore_support(y_true, y_score, average="macro")
        weighted = precision_recall_fscore_support(y_true, y_score, average="weighted")
        print("fold{0}, m:".format(fold))
        print("micro: ", micro)
        print("macro: ", macro)
        print("weighted: ", weighted)

        with open(os.path.join(self.save_path, "result.txt"), "a") as fp:
           t = datetime.now().strftime("%y-%m-%d %H:%M:%S")
           fp.write("Dataset: {0}\n".format(self.data_name))
           fp.write("{0} - Multi-class, fold{1}\n".format(t, fold))
           fp.write(report)
           fp.write("\n")




        





    