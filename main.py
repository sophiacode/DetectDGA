import os
import pickle
import warnings
import tensorflow as tf

from keras.backend import set_session

from detect.classifiers import *
from data_utils import DataUtils, load_data


def build_classifier(model_name):
   if model_name == "lstm":
      classifier = LSTMClassifier()
   if model_name == "gru":
      classifier = GRUClassifier()
   if model_name == "bilstm":
      classifier = BiLSTMClassifier()
   if model_name == "expose":
      classifier = EXposeClassifier()
   if model_name == "cnnlstm":
      classifier = CNNLSTMClassifier()
   if model_name == "stacked":
      classifier = StackedCNNClassifier()
   if model_name == "tcn":
      classifier = TCNClassifier()
   if model_name == "attention":
      classifier = AttentionClassifier()
   if model_name == "capsule":
      classifier = CapsuleClassifier()
   
   return classifier

if __name__ == "__main__":
   warnings.filterwarnings('ignore')

   os.environ["CUDA_VISIBLE_DEVICES"] = "0"
   config = tf.ConfigProto()
   config.gpu_options.allow_growth = True
   session = tf.Session(config=config)
   set_session(session)

   data_file = "./dataset/.cache/data.pkl"
   with open(data_file, "rb") as fp:
      dataUtils = pickle.load(fp) 
        
   classifier = build_classifier("tcn")
   classifier.load_data(dataUtils, data_file)
   classifier.run(mode=0)


