import joblib
import numpy as np
import pandas as pd
from feature_ext import one_hot, two_hot, two_gap1_hot, two_gap2_hot, two_gap3_hot
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from train_need import read_feature_data, Model_Evaluate
from numpy import sort
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
import gc
import warnings
import sys
warnings.filterwarnings('ignore')

    
def get_feature(seqs):
    features = []
    for name, seq in seqs.items():
        seq = seq[0]
        vec = one_hot(seq)+two_hot(seq)+two_gap1_hot(seq)+two_gap2_hot(seq)+two_gap3_hot(seq)
        features.append(vec)

    return np.array(features)


def get_predictions(seqs, name):

    features = get_feature(seqs)
    selection = joblib.load('../feature_pkl/{0}_selection'.format(name))
    select_X = selection.transform(features)

    selection_model = joblib.load('../pkl/{0}.pkl'.format(name))
    y_pred = selection_model.predict(select_X)
    predictions = [round(value) for value in y_pred]

    for i, (name, seq_list) in enumerate(seqs.items()):
        seqs[name] = [seq_list[0], seq_list[1], seq_list[2], predictions[i]]
    return seqs


def predicted(species, sequence):
    return_data = get_predictions(sequence, species)
    return return_data





