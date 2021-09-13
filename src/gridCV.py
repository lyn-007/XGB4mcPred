import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from xgboost import XGBClassifier
from train_need import read_feature_data, Model_Evaluate
from numpy import sort
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import gc
import warnings
import sys
warnings.filterwarnings('ignore')

def model_gridsearchCV(name, model, model_name, X_train, Y_train, X_test, Y_test, parama_dict):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=777)
    grid = GridSearchCV(model, param_grid=parama_dict, n_jobs=10, cv=skf)
    grid.fit(X_train, Y_train)

    best_param = grid.best_params_
    best_score = grid.best_score_
    print("==============BEGIN==================")
    print(model_name)
    print("%0.6f for %r"% (best_score, best_param))
    joblib.dump(grid ,"var/{0}_{1}_grid".format(model_name, name))

    print("----------------")
    model.set_params(**best_param)
    predicted = cross_val_predict(model, X_train, Y_train, cv=skf)
    confus_matrix = confusion_matrix(Y_train, predicted, labels=None, sample_weight=None)  
    SN, SP, ACC, MCC = Model_Evaluate(confus_matrix)
    
    model.set_params(**best_param)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    confus_matrix = confusion_matrix(Y_test, pred, labels=None, sample_weight=None)  
    SN_TEST, SP_TEST, ACC_TEST, MCC_TEST = Model_Evaluate(confus_matrix)
    print("==============END==================")
