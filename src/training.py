from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from xgboost import XGBClassifier
from train_need import read_feature_data, Model_Evaluate
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import numpy as np
import joblib
import gc
import warnings
import sys
from config import thresh_list, paramas_dicts
warnings.filterwarnings('ignore')

from numpy import sort

# xgb 0.90
# sklearn 0.20.0
file_names = ['E.co','C.el','D.me','A.th','G.su','G.pi']
# file_names = ['G.su']


roc_auc_pred = []
for i, name in enumerate(file_names):
    print("==============={0}==============".format(name))
    file_name = name
    train_file = '../Benchmark_Datasets/GAP_{0}.csv'.format(file_name)
    test_file = '../Independent_Datasets/GAP_{0}_indep.csv'.format(file_name)
    X_train, Y_train = read_feature_data(train_file)
    X_test, Y_test = read_feature_data(test_file)

    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, Y_train)
    skf = StratifiedKFold(n_splits=10, random_state=777, shuffle=True)
    thresh = thresh_list[file_name]
    paramas_dict = paramas_dicts[file_name]

    selection = SelectFromModel(xgb_model, threshold=thresh, prefit=True)
    # joblib.dump(selection, '{0}_selection'.format(name))
    # selection = joblib.load('{0}_selection'.format(name))

    select_X_train = selection.transform(X_train)

    selection_model = XGBClassifier(**paramas_dict)
    selection_model.fit(select_X_train, Y_train)
    
    select_X_test = selection.transform(X_test)
    y_pred = cross_val_predict(selection_model, select_X_train, Y_train, cv = skf)
    _confusion = confusion_matrix(y_pred, Y_train)
    print(_confusion)
    SN, SP, ACC, MCC = Model_Evaluate(_confusion)

    # selection_model = joblib.load('work/{0}.pkl'.format(name))
    y_pred2 = selection_model.predict_proba(select_X_test)[:,1]
    roc_auc_pred.append(y_pred2)
    predictions = [round(value) for value in y_pred2]
    accuracy = accuracy_score(Y_test, predictions)
    print("Thresh=%f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

