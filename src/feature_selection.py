from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
from train_need import read_feature_data, Model_Evaluate
from numpy import sort

# file_names = ['G.pi','E.co','A.th','D.me','G.su','C.el']

file_name = 'E.co'
train_file = '../Benchmark_Datasets/GAP_{0}.csv'.format(file_name)
test_file = '../Independent_Datasets/GAP_{0}_indep.csv'.format(file_name)
X_train, Y_train = read_feature_data(train_file)
X_test, Y_test = read_feature_data(test_file)

xgb_model = XGBClassifier()
xgb_model.fit(X_train, Y_train)

feature_sorted = xgb_model.feature_importances_
thresholds = sort(list(set(feature_sorted)))