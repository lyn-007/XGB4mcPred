import pandas as pd
import numpy as np
import math


def read_fasta_file(file_name):
    with open(file_name, encoding='utf-8') as Data_file:
        file_str = ''
        for i, line in enumerate(Data_file):
            file_str += line

    all_sequence = file_str.split('\n')
    len_sequence = len(all_sequence)
    sequence_dict = {}
    for i in range(0, len_sequence-1, 2):
        sequence_dict[all_sequence[i]] = [all_sequence[i+1], 'Non-4mC']

    return sequence_dict


def read_feature_data(data_path):

    Data = pd.read_csv(data_path)
    len_colums = len(np.array(Data.T[1])) - 1

    colums = []
    for i in range(len_colums-1):
        colums.append(str(i))
    Data_feature = Data[:][colums].values
    Data_label = Data[:][str(len_colums-1)].values
    X_train = Data_feature
    Y_train = Data_label
    return X_train, Y_train

    
    
def Model_Evaluate(confus_matrix):
    TN, FP, FN, TP = confus_matrix.ravel()
    
    SN = TP / ( TP + FN )
    SP = TN / ( TN + FP )
    ACC = ( TP + TN ) / ( TP + TN + FN + FP )
    MCC = (( TP * TN ) - ( FP * FN )) / (math.sqrt(( TP + FN ) * ( TP + FP ) * ( TN + FP ) * ( TN + FN )))
    
    print("Model score --- SN:{0:<20}SP:{1:<20}ACC:{2:<20}MCC:{3:<20}\n".format(SN, SP, ACC, MCC))
    
    return SN, SP, ACC, MCC
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    