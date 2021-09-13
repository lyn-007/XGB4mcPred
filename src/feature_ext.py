import numpy as np
from train_need import read_feature_data, Model_Evaluate
import os


def read_fasta_file(file_name):
    with open(file_name, encoding='utf-8') as Data_file:
        file_str = ''
        for i, line in enumerate(Data_file):
            file_str += line

    all_sequence = file_str.split('\n')
    len_sequence = len(all_sequence)
    sequence_dict = {}
    for i in range(0, len_sequence-1, 2):
        if all_sequence[i][:2] == ">P":
            seq_label = 1
        else:
            seq_label = 0
        sequence_dict[all_sequence[i]] = [all_sequence[i+1], seq_label]

    return sequence_dict


def one_hot(sequence):
    nucleotides = {
    'A' : [1,0,0,0],
    'C' : [0,1,0,0],
    'G' : [0,0,1,0],
    'T' : [0,0,0,1]
    }

    vector = []
    for i, nucleo in enumerate(sequence):
        vector += nucleotides[nucleo]
    return vector


def two_hot(sequence):
    nucleotides = {
    'AA' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    'AC' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    'AG' : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    'AT' : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    'CA' : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    'CC' : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    'CG' : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    'CT' : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    'GA' : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    'GC' : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    'GG' : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    'GT' : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    'TA' : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    'TC' : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'TG' : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'TT' : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    }

    vector = []
    for i in range(len(sequence)-1):
        vector += nucleotides[sequence[i:i+2]]
    return vector


def two_gap1_hot(sequence):
    nucleotides = {
    'AA' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    'AC' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    'AG' : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    'AT' : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    'CA' : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    'CC' : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    'CG' : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    'CT' : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    'GA' : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    'GC' : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    'GG' : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    'GT' : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    'TA' : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    'TC' : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'TG' : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'TT' : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    }

    vector = []
    for i in range(len(sequence)-2):
        vector += nucleotides[sequence[i]+sequence[i+2]]
    return vector


def two_gap2_hot(sequence):
    nucleotides = {
    'AA' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    'AC' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    'AG' : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    'AT' : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    'CA' : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    'CC' : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    'CG' : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    'CT' : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    'GA' : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    'GC' : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    'GG' : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    'GT' : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    'TA' : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    'TC' : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'TG' : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'TT' : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    }

    vector = []
    for i in range(len(sequence)-3):
        vector += nucleotides[sequence[i]+sequence[i+3]]
    return vector


def two_gap3_hot(sequence):
    nucleotides = {
    'AA' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    'AC' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    'AG' : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    'AT' : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    'CA' : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    'CC' : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    'CG' : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    'CT' : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    'GA' : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    'GC' : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    'GG' : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    'GT' : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    'TA' : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    'TC' : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'TG' : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'TT' : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    }

    vector = []
    for i in range(len(sequence)-4):
        vector += nucleotides[sequence[i]+sequence[i+4]]
    return vector



# def get_sequecne(seq, w, lambd):
#     psednc = calc_pseDNC(seq, w, lambd)
#     k_mer1 = make_kmer_vector(seq, 1)
#     k_mer2 = make_kmer_vector(seq, 2)
#     ncp = NCP(seq)
#     one = one_hot(seq)
#     two = TWO_HOT(seq)
#     nd = ND(seq)
#     dd = DD(seq)

#     fusion_vec = np.hstack((k_mer2, k_mer1, ncp, nd, dd, one, two, psednc))
#     return fusion_vec
