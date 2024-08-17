import numpy as np
import json
import pickle
from collections import OrderedDict
import math
import tensorflow as tf


class DataSet:
    def __init__(self, fpath, setting_no, seqlen, smilen, dataset_type, convert_to_log=False):
        self.fpath = fpath
        self.setting_no = setting_no
        self.SEQLEN = seqlen
        self.SMILEN = smilen
        self.dataset_type = dataset_type
        self.convert_to_log = convert_to_log

        self.charseqset = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                           "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                           "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                           "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24,
                           "Z": 25}
        self.charseqset_size = 25

        self.charsmiset = {"#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
                           ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
                           "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
                           "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
                           "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
                           "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
                           "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
                           "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
                           "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
                           "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
                           "t": 61, "y": 62}
        self.charsmiset_size = 62

        self.ligands = None
        self.proteins = None

    def read_sets(self):
        print(f"Reading {self.dataset_type} dataset from {self.fpath}")
        test_fold = json.load(open(self.fpath + f"folds/test_fold_setting{self.setting_no}.txt"))
        train_folds = json.load(open(self.fpath + f"folds/train_fold_setting{self.setting_no}.txt"))
        return test_fold, train_folds

    def parse_data(self):
        print(f"Parsing {self.dataset_type} dataset")
        self.ligands = json.load(open(self.fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
        self.proteins = json.load(open(self.fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
        Y = pickle.load(open(self.fpath + "Y", "rb"), encoding='latin1')

        if self.convert_to_log:
            Y = -(np.log10(Y / (math.pow(10, 9))))

        XD = [self.label_smiles(self.ligands[d], self.SMILEN, self.charsmiset) for d in self.ligands.keys()]
        XT = [self.label_sequence(self.proteins[t], self.SEQLEN, self.charseqset) for t in self.proteins.keys()]

        return XD, XT, Y

    @staticmethod
    def label_smiles(line, max_smi_len, smi_ch_ind):
        X = np.zeros(max_smi_len, dtype=np.int64)
        for i, ch in enumerate(line[:max_smi_len]):
            X[i] = smi_ch_ind[ch]
        return X

    @staticmethod
    def label_sequence(line, max_seq_len, smi_ch_ind):
        X = np.zeros(max_seq_len, dtype=np.int64)
        for i, ch in enumerate(line[:max_seq_len]):
            X[i] = smi_ch_ind[ch]
        return X

    def get_data(self):
        test_fold, train_folds = self.read_sets()
        XD, XT, Y = self.parse_data()

        XD = np.asarray(XD)
        XT = np.asarray(XT)
        Y = np.asarray(Y)

        label_row_inds, label_col_inds = np.where(np.isnan(Y) == False)

        return XD, XT, Y, label_row_inds, label_col_inds, test_fold, train_folds


def prepare_interaction_pairs(XD, XT, Y, rows, cols):
    drugs = []
    targets = []
    affinity = []

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target = XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind], cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data, target_data, affinity


def create_tf_dataset(drug_data, target_data, affinity, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(((drug_data, target_data), affinity))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(drug_data))
    dataset = dataset.batch(batch_size)
    return dataset
