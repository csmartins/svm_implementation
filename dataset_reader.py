# -*- coding: utf-8 -*-

import sys
import numpy as np

POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1

# Lê o dataset na forma (X, y) onde:
# X = Lista de todos os samples, onde cada sample é a lista de seus atributos
# y = Lista de classes dos respectivos samples em y
# Esse formato é utilizado em algumas implementações de classificadores do scikit
def read_dataset(path):
    X = []
    y = []

    dataset_file = open(path, 'r')
    for line in dataset_file:
        sample = []

        attributes = line.split(',')
        for attribute in attributes[:-1]:
            sample.append(float(attribute.strip()))

        clazz = int(attributes[-1].strip())

        X.append(sample)
        y.append(clazz)

    dataset_file.close()

    y = fix_classes(y)

    return X, y

# Converte o dataset fa forma (X, y) para um dict separado por calsse
# Retorno: {-1: samples_negativos, 1: samples_positivos}
# Esse formato é utilizado na nossa implementação do SVM
def build_data_dict(X, y):
    temp_data_dict = {NEGATIVE_CLASS: [], POSITIVE_CLASS: []}

    for i in range(len(y)):
        temp_data_dict[y[i]].append(X[i])

    negatives = temp_data_dict[NEGATIVE_CLASS]
    positives = temp_data_dict[POSITIVE_CLASS]
    data_dict = {NEGATIVE_CLASS: np.array(negatives), POSITIVE_CLASS: np.array(positives)}

    return data_dict

# Lê o dataset e passa para o formato utilizado no na nossa implementação do SVM.
def read_data_dict(path):
    (X, y) = read_dataset(path)
    data_dict = build_data_dict(X,y)
    return data_dict

# Corrige as classes dos datasets (podem estar 0|1, 1|2 etc) para -1|1
def fix_classes(classes):
    classes_dict = {}

    for entry in classes:
        classes_dict[entry] = True

    different_classes = classes_dict.keys()

    positive_class = max(different_classes)

    fixed_classes = []

    for clazz in classes:
        if clazz == positive_class:
            fixed_classes.append(POSITIVE_CLASS)
        else:
            fixed_classes.append(NEGATIVE_CLASS)

    return fixed_classes

if __name__ == "__main__":
    dataset_path = sys.argv[1]

    data_dict = read_data_dict(dataset_path)

    print "Positives: {}".format(len(data_dict[POSITIVE_CLASS]))
    print "Negatives: {}".format(len(data_dict[NEGATIVE_CLASS]))
