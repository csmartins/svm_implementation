# -*- coding: utf-8 -*-

import random

# Retorna uma lista aleatória de indíces a serem usados no k-fold
def choose_indices(k, dataset_size):
    fold_size = dataset_size / k

    indices = range(dataset_size)
    random.shuffle(indices)

    folds = []
    for i in range(k-1):
        fold_start = i*fold_size
        fold_end = (i+1)*fold_size

        fold = indices[fold_start : fold_end]
        folds.append(fold)

    last_fold = indices[(k-1)*fold_size : ]
    folds.append(last_fold)

    return folds

# Separa o dataset em k partes.
# O retorno segue o formato [(X1, y1), (X2, y2), ...]
def split(X, y, k):
    fold_indices = choose_indices(k, len(X))

    folds = []

    for i in range(k):
        indices = fold_indices[i]
        fold_X = []
        fold_y = []

        for index in indices:
            fold_X.append(X[index])
            fold_y.append(y[index])

        fold = (fold_X, fold_y)
        folds.append(fold)

    return folds

# Monta o conjunto de treinamento com todas as folds, menos a de teste
def get_train_set(folds, test_fold_id):
    X_train = []
    y_train = []

    for i in range(len(folds)):
        if i != test_fold_id:
            X_train.extend(folds[i][0])
            y_train.extend(folds[i][1])

    return X_train, y_train
