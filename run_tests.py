# -*- coding: utf-8 -*-

#########################################################################
# Para adicionar outro tipo de classificador é só criar uma função que  #
# recebe X_train, y_train e X_test, cria o classificador, classifica    #
# X_test e retorna as previsões em um array. Essa função deve ser       #
# adicionada no METHOD_FUNCTION_DICT na main.                           #
#########################################################################

import sys
import numpy as np
import math

from sklearn import svm as scipy_svm
from sklearn.metrics import accuracy_score

import dataset_reader
import k_fold

DEFAULT_FOLDS = 10
POS = 1
NEG = -1

# Testa um classificador em um dataset
# X: lista de samples do dataset
# y: classes dos respectivos samples
# classifier_test_function: função que recebe os conjuntos de treinamento e teste,
#                           treina um classificador e retorna as predições para o
#                           conjunto de teste
def test_classifier(X, y, classifier_test_function, verbose=False):
    folds = k_fold.split(X, y, DEFAULT_FOLDS)

    test_metrics = []

    for i in range(DEFAULT_FOLDS):
        if verbose:
            print "Running {}".format(i)

        fold = folds[i]
        X_test = fold[0]
        y_test = fold[1]

        (X_train, y_train) = k_fold.get_train_set(folds, i)

        y_pred = classifier_test_function(X_train, y_train, X_test)

        metrics = calculate_metrics(y_pred, y_test, verbose=verbose)
        test_metrics.append(metrics)

    average_metrics = calculate_average_metrics(test_metrics)
    return average_metrics

# Recebe a lista de classes preditas pelo classificador e a lista real
# Retorna um dict com as métricas calculadas: accuracy, precision, recall e f-measure
def calculate_metrics(y_pred, y_real, verbose=False):
    predictions = len(y_pred)

    # Matriz para mapear [classe real] -> [predição]
    confusion_matrix = {NEG: {NEG: 0, POS: 0},
                         POS: {NEG: 0, POS: 0}}

    for i in range(predictions):
        confusion_matrix[y_real[i]][y_pred[i]] += 1

    #Para facilitar a leitura do código:
    #true positives
    tp = confusion_matrix[POS][POS]
    #false positives
    fp = confusion_matrix[NEG][POS]
    #true negatives
    tn = confusion_matrix[NEG][NEG]
    #false negatives
    fn = confusion_matrix[POS][NEG]

    accuracy = float(tp + tn) / predictions

    try:
        precision = float(tp) / (tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    try:
        recall = float(tp) / (tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    try:
        mcc = float(tp*tn + fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    except ZeroDivisionError:
        mcc = 0.0

    if verbose:
        print_confusion_table(confusion_matrix)
        print "Acc: {}".format(accuracy)
        print "MCC: {}".format(mcc)
        print ""

    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'matthews': mcc}
    return metrics

def print_confusion_table(confusion_matrix):
    print " r/p ___1___ __-1___ "
    print "  1 | %5d | %5d |" % (confusion_matrix[POS][POS], confusion_matrix[POS][NEG])
    print " -1 | %5d | %5d |" % (confusion_matrix[NEG][POS], confusion_matrix[NEG][NEG])

# Calcula as médias das métricas tiradas em cada teste do k-fold
def calculate_average_metrics(metrics_list):
    average_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'matthews': 0.0}

    for test_metrics in metrics_list:
        for metric in test_metrics.keys():
            average_metrics[metric] += test_metrics[metric]

    for metric in average_metrics.keys():
        average_metrics[metric] = average_metrics[metric] / len(metrics_list)

    return average_metrics

# Função para testar a implementação de SVM do scipy
# Cria um classificador, treina com o conjunto de treinamento
# Retorna a predição para o conjunto de teste
def scipy_SVM_test_function(X_train, y_train, X_test):
    clf = scipy_svm.SVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return y_pred

# Mostra as instruções de uso do programa
def print_help():
    print "Usage: {} method dataset_path [-v]".format(sys.argv[0])
    print "Methods available: {}".format(METHOD_FUNCTION_DICT.keys())

if __name__ == "__main__":
    METHOD_FUNCTION_DICT = {'scipy-svm': scipy_SVM_test_function}

    if len(sys.argv) < 3 or sys.argv[1] not in METHOD_FUNCTION_DICT.keys():
        print_help()
    else:
        method = sys.argv[1]
        dataset_path = sys.argv[2]
        verbose = len(sys.argv) > 3 and sys.argv[3] == '-v'

        (X, y) = dataset_reader.read_dataset(dataset_path)
        func = METHOD_FUNCTION_DICT[method]
        metrics = test_classifier(X, y, scipy_SVM_test_function, verbose=verbose)

        if verbose:
            print ""

        print "RESULTS FOR [{}] RUNNING ON [{}]:".format(method, dataset_path)
        print "Accuracy: {}".format(metrics['accuracy'])
        print "Precision: {}".format(metrics['precision'])
        print "Recall: {}".format(metrics['recall'])
        print "Matthew's Correlation Coefficient: {}".format(metrics['matthews'])
