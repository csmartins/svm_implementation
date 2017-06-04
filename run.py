import numpy as np
from svm import SVM

positives = [[5,1], [6,-1], [7,3]]
negatives = [[1,7], [2,8], [3,8]]

data_dict = {-1: np.array(negatives), 1: np.array(positives)}

svm = SVM(weights=np.array([0.1, 0.2]), bias=0.1, data=data_dict)

test_data = [[0,10], [1,3], [3,4], [3,5], [5,5], [5,6], [6,-5], [5,8]]
svm.plot()