import numpy as np
from svm import SVM
from kernels import Kernel

positives = [[4,4], [5,5]]#[[5,1], [6,-1], [7,3]]
negatives = [[2,1], [1,2]]#[[1,7], [2,8], [3,8]]

data_dict = {-1: np.array(negatives), 1: np.array(positives)}

svm = SVM(weights=np.array([0.1, 0.2]), bias=0.1, data=data_dict, kernel=Kernel.linear(), c=1)

svm.fit()
# test_data = [[0,10], [1,3], [3,4], [3,5], [5,5], [5,6], [6,-5], [5,8]]

# print(svm.predict(test_data))
svm.plot()
