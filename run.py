import numpy as np
from svm import SVM
from kernels import Kernel
from plot import plot

# positives = [[4,4], [5,5]]
positives = [[5,1], [6,-1], [7,3]]

# positives = [[-0.23530383,  1.70585848],
#             [ 0.8157926 ,  0.04591391],
#             [ 0.03237168,  1.36243792]]

# negatives = [[2,1], [1,2]]
negatives = [[1,7], [2,8], [3,8]]

# negatives = [[-0.07810244, -0.65502153],
#             [ 0.25648505, -0.79438534],
#             [-0.83531028, -0.18554141],
#             [ 0.41896733, -0.73003242],
#             [-1.00007796,  0.00366544],
#             [-1.58005843,  0.83875439],
#             [ 0.77187267, -1.67242829]]

data_dict = {-1: np.array(negatives), 1: np.array(positives)}

num_samples = len(positives) + len(negatives)
samples = np.append(np.array(positives), np.array(negatives)).reshape(num_samples, 2)
labels = np.append(np.ones(len(positives)), np.ones(len(negatives)) * -1)
svm = SVM(data=data_dict, kernel=Kernel.linear(), c=0.1)
svm.fit()

plot(svm, samples, labels, 20, "svm_run.pdf")

#print(svm.predict(test_data))
