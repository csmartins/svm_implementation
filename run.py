import numpy as np
from svm import SVM
from kernels import Kernel

positives = [[4,4], [5,5]]
# positives = [[5,1], [6,-1], [7,3]]

# positives = [[-0.23530383,  1.70585848],
#             [ 0.8157926 ,  0.04591391],
#             [ 0.03237168,  1.36243792]]

negatives = [[2,1], [1,2]]
# negatives = [[1,7], [2,8], [3,8]]

# negatives = [[-0.07810244, -0.65502153],
#             [ 0.25648505, -0.79438534],
#             [-0.83531028, -0.18554141],
#             [ 0.41896733, -0.73003242],
#             [-1.00007796,  0.00366544],
#             [-1.58005843,  0.83875439],
#             [ 0.77187267, -1.67242829]]

data_dict = {-1: np.array(negatives), 1: np.array(positives)}

svm = SVM(data=data_dict, kernel=Kernel.linear(), c=1)

svm.fit()
test_data = [[0,10], [1,3], [3,4], [3,5], [5,6], [6,-5], [5,8]]

print(svm.predict(test_data))
svm.plot()
