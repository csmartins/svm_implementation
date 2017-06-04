import numpy as np
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, bias, weights, data):
        self.bias = bias
        self.weights = weights
        self.data_dict = data

    def plot(self):
        colors = {1:'r',-1:'b'}
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        [[ax.scatter(x[0],x[1],s=100,color=colors[i]) for x in self.data_dict[i]] for i in self.data_dict]

        plt.show()

    def predict(self, features):
        features = np.array(features)
        dot_product = np.dot(features, self.weights)
        classification = np.sign(dot_product + self.bias)

        return classification

    def fit(self):
        pass
