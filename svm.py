import numpy as np
import matplotlib.pyplot as plt
import cvxopt

class SVM:
    def __init__(self, bias, weights, data, kernel):
        self.kernel = kernel
        self.bias = bias
        self.weights = weights
        self.data_dict = data
        self.X = np.append(self.data_dict[1], self.data_dict[-1], axis=0)

        self.Y = np.zeros(self.X.shape[0])
        len_pos = len(self.data_dict[1])
        len_neg = len(self.data_dict[-1])

        for i in range(0, len_pos):
            self.Y[i] = 1
        for i in range(0, len_neg):
            self.Y[i+len_pos] = -1

    def get_lagrange_multipliers(self, X, y):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))

        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self.kernel(x_i, x_j)

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self.bias)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        return np.ravel(solution['x'])

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
        multipliers = self.get_lagrange_multipliers(self.X, self.Y)
