import numpy as np
import matplotlib.pyplot as plt
import cvxopt

class SVM:
    def __init__(self, data, kernel, c):
        self._c = c
        self.kernel = kernel
        self.bias = 0.0
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
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        # cvxopt.solvers['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        return np.ravel(solution['x'])

    def fit(self):
        lagrange_multipliers = self.get_lagrange_multipliers(self.X, self.Y)
        # print(lagrange_multipliers)
        indices = \
                lagrange_multipliers > 1e-5


        # print( "Indices: ",indices);

        self.weights = lagrange_multipliers[indices]
        # print("Weights")
        # print(self.weights)
        self.support_vectors = self.X[indices]
        # print(self.support_vectors)
        self.support_vectors_labels = self.Y[indices]
        # print(self.support_vectors_labels)
        results = []
        for (y, x) in zip(self.support_vectors_labels, self.support_vectors):
            pred = self.bias
            for z_i, x_i, y_i in zip(self.weights, self.support_vectors, self.support_vectors_labels):
                pred += z_i * y_i * self.kernel(x_i, x)
            pred = np.sign(pred).item()
            results.append(y - pred)

        self.bias = np.mean(results)
        # print(self.bias)

    def predict(self, x):
        result = self.bias
        for z_i, x_i, y_i in zip(self.weights,
                                 self.support_vectors,
                                 self.support_vectors_labels):
            result += z_i * y_i * self.kernel(x_i, x)
        return np.sign(result).item()
