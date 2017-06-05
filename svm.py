import numpy as np
import matplotlib.pyplot as plt
import cvxopt

class SVM:
    def __init__(self, bias, weights, data, kernel, c):
        self._c = c
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
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

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
        # [ax.scatter(x[0], x[1], s=100, color='k') for x in self.features]
        # plt.show()


        #pv_1 = (-self.weights[0]*self.support_vectors[0]-self.bias+1) / self.weights[1]
        #pv_2 = (-self.weights[0]*self.support_vectors[0]-self.bias+1) / self.weights[1]
        #ax.plot(self.support_vectors, [pv_1], 'k')
        #
        # nv_1 = (-self.weights[0]*self.support_vectors[1]-self.bias-1) / self.weights[1]
        # nv_2 = (-self.weights[0]*self.support_vectors[1]-self.bias-1) / self.weights[1]
        # ax.plot(self.support_vectors, [nv_1, nv_2], 'k')
        #
        # l_1 = (-self.weights[0]*self.support_vectors[0]-self.bias-0) / self.weights[1]
        # l_2 = (-self.weights[0]*self.support_vectors[1]-self.bias-0) / self.weights[1]
        # ax.plot(self.support_vectors, [l_1, l_2], 'k')

        plt.show()

    def predict(self, features):
        self.features = np.array(features)
        classes = []
        for feat in features:
            dot_product = np.dot(feat, self.weights)
            classes.append(np.sign(dot_product + self.bias))

        return classes

    def fit(self):
        lagrange_multipliers = self.get_lagrange_multipliers(self.X, self.Y)
        indices = lagrange_multipliers > 1e-5
        self.weights = lagrange_multipliers[indices]
        print(self.weights)
        self.support_vectors = self.X[indices]
        #print(self.support_vectors)
        self.support_vectors_labels = self.Y[indices]
        #print(self.support_vectors_labels)
        results = []
        for (y, x) in zip(self.support_vectors_labels, self.support_vectors):
            pred = 0
            for z_i, x_i, y_i in zip(self.weights, self.support_vectors, self.support_vectors_labels):
                pred += z_i * y_i * self.kernel(x_i, x)
            #pred = np.sign(pred).item()
            results.append(y - pred)

        #results = np.sign(results)
        self.bias = np.mean(results)
        print(self.bias)
