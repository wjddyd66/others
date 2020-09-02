import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from scipy import linalg
from math import sqrt
from sklearn.metrics.pairwise import rbf_kernel


class ELM(BaseEstimator, ClassifierMixin):
    # Input -> Hidden Layer -> Output
    def __init__(self, hid_num, linear, ae, activation='sigmoid'):
        # hid_num (int): number of hidden neurons
        self.hid_num = hid_num
        # linear (bool): Linear ELM AutoEncoder or Non Linear AutoEncoder
        self.linear = linear
        # ae (bool): ELM AutoEncoder or ELM Classifier
        self.ae = ae
        # Activation Function: Sigmoid or ReLu or Tanh
        self.activation = activation

    # Sigmoid Function with Clip Data
    def _sigmoid(self, x):
        # Sigmoid Function with Clip
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-1 * x))

    # ReLU Function
    def _relu(self, x):
        return x * (x > 0)

    # Tanh Function with Clip Data
    def _tanh(self, x):
        return 2*self._sigmoid(2*x)-1

    # For Last Layer => Classification => Multiclass
    def _ltov(self, n, label):
        # Trasform label scalar to vecto
        return [-1 if i != label else 1 for i in range(1, n + 1)]

    # Weight Initialization => Unit Orthogonal Vector, Unit Vector
    def weight_initialization(self, X):
        # Weight Initialization => Orthogonal Matrix, Scaling = 1
        u, s, vh = np.linalg.svd(np.random.randn(self.hid_num, X.shape[1]), full_matrices=False)
        W = np.dot(u, vh)

        # Bias Initialization => Unit Vector
        b = np.random.uniform(-1., 1., (1, self.hid_num))
        # find inverse weight matrix
        length = np.linalg.norm(b)
        b = b / length

        return W, b

    # For Fista Algorithm
    def _soft_thresh(self, x, l):
        return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

    # Fista algorithm for L1 regularization
    def fista(self, X, Y, l, maxit):
        if not self.ae:
            x = np.zeros(X.shape[1])
        else:
            x = np.zeros((X.shape[1], Y.shape[1]))
        t = 1
        z = x.copy()
        L = np.maximum(linalg.norm(X) ** 2, 1e-4)

        for _ in range(maxit):
            xold = x.copy()
            z = z + X.T.dot(Y - X.dot(z)) / L
            x = self._soft_thresh(z, l / L)
            t0 = t
            t = (1. + sqrt(1. + 4. * t ** 2)) / 2.
            z = x + ((t0 - 1.) / t) * (x - xold)
        return x

    # Training => Find β
    def fit(self, X, y, iteration=1000, l=0.01):

        # For One Single Layer AutoEncoder
        if not self.ae:
            # number of class, number of output neuron
            self.out_num = max(y)

            if self.out_num != 1:
                y = np.array([self._ltov(self.out_num, _y) for _y in y])

        # Orthogonal Unit Matrix
        self.W, self.b = self.weight_initialization(X)

        # Linear ELM Auto Encoder
        # H = Orthogonal Matrix Mapping
        if self.linear:
            self.H = np.dot(X, self.W.T) + self.b
            u, s, vh = np.linalg.svd(self.H, full_matrices=False)
            self.H = np.dot(u, vh)

        # Non Linear EML Auto Encoder
        # H = Sigmoid(wx + b) or ReLU(wx + b) or Tanh(wx + b)
        else:
            # Activation Function => Sigmoid Function
            if self.activation == 'sigmoid':
                self.H = self._sigmoid(np.dot(self.W, X.T) + self.b.T).T

            # Activation Function => ReLU Function
            elif self.activation == 'relu':
                self.H = self._relu(np.dot(self.W, X.T) + self.b.T).T

            # Activation Function => Tanh Function
            else:
                self.H = self._tanh(np.dot(self.W, X.T) + self.b.T).T

        # Single Layer ELM or For ELM AutoEncoder
        if not self.ae:
            self.beta = self.fista(self.H, y, l, iteration)

        else:
            self.beta = self.fista(self.H, X, l, iteration)

        return self

    # if One Single Layer ELM => Predict
    def predict(self, X):

        if self.linear:
            H = np.dot(X, self.W.T) + self.b
            u, s, vh = np.linalg.svd(H, full_matrices=False)
            H = np.dot(u, vh)
            y = np.dot(H, self.beta)

        else:
            H = self._sigmoid(np.dot(self.W, X.T) + self.b.T)
            y = np.dot(H.T, self.beta)

        if self.ae == True:
            return y

        else:
            return np.sign(y)


class Linear_ELM_AE(ELM, BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_units):
        # hidden_uinits (tuple) : Num of hidden layer
        self.hidden_uinits = hidden_units
        # For Hidden space => Training β
        self.betas = []

    # Linear AutoEncoder ELM
    # - First Layer => H = Xβ^{-1}
    # - Other Layer => H = Xβ^{T} (X is Orthogonal Matrix)
    def calc_hidden_layer(self, X):
        for i, beta in enumerate(self.betas):
            if i == 0:
                X = np.dot(X, np.linalg.pinv(beta))
            else:
                X = np.dot(X, beta.T)
        return X

    # Stacking AutoEncoder Layer
    def fit(self, X, iteration=1000):
        input = X
        # Reset β
        self.betas = []

        for i, hid_num in enumerate(self.hidden_uinits):
            self.elm = ELM(hid_num, linear=True, ae=True)
            self.elm.fit(input, input, iteration)
            self.betas.append(self.elm.beta)
            input = self.calc_hidden_layer(X)

        return self

    # For AutoEncoder Layer => Hidden Layer 0,1,2...
    def feature_extractor(self, X, layer_num):
        for i, beta in enumerate(self.betas[:layer_num + 1]):
            if i == 0:
                X = np.dot(X, np.linalg.pinv(beta))
            else:
                X = np.dot(X, beta.T)

        return X


class Non_Linear_ELM_AE(ELM, BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_units):
        # hidden_uinits (tuple) : Num of hidden layer
        self.hidden_uinits = hidden_units
        # For Hidden space => Training β
        self.betas = []

    # Non_Linear AutoEncoder ELM
    # - All Layer => H = Xβ^{-1}
    def calc_hidden_layer(self, X):
        for i, beta in enumerate(self.betas):
            X = np.dot(X, np.linalg.pinv(beta))
        return X

    # Stacking AutoEncoder Layer
    def fit(self, X, iteration=1000):
        input = X
        # Reset β
        self.betas = []

        for i, hid_num in enumerate(self.hidden_uinits):
            self.elm = ELM(hid_num, linear=False, ae=True)
            self.elm.fit(input, input, iteration)
            self.betas.append(self.elm.beta)
            input = self.calc_hidden_layer(X)

        return self

    # For AutoEncoder Layer => Hidden Layer 0,1,2...
    def feature_extractor(self, X, layer_num):
        for i, beta in enumerate(self.betas[:layer_num + 1]):
            X = np.dot(X, np.linalg.pinv(beta))

        return X


class KELM(BaseEstimator, ClassifierMixin):
    # Input -> Hidden Layer -> Output
    def __init__(self):
        # Kernel
        self.kernel = None
        # l (float) : regularization term
        self.l = 0.001

    def fit(self, X, y, l=0.001):
        self.X = X
        self.y = y

        self.out_num = max(y)

        # Train Kernel and Hidden Space
        self.kernel = rbf_kernel(self.X, self.X)
        self._H = np.linalg.inv(np.diag(np.tile(l, self.kernel.shape[0])) + self.kernel) @ y

        return self

    def predict(self, test_X):
        # Predict by rbf kernel
        y = np.ones((self.X.shape[0], test_X.shape[0]))

        for i in range(self.X.shape[0]):
            y[i] = np.array(rbf_kernel(test_X, self.X[i].reshape(1, -1))).squeeze()

        y = np.dot(y.T, self._H)

        if self.out_num == 1:
            return np.sign(y)
        else:
            return np.argmax(y, 1) + np.ones(y.shape[0])

    def probability(self, test_X):
        # Predict by rbf kernel
        y = np.ones((self.X.shape[0], test_X.shape[0]))

        for i in range(self.X.shape[0]):
            y[i] = np.array(rbf_kernel(test_X, self.X[i].reshape(1, -1))).squeeze()

        y = np.dot(y.T, self._H)

        return y
