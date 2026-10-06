import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.kernel_approximation import Nystroem


# Kernels propios (reciben dos vectores 1-D). Son funciones de módulo,
# por lo que el modelo se puede guardar con joblib/pickle.
def triangle(x, y, gamma=0.1):
    return max(0.0, 1 - np.linalg.norm(x - y) / gamma)

def canberra(x, y, gamma=0.1):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = gamma * np.abs(x - y) / (np.abs(x) + np.abs(y))
    return 1 - np.sum(r[~np.isnan(r)]) / x.shape[0]

def truncated(x, y, gamma=0.1):
    v = 1 - np.abs(x - y) / gamma
    return np.sum(v[v > 0]) / x.shape[0]


# nombre -> (kernel para Nystroem, parametros del estimador que usa)
KERNELS = {
    "rbf":        ("rbf",        ["gamma"]),
    "poly":       ("polynomial", ["gamma", "degree"]),   # coef0 fijo en 1: (gamma<x,y>+1)^degree
    "hyperbolic": ("sigmoid",    ["gamma", "coef0"]),
    "triangle":   (triangle,     ["gamma"]),
    "can":        (canberra,     ["gamma"]),
    "tru":        (truncated,    ["gamma"]),
}


class KRidgeClassifier(RidgeClassifier):
    """RidgeClassifier + truco kernel (aproximacion de Nystrom).

    Z = phi(X) con Nystrom (n_components puntos de referencia) y luego Ridge sobre Z.
    Con n_components >= n_muestras equivale a kernel ridge exacto.
    """
    def __init__(self, alpha=1.0, fit_intercept=True, copy_X=True,
                 max_iter=None, tol=1e-4, class_weight=None,
                 solver="auto", positive=False, random_state=None,
                 kernel="rbf", degree=2, gamma=0.1, coef0=1.0,
                 n_components=100):
        super().__init__(
            alpha=alpha, fit_intercept=fit_intercept, copy_X=copy_X,
            max_iter=max_iter, tol=tol, class_weight=class_weight,
            solver=solver, positive=positive, random_state=random_state,
        )
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.n_components = n_components

    def _feature_map(self, n_samples):
        func, used = KERNELS[self.kernel]
        params = {p: getattr(self, p) for p in used}
        if self.kernel == "poly":
            params["coef0"] = 1.0
        return Nystroem(kernel=func, kernel_params=params,
                        n_components=min(self.n_components, n_samples),
                        random_state=self.random_state)

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X)
        if self.kernel == "linear":
            self.nystroem_ = None
            return super().fit(X, y, sample_weight=sample_weight)
        self.nystroem_ = self._feature_map(X.shape[0])
        Z = self.nystroem_.fit_transform(X)
        return super().fit(Z, y, sample_weight=sample_weight)

    def decision_function(self, X):
        X = np.asarray(X)
        if getattr(self, "nystroem_", None) is not None:
            X = self.nystroem_.transform(X)
        return super().decision_function(X)
