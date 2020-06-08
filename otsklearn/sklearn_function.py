"""
    Scikit learn metamodels algorithm
    =================================
    Interpret sklearn algorithms as openturns Function
"""
import openturns as ot
import numpy as np
from sklearn.utils.validation import check_is_fitted

class SklearnPyFunction(ot.OpenTURNSPythonFunction):
    """
    Define a OpenTURNS Function using Machine learning algorithms from scikit.

    Parameters
    ----------
    algo : a scikit algo
        Algo for response surface, already trained/validated
    in_dim : int
        Input dimension
    out_dim: int
        Output dimension

    Examples
    --------
    import openturns as ot
    from sklearn.ensemble import RandomForestRegressor
    size = 10
    model = ot.SymbolicFunction("x", "(1.0 + sign(x)) * cos(x) - (sign(x) - 1) * sin(2*x)")
    dataX = ot.Uniform().getSample(size)
    dataY = model(dataX)
    algo = RandomForestRegressor()
    algo.fit(dataX, dataY)
    py_func = SklearnPyFunction(algo, 1, 1)
    """
    def __init__(self, algo, in_dim, out_dim):
        super(SklearnPyFunction, self).__init__(in_dim, out_dim)
        check_is_fitted(algo)
        self.algo = algo

    def _exec(self, x):
        X = np.reshape(x, (1, -1))
        return self.algo.predict(X).ravel()

    def _exec_sample(self, x):
        X = np.array(x)
        size = len(X)
        return self.algo.predict(X).reshape(size, self.getOutputDimension())


class SklearnFunction(ot.Function):
    """
    Define an OpenTURNS Function using sklearn algorithms

    Parameters
    ----------
    algo : a scikit algo
        Algo for response surface, already trained/validated
    in_dim : int
        Input dimension
    out_dim: int
        Output dimension

    Examples
    --------
    import openturns as ot
    from sklearn.ensemble import RandomForestRegressor
    size = 10
    model = ot.SymbolicFunction("x", "(1.0 + sign(x)) * cos(x) - (sign(x) - 1) * sin(2*x)")
    dataX = ot.Uniform().getSample(size)
    dataY = model(dataX)
    algo = RandomForestRegressor()
    algo.fit(dataX, dataY)
    f = SklearnFunction(algo, 1, 1)
    """
    def __new__(self, algo, in_dim, out_dim):
        python_function = SklearnPyFunction(algo, in_dim, out_dim)
        return ot.Function(python_function)

