import otsklearn
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError
import openturns as ot
import numpy as np
from distutils.version import LooseVersion
import pytest
import numpy.testing as npt

def test_missing_fit_chaos():
    with pytest.raises(NotFittedError):
        algo = otsklearn.FunctionalChaos()
        algo.predict([1])

def test_missing_fit_kriging():
    with pytest.raises(NotFittedError):
        algo = otsklearn.Kriging()
        algo.predict([1])

def test_missing_fit_tensor():
    with pytest.raises(NotFittedError):
        algo = otsklearn.TensorApproximation()
        algo.predict([1])

def test_missing_fit_lm():
    with pytest.raises(NotFittedError):
        algo = otsklearn.LinearModel()
        algo.predict([1])
