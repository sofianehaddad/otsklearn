import otsklearn
from sklearn import datasets
import openturns as ot
import numpy as np
import pytest
import numpy.testing as npt
from sklearn.linear_model import LinearRegression

@pytest.fixture(scope="session")
def data():
    """Load diabetes dataset."""
    dataset = datasets.load_diabetes()
    X = dataset.data
    y = dataset.target.reshape(-1,1)
    dim = X.shape[1]
    return X, y, dim


def test_linear(data):
    X, y, dim = data

    estimator = LinearRegression()
    estimator.fit(X, y)
    f = otsklearn.SklearnFunction(estimator, dim, 1)
    X8 = X[8, :]
    assert np.ravel(f(X8)) == pytest.approx(158.8, abs=0.1)

