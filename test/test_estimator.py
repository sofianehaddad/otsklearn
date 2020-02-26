import otsklearn
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
import openturns as ot
import numpy as np
from distutils.version import LooseVersion
import pytest
import numpy.testing as npt


@pytest.fixture(scope="session")
def data():
    """Load diabetes dataset."""
    dataset = datasets.load_diabetes()
    X = dataset.data
    y = dataset.target.reshape(-1,1)
    dim = X.shape[1]
    distribution = ot.ComposedDistribution([ot.HistogramFactory().build(ot.Sample(X).getMarginal(i))
                                            for i in range(dim)])

    return X, y, dim, distribution


def test_chaos(data):
    X, y, dim, distribution = data

    estimator = otsklearn.FunctionalChaos(degree=3, distribution=distribution)
    estimator.fit(X, y)
    X8 = X[8, :].reshape(1, dim)

    assert np.array(estimator.predict(X8)) == pytest.approx(121.4, abs=0.1)
    assert estimator.score(X, y) == pytest.approx(0.79, abs=0.1)

    npt.assert_almost_equal(estimator.feature_importances_,
                            [1.51e-05, 4.81e-02, 2.45e-05, 1.26e-04, 1.72e-01,
                             1.56e-01, 2.61e-03, 6.23e-05, 7.77e-04, 5.37e-06],
                            decimal=2)

    # through grid search
    parameters = {'degree': [2, 3, 4]}
    clf = GridSearchCV(estimator, parameters, scoring='r2')
    clf.fit(X, y)

    assert clf.best_params_ == {'degree': 2}
    assert clf.best_score_ == pytest.approx(0.40, abs=0.1)


def test_kriging(data):
    X, y, dim, distribution = data

    estimator = otsklearn.Kriging()
    estimator.fit(X, y)
    X8 = X[8, :].reshape(1, dim)

    assert np.array(estimator.predict(X8)) == pytest.approx(110, abs=0.01)
    assert estimator.score(X, y) == 1


def test_tensor(data):
    X, y, dim, distribution = data

    estimator = otsklearn.TensorApproximation(2, 5, distribution=distribution)
    estimator.fit(X, y)
    X8 = X[8, :].reshape(1, dim)

    assert np.array(estimator.predict(X8)) == pytest.approx(145.16, abs=0.1)
    assert estimator.score(X, y) == pytest.approx(0.59, abs=0.1)


@pytest.mark.skipif(LooseVersion(ot.__version__) < '1.13',
                    reason="Requires openturns 1.13 or higher")
def test_linear(data):
    X, y, dim, distribution = data

    estimator = otsklearn.LinearModel()
    estimator.fit(X, y)
    X8 = X[8, :].reshape(1, dim)
    y8 = y[8].reshape(1)
    print('prediction=', estimator.predict(X8), y8)
    print('score=', estimator.score(X, y))

