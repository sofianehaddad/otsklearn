import openturns as ot
from sklearn.base import BaseEstimator, RegressorMixin

class FunctionalChaos(BaseEstimator, RegressorMixin):

    def __init__(self, degree=2, sparse=False, enumerate='linear', q=0.4, distribution=None):
        """
        Functional chaos estimator.

        Parameters
        ----------
        degree : int
            maximum degree
        sparse : bool, optional, default=False
            Whether to use sparse approximation using LARS that prevents overfitting
        enumerate : str, either linear or hyperbolic
            Type of the basis terms domain
        q : float
            Value of the hyperbolic enumerate function shape
        """
        super(FunctionalChaos, self).__init__()
        self.degree = degree
        self.sparse = sparse
        self.enumerate = enumerate
        self.q = q
        self.distribution = distribution
        self.result = None

    def fit(self, X, y, **fit_params):
        dimension = X.shape[1]
        if self.distribution is None:
            distribution = ot.FunctionalChaosAlgorithm.BuildDistribution(X)

        if self.enumerate == 'linear':
            enumerateFunction = ot.LinearEnumerateFunction(dimension)
        elif self.enumerate == 'hyperbolic':
            enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(dimension, self.q)
        else:
            raise ValueError('enumerate should be "linear" or "hyperbolic"')
        polynomials = [ot.StandardDistributionPolynomialFactory(distribution.getMarginal(i)) for i in range(dimension)]
        productBasis = ot.OrthogonalProductPolynomialFactory(polynomials, enumerateFunction)
        adaptiveStrategy = ot.FixedStrategy(productBasis, enumerateFunction.getStrataCumulatedCardinal(self.degree))
        if self.sparse:
            projectionStrategy = ot.LeastSquaresStrategy(ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut()))
        else:
            projectionStrategy = ot.LeastSquaresStrategy(X, y.reshape(-1, 1))
        algo = ot.FunctionalChaosAlgorithm(X, y.reshape(-1, 1), distribution, adaptiveStrategy, projectionStrategy)
        algo.run()
        self.result_ = algo.getResult()
        return self

    def predict(self, X):
        if self.result_ is None:
            raise RuntimeError('call fit first')

        return self.result_.getMetaModel()(X)
