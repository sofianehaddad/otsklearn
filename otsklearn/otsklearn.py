import openturns as ot
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np



def BuildDistribution(X):
    #return ot.FunctionalChaosAlgorithm.BuildDistribution(X)
    input_dimension = X.shape[1]
    marginals = []
    for j in range(input_dimension):
        marginals.append(ot.HistogramFactory().build(X[:,j].reshape(-1, 1)))
    isIndependent = True
    for j in range(input_dimension):
        marginalJ = X[:,j].reshape(-1, 1)
        for i in range(j + 1, input_dimension):
            marginalI = X[:,i].reshape(-1, 1)
            testResult = ot.HypothesisTest.Spearman(marginalI, marginalJ)
            isIndependent = isIndependent and testResult.getBinaryQualityMeasure()
    copula = ot.IndependentCopula(input_dimension)
    if not isIndependent:
        copula = ot.NormalCopulaFactory().build(X)
    distribution = ot.ComposedDistribution(marginals, copula)
    return distribution



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
        enumerate : str, either 'linear' or 'hyperbolic'
            Type of the basis terms domain
        q : float
            Value of the hyperbolic enumerate function shape
        distribution : :py:class:`openturns.Distribution`, default=None
            Distribution of the inputs
            If not provided, the distribution is estimated from the sample
        """
        super(FunctionalChaos, self).__init__()
        self.degree = degree
        self.sparse = sparse
        self.enumerate = enumerate
        self.q = q
        self.distribution = distribution
        self._result = None

    def fit(self, X, y, **fit_params):
        input_dimension = X.shape[1]
        if self.distribution is None:
            self.distribution = BuildDistribution(X)
        if self.enumerate == 'linear':
            enumerateFunction = ot.LinearEnumerateFunction(input_dimension)
        elif self.enumerate == 'hyperbolic':
            enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(input_dimension, self.q)
        else:
            raise ValueError('enumerate should be "linear" or "hyperbolic"')
        polynomials = [ot.StandardDistributionPolynomialFactory(self.distribution.getMarginal(i)) for i in range(input_dimension)]
        productBasis = ot.OrthogonalProductPolynomialFactory(polynomials, enumerateFunction)
        adaptiveStrategy = ot.FixedStrategy(productBasis, enumerateFunction.getStrataCumulatedCardinal(self.degree))
        if self.sparse:
            projectionStrategy = ot.LeastSquaresStrategy(ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut()))
        else:
            projectionStrategy = ot.LeastSquaresStrategy(X, y.reshape(-1, 1))
        algo = ot.FunctionalChaosAlgorithm(X, y.reshape(-1, 1), self.distribution, adaptiveStrategy, projectionStrategy)
        algo.run()
        self._result = algo.getResult()
        output_dimension = self._result.getMetaModel().getOutputDimension()

        # sensitivity
        si = ot.FunctionalChaosSobolIndices(self._result)
        if output_dimension == 1:
            self.feature_importances_ = [si.getSobolIndex(i) for i in range(input_dimension)]
        else:
            self.feature_importances_ = [[0.0] * input_dimension] * output_dimension
            for k in range(output_dimension):
                for i in range(input_dimension):
                    self.feature_importances_[k][i] = si.getSobolIndex(i, k)
        self.feature_importances_ = np.array(self.feature_importances_)
        return self

    def predict(self, X):
        if self._result is None:
            raise RuntimeError('call fit first')
        return self._result.getMetaModel()(X)



class Kriging(BaseEstimator, RegressorMixin):

    def __init__(self, kernel='SquaredExponential', basis='Constant'):
        """
        Kriging estimator.

        Parameters
        ----------
        kernel : str
            Covariance model type
        basis : str
            Basis type
        """
        super(Kriging, self).__init__()
        self.kernel = kernel
        self.basis = basis
        self._result = None

    def fit(self, X, y, **fit_params):
        input_dimension = X.shape[1]
        covarianceModel = eval('ot.' + self.kernel + "("+str(input_dimension)+")")
        basisCollection = eval('ot.'+ self.basis + "BasisFactory("+str(input_dimension)+").build()")
        algo = ot.KrigingAlgorithm(X, y.reshape(-1, 1), covarianceModel, basisCollection)
        algo.run()
        self._result = algo.getResult()
        return self

    def predict(self, X):
        if self._result is None:
            raise RuntimeError('call fit first')
        return self._result.getMetaModel()(X)


class TensorApproximation(BaseEstimator, RegressorMixin):

    def __init__(self, nk=10, max_rank=5, distribution=None):
        """
        Tensor estimator.

        Parameters
        ----------
        nk : int
            Covariance model type
        max_rank : max_rank
            Basis type
        """
        super(TensorApproximation, self).__init__()
        self.nk = nk
        self.max_rank = max_rank
        self.distribution = distribution
        self._result = None

    def fit(self, X, y, **fit_params):
        input_dimension = X.shape[1]
        if self.distribution is None:
            self.distribution = BuildDistribution(X)
        factoryCollection = [ot.OrthogonalUniVariateFunctionFamily(        ot.OrthogonalUniVariatePolynomialFunctionFactory(ot.StandardDistributionPolynomialFactory(self.distribution.getMarginal(i)))) for i in range(input_dimension)]
        functionFactory = ot.OrthogonalProductFunctionFactory(factoryCollection)
        algo = ot.TensorApproximationAlgorithm(X, y.reshape(-1, 1), self.distribution, functionFactory, [self.nk]*input_dimension, self.max_rank)
        algo.run()
        self._result = algo.getResult()
        return self

    def predict(self, X):
        if self._result is None:
            raise RuntimeError('call fit first')
        return self._result.getMetaModel()(X)

class LinearModel(BaseEstimator, RegressorMixin):

    def __init__(self):
        """
        Linear model estimator.

        """
        super(LinearModel, self).__init__()
        self._result = None

    def fit(self, X, y, **fit_params):
        input_dimension = X.shape[1]
        algo = ot.LinearModelAlgorithm(X, y.reshape(-1, 1))
        algo.run()
        self._result = algo.getResult()
        return self

    def predict(self, X):
        if self._result is None:
            raise RuntimeError('call fit first')
        return self._result.getMetaModel()(X)
