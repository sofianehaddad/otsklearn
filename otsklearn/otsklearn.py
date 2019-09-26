import openturns as ot
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


def BuildDistribution(X, level=0.01):
    # return ot.FunctionalChaosAlgorithm.BuildDistribution(X)
    input_dimension = len(X[1])
    marginals = []
    for j in range(input_dimension):
        marginals.append(ot.HistogramFactory().build(X[:, j:j+1]))
    isIndependent = True
    for j in range(input_dimension):
        marginalJ = X[:, j:j+1]
        for i in range(j + 1, input_dimension):
            marginalI = X[:, i:i+1]
            testResult = ot.HypothesisTest.Spearman(marginalI, marginalJ, level)
            isIndependent = isIndependent and testResult.getBinaryQualityMeasure()
    copula = ot.IndependentCopula(input_dimension)
    if not isIndependent:
        copula = ot.NormalCopulaFactory().build(X)
    distribution = ot.ComposedDistribution(marginals, copula)
    return distribution


class FunctionalChaos(BaseEstimator, RegressorMixin):

    def __init__(self, degree=2, sparse=False, enumeratef='linear', q=0.4, distribution=None):
        """Functional chaos estimator.

        Parameters
        ----------
        degree : int
            maximum degree
        sparse : bool, optional, default=False
            Whether to use sparse approximation using LARS that prevents overfitting
        enumeratef : str, either 'linear' or 'hyperbolic'
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
        self.enumeratef = enumeratef
        self.q = q
        self.distribution = distribution
        self._result = None

    def fit(self, X, y, **fit_params):
        """Fit PC regression model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data.
        y : array-like, shape = (n_samples, [n_output_dims])
            Target values.

        Returns
        -------
        self : returns an instance of self.

        """
        if len(X) == 0:
            raise ValueError("Can not perform chaos expansion with empty sample")
        # check data type is accurate
        if (len(np.shape(X)) != 2):
            raise ValueError("X has incorrect shape.")
        input_dimension = len(X[1])
        if (len(np.shape(y)) != 2):
            raise ValueError("y has incorrect shape.")
        if self.distribution is None:
            self.distribution = BuildDistribution(X)
        if self.enumeratef == 'linear':
            enumerateFunction = ot.LinearEnumerateFunction(input_dimension)
        elif self.enumeratef == 'hyperbolic':
            enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(
                input_dimension, self.q)
        else:
            raise ValueError('enumeratef should be "linear" or "hyperbolic"')
        polynomials = [ot.StandardDistributionPolynomialFactory(
            self.distribution.getMarginal(i)) for i in range(input_dimension)]
        productBasis = ot.OrthogonalProductPolynomialFactory(
            polynomials, enumerateFunction)
        adaptiveStrategy = ot.FixedStrategy(
            productBasis, enumerateFunction.getStrataCumulatedCardinal(self.degree))
        if self.sparse:
            projectionStrategy = ot.LeastSquaresStrategy(
                ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut()))
        else:
            projectionStrategy = ot.LeastSquaresStrategy(X, y)
        algo = ot.FunctionalChaosAlgorithm(
            X, y, self.distribution, adaptiveStrategy, projectionStrategy)
        algo.run()
        self._result = algo.getResult()
        output_dimension = self._result.getMetaModel().getOutputDimension()

        # sensitivity
        si = ot.FunctionalChaosSobolIndices(self._result)
        if output_dimension == 1:
            self.feature_importances_ = [
                si.getSobolIndex(i) for i in range(input_dimension)]
        else:
            self.feature_importances_ = [[0.0] * input_dimension] * output_dimension
            for k in range(output_dimension):
                for i in range(input_dimension):
                    self.feature_importances_[k][i] = si.getSobolIndex(i, k)
        self.feature_importances_ = np.array(self.feature_importances_)
        return self

    def predict(self, X):
        """Predict using the PC regression model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the model is evaluated.

        Returns
        -------
        y : array, shape = (n_samples, [n_output_dims])
            Predictions at query points.

        """
        if self._result is None:
            raise RuntimeError('call fit first')
        return self._result.getMetaModel()(X)


class Kriging(BaseEstimator, RegressorMixin):

    def __init__(self, kernel='SquaredExponential', basis='Constant'):
        """Kriging estimator.

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
        """Fit Kriging regression model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data.
        y : array-like, shape = (n_samples, [n_output_dims])
            Target values.

        Returns
        -------
        self : returns an instance of self.

        """
        if len(X) == 0:
            raise ValueError("Can not perform a kriging algorithm with empty sample")
        # check data type is accurate
        if (len(np.shape(X)) != 2):
            raise ValueError("X has incorrect shape.")
        input_dimension = len(X[1])
        if (len(np.shape(y)) != 2):
            raise ValueError("y has incorrect shape.")
        covarianceModel = eval('ot.' + self.kernel + "(" + str(input_dimension) + ")")
        basisCollection = eval('ot.' + self.basis +
                               "BasisFactory(" + str(input_dimension) + ").build()")
        algo = ot.KrigingAlgorithm(
            X, y, covarianceModel, basisCollection, True)
        algo.run()
        self._result = algo.getResult()
        return self

    def predict(self, X, return_std=False):
        """Predict using the Kriging regression model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the Kriging is evaluated.
        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution at query points.
        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.

        """
        if self._result is None:
            raise RuntimeError('call fit first')

        y_mean = self._result.getMetaModel()(X)

        if return_std:
            y_std = self._result.getConditionalCovariance(X)
            y_std = ot.Sample([np.sqrt(y_std[i, i])
                               for i in range(y_std.getNbRows())], 1)
            return y_mean, y_std
        else:
            return y_mean


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
        """Fit Tensor regression model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data.
        y : array-like, shape = (n_samples, [n_output_dims])
            Target values.

        Returns
        -------
        self : returns an instance of self.

        """
        if len(X) == 0:
            raise ValueError("Can not perform a tensor approximation with empty sample")
        # check data type is accurate
        if (len(np.shape(X)) != 2):
            raise ValueError("X has incorrect shape.")
        input_dimension = len(X[1])
        if (len(np.shape(y)) != 2):
            raise ValueError("y has incorrect shape.")
        if self.distribution is None:
            self.distribution = BuildDistribution(X)
        factoryCollection = [ot.OrthogonalUniVariateFunctionFamily(ot.OrthogonalUniVariatePolynomialFunctionFactory(
            ot.StandardDistributionPolynomialFactory(self.distribution.getMarginal(i))))
                             for i in range(input_dimension)]
        functionFactory = ot.OrthogonalProductFunctionFactory(factoryCollection)
        algo = ot.TensorApproximationAlgorithm(X, y,
                                               self.distribution,
                                               functionFactory,
                                               [self.nk] * input_dimension,
                                               self.max_rank)
        algo.run()
        self._result = algo.getResult()
        return self

    def predict(self, X):
        """Predict using the Tensor regression model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the model is evaluated.

        Returns
        -------
        y : array, shape = (n_samples, [n_output_dims])
            Predictions at query points.

        """
        if self._result is None:
            raise RuntimeError('call fit first')
        return self._result.getMetaModel()(X)


class LinearModel(BaseEstimator, RegressorMixin):

    def __init__(self):
        """Linear model estimator."""
        super(LinearModel, self).__init__()
        self._result = None

    def fit(self, X, y, **fit_params):
        """Fit Linear regression model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data.
        y : array-like, shape = (n_samples, [n_output_dims])
            Target values.

        Returns
        -------
        self : returns an instance of self.

        """
        algo = ot.LinearModelAlgorithm(X, y)
        algo.run()
        self._result = algo.getResult()
        return self

    def predict(self, X):
        """Predict using the Linear regression model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the model is evaluated.

        Returns
        -------
        y : array, shape = (n_samples, [n_output_dims])
            Predictions at query points.

        """
        if self._result is None:
            raise RuntimeError('call fit first')
        return self._result.getMetaModel()(X)
