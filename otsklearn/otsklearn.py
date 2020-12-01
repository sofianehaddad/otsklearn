import openturns as ot
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np

class FunctionalChaos(BaseEstimator, RegressorMixin):

    def __init__(self, degree=2, sparse=False, enumeratef='linear', q=0.7,
                 sparse_fitting_algorithm="cloo", distribution=None):
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
        sparse_fitting_algorithm : str, either 'cloo' or 'kfolf'
            Type of fitting algorithm that should be used in case of sparse algorithm
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
        self.sparse_fitting_algorithm = sparse_fitting_algorithm

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
            self.distribution = ot.MetaModelAlgorithm.BuildDistribution(X)
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
            # Filter according to the sparse_fitting_algorithm key
            if self.sparse_fitting_algorithm == "cloo":
                fitting_algorithm = ot.CorrectedLeaveOneOut()
            else:
                fitting_algorithm = ot.KFold()
            # Define the correspondinding projection strategy
            projectionStrategy = ot.LeastSquaresStrategy(
                ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), fitting_algorithm))
        else:
            projectionStrategy = ot.LeastSquaresStrategy(X, y)
        algo = ot.FunctionalChaosAlgorithm(
            X, y, self.distribution, adaptiveStrategy, projectionStrategy)
        algo.run()
        self.result_ = algo.getResult()
        output_dimension = self.result_.getMetaModel().getOutputDimension()

        # sensitivity
        si = ot.FunctionalChaosSobolIndices(self.result_)
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
        check_is_fitted(self)
        return np.array(self.result_.getMetaModel()(X))


class Kriging(BaseEstimator, RegressorMixin):

    def __init__(self, kernel='SquaredExponential', basis='Constant',
                 n_iter_opt = 100, normalize_data = True, linalg_meth = "LAPACK"):
        """Kriging estimator.

        Parameters
        ----------
        kernel : str or :py:class:`openturns.CovarianceModel`
            Covariance model type
        basis : str or :py:class:`openturns.Basis`
            Basis type
        n_iter_opt : int
            Maximal number of optimization iterations
        normalize_data : bool
            Tells whether input data should be normalized or not.
        linalg_meth : str
            Select the linear algebra
            Values are LAPACK or HMAT
        """
        super(Kriging, self).__init__()
        self.kernel = kernel
        self.basis = basis
        self.n_iter_opt = n_iter_opt
        self.normalize_data = normalize_data
        self.linalg_meth = linalg_meth

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

        if type(self.kernel) is str:
            covarianceModel = eval('ot.' + self.kernel + "(" + str(input_dimension) + ")")
        else :
            covarianceModel = ot.CovarianceModel(self.kernel)
        if type(self.basis) is str:
            basisCollection = eval('ot.' + self.basis +
                                   "BasisFactory(" + str(input_dimension) + ").build()")
        else:
            basisCollection = ot.Basis(self.basis)
        ot.ResourceMap.SetAsString(
            "KrigingAlgorithm-LinearAlgebra",  str(self.linalg_meth).upper())
        algo = ot.KrigingAlgorithm(
            X, y, covarianceModel, basisCollection, self.normalize_data)
        if self.n_iter_opt:
            opt_algo = algo.getOptimizationAlgorithm()
            opt_algo.setMaximumIterationNumber(self.n_iter_opt)
            algo.setOptimizationAlgorithm(opt_algo)
        else:
            algo.setOptimizeParameters(False)
        algo.run()
        self.result_ = algo.getResult()
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
        check_is_fitted(self)
        y_mean = np.array(self.result_.getMetaModel()(X))

        if return_std:
            # Do not perfom conditional covariance on sample as it is compute
            # a full covariance matrix & we focus only on diagonal
            # TODO update using new API (getConditionalVariance)
            y_std = np.array([self.result_.getConditionalCovariance(x) for x in X])
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
            The size of the basis for each component
        max_rank : int
            The maximum rank
        """
        super(TensorApproximation, self).__init__()
        self.nk = nk
        self.max_rank = max_rank
        self.distribution = distribution

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
        self.result_ = algo.getResult()
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
        check_is_fitted(self)
        return np.array(self.result_.getMetaModel()(X))


class LinearModel(BaseEstimator, RegressorMixin):

    def __init__(self):
        """Linear model estimator."""
        super(LinearModel, self).__init__()

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
        self.result_ = algo.getResult()
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
        check_is_fitted(self)
        return np.array(self.result_.getMetaModel()(X))
