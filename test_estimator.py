import otsklearn
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
import openturns as ot
from distutils.version import LooseVersion

# load dataset
dataset = datasets.load_diabetes()
X = dataset.data
y = dataset.target
dim = X.shape[1]
distribution = ot.ComposedDistribution([ot.HistogramFactory().build(ot.Sample(X).getMarginal(i)) for i in range(dim)])

# Chaos
estimator = otsklearn.FunctionalChaos(degree=3, distribution=distribution)
estimator.fit(X, y)
X8 = X[8,:].reshape(1, dim)
y8 = y[8].reshape(1)
print('prediction=', estimator.predict(X8), y8)
print('score=', estimator.score(X, y))
print('importance factors', estimator.feature_importances_)

# through grid search
parameters={'degree':[2, 3, 4]}
clf = GridSearchCV(estimator, parameters, scoring='r2')
clf.fit(X, y)
print(clf.best_estimator_)
print(clf.best_params_)
print(clf.best_score_)

# Kriging
estimator = otsklearn.Kriging()
estimator.fit(X, y)
X8 = X[8,:].reshape(1, dim)
y8 = y[8].reshape(1)
print('prediction=', estimator.predict(X8), y8)
print('score=', estimator.score(X, y))

# Tensor
estimator = otsklearn.TensorApproximation(distribution=distribution)
estimator.fit(X, y)
X8 = X[8,:].reshape(1, dim)
y8 = y[8].reshape(1)
print('prediction=', estimator.predict(X8), y8)
print('score=', estimator.score(X, y))

if LooseVersion(ot.__version__) >= '1.13':
    # LM
    estimator = otsklearn.LinearModel()
    estimator.fit(X, y)
    X8 = X[8,:].reshape(1, dim)
    y8 = y[8].reshape(1)
    print('prediction=', estimator.predict(X8), y8)
    print('score=', estimator.score(X, y))

