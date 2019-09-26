.. image:: https://travis-ci.org/openturns/otsklearn.svg?branch=master
    :target: https://travis-ci.org/openturns/otsklearn

otsklearn
=========

Simple module to use OT metamodels with the scikit-learn estimator API (fit/predict)
Currently it's more a proof of concept, not ready for production use.

Examples
--------
>>> from otsklearn import FunctionalChaos
>>> from sklearn import datasets
>>> from sklearn.model_selection import GridSearchCV
>>> dataset = datasets.load_iris()
>>> parameters={'degree':[2,3,4]}
>>> estimator = FunctionalChaos()
>>> print(estimator.get_params())
>>> clf = GridSearchCV(estimator, parameters, scoring='r2')
>>> clf.fit(dataset.data, dataset.target.reshape(-1,1))
>>> print(clf.best_estimator_)

