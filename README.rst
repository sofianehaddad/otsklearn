otsklearn
=========

Simple module to use OT metamodels (Functional Chaos for now) with a simple scikit-learn API (kwargs, fit/predict)
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
>>> clf.fit(dataset.data, dataset.target)
>>> print(clf.best_estimator_)

