from otsklearn import FunctionalChaos
from sklearn import datasets
from sklearn.model_selection import GridSearchCV

dataset = datasets.load_iris()
parameters={'degree':[2,3,4]}
estimator = FunctionalChaos()
print(estimator.get_params())
clf = GridSearchCV(estimator, parameters, scoring='r2')

clf.fit(dataset.data, dataset.target)
print(clf.best_estimator_)
print(clf.best_params_)
print(clf.best_score_)

dim = dataset.data.shape[1]
X = dataset.data[8,:].reshape(1, dim)
y = dataset.target[8].reshape(1)
print(X)
print(clf.predict(X), dataset.target[8])
print(clf.best_params_)
print(clf.best_score_)
