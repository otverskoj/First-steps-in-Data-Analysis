from preprocessing import get_preprocessed_classification_dataset
from linear_regression import LinearRegressor


X_train, X_test, y_train, y_test = get_preprocessed_classification_dataset()

estimators = [LinearRegressor()]
mse_scores = []

for estimator in estimators:
    estimator.fit(X_train, y_train)
    mse_scores.append(estimator.mse_score(X_test, y_test))

names = ['Linear Regressior', 'Decision Tree Regressor', 'ElasticNet Regressor']
for name, mse_score in zip(names, mse_scores):
    print(f'{name}: {mse_score}')
