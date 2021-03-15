from preprocessing import get_preprocessed_classification_dataset
from linear_regression import LinearRegressor
from elastic_net_regression import ElasticNetRegressor
from decision_tree_regression import DecisionTreeRegressor


X_train, X_test, y_train, y_test = get_preprocessed_classification_dataset()

estimators = [LinearRegressor(), ElasticNetRegressor(), DecisionTreeRegressor()]
rmse_scores = []

for estimator in estimators:
    estimator.fit(X_train, y_train)
    rmse_scores.append(estimator.rmse_score(X_test, y_test))

names = ['Linear Regressor', 'Elastic Net Regressor', 'Decision Tree Regressor']
for name, rmse_score in zip(names, rmse_scores):
    print(f'{name}: {rmse_score}')
