from preprocessing import get_preprocessed_classification_dataset
from k_nearest_neighbors import KNearesrtNeighbors
from gaussian_naive_bayes import GaussianNaiveBayes
from SVM import LinearSVM


X_train, X_test, y_train, y_test = get_preprocessed_classification_dataset()


#    !!!
#    Внимание! kNN работает ДОЛГО, минимум 5 минут
#    !!! 

estimators = [KNearesrtNeighbors(), GaussianNaiveBayes(), LinearSVM()]
accuracy_scores = []

for estimator in estimators:
    estimator.fit(X_train, y_train)
    accuracy_scores.append(estimator.score(X_test, y_test))

for estimator, accuracy_score in zip(['kNN', 'Gaussian Naive Bayes', 'Linear SVM'], accuracy_scores):
    print(f'{estimator}: {accuracy_score}')

# kNN: 0.9931906614785992
# Gaussian Naive Bayes: 0.7689688715953308
# Linear SVM: 0.9878404669260701
