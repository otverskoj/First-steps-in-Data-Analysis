from preprocessing import get_preprocessed_classification_dataset
from k_nearest_neighbors import KNearesrtNeighbors
from gaussian_naive_bayes import GaussianNaiveBayes


X_train, X_test, y_train, y_test = get_preprocessed_classification_dataset()

# knn = KNearesrtNeighbors(n_neighbors=5)

# knn.fit(X_train, y_train)

# print(knn.score(X_test, y_test)) # 0.9931906614785992

gnb = GaussianNaiveBayes()

gnb.fit(X_train, y_train)

print(gnb.score(X_test, y_test)) # 0.7689688715953308
