from preprocessing import get_preprocessed_classification_dataset
from k_nearest_neighbors import KNearesrtNeighbors


X_train, X_test, y_train, y_test = get_preprocessed_classification_dataset()

knn = KNearesrtNeighbors(n_neighbors=5)

knn.fit(X_train, y_train)

print(knn.score(X_test, y_test)) # 0.9931906614785992
