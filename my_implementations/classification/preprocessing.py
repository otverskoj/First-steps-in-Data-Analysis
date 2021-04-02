from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def get_preprocessed_classification_dataset():
    path_to_file = '../../datasets/classification/occupancy_detection_preprocessed.csv'
    dataset = pd.read_csv(path_to_file)

    dataset = dataset.drop(['Unnamed: 0', 'date'], axis=1)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 5].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
