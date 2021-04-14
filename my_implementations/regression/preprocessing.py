from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def get_preprocessed_classification_dataset():
    path_to_file = '../../datasets/regression/metro_interstate_traffic_volume_preprocessed.csv'
    dataset = pd.read_csv(path_to_file)

    dataset = dataset.drop(['Unnamed: 0', 'date_time'], axis=1)

    X = dataset.iloc[:, [0, 1, 2, 3] + list(range(5, 65))]
    y = dataset.iloc[:, 4]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
