import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    """
    Load the Boston Housing dataset from the URL manually.
    Returns a pandas DataFrame with features and target.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

    # Combine the features spread across two rows
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # target variable

    return df


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Splits the data into train/test and applies standard scaling.
    Returns: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns='MEDV')
    y = df['MEDV']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test