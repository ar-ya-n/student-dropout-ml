import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path):
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    return df


def preprocess_data(df):
    if 'Student_ID' in df.columns:
        df = df.drop('Student_ID', axis=1)    #Droping useless column ("Student_ID")



    X = df.drop('Dropout', axis=1)    # Spliting features and target
    y = df['Dropout']


    X_train, X_test, y_train, y_test = train_test_split(          # TRAIN-TEST SPLIT (AVOID LEAKAGE)
        X, y, test_size=0.2, random_state=42
    )

    encoders = {}
    for col in X_train.select_dtypes(include='object').columns:        # ENCODE CATEGORICAL FEATURES
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        encoders[col] = le


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)          # FEATURE SCALING
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, encoders


