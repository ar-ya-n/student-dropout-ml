import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


TARGET_COLUMN = "Dropout"
ID_COLUMNS = ("Student_ID",)


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def _encode_categorical(train_df, test_df):
    categorical_cols = train_df.select_dtypes(include=["object", "category", "bool"]).columns
    train_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=False)
    test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=False)
    train_encoded, test_encoded = train_encoded.align(test_encoded, join="left", axis=1, fill_value=0)
    return train_encoded, test_encoded


def _select_features_by_correlation(X_train_df, y_train, threshold=0.05):
    train_with_target = X_train_df.copy()
    train_with_target[TARGET_COLUMN] = y_train.values

    corr_to_target = train_with_target.corr(numeric_only=True)[TARGET_COLUMN].drop(labels=[TARGET_COLUMN])
    corr_to_target = corr_to_target.abs().sort_values(ascending=False)

    selected_features = corr_to_target[corr_to_target >= threshold].index.tolist()
    if not selected_features:
        selected_features = X_train_df.columns.tolist()

    return selected_features, corr_to_target


def _balance_training_data(X_train_df, y_train, strategy="oversample"):
    if strategy is None:
        return X_train_df, y_train

    train_df = X_train_df.copy()
    train_df[TARGET_COLUMN] = y_train.values

    class_counts = train_df[TARGET_COLUMN].value_counts()
    if len(class_counts) != 2 or class_counts.min() == class_counts.max():
        return X_train_df, y_train

    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    majority_df = train_df[train_df[TARGET_COLUMN] == majority_class]
    minority_df = train_df[train_df[TARGET_COLUMN] == minority_class]

    minority_upsampled = resample(
        minority_df,
        replace=True,
        n_samples=len(majority_df),
        random_state=42,
    )

    balanced_df = pd.concat([majority_df, minority_upsampled], axis=0).sample(frac=1, random_state=42)
    X_balanced = balanced_df.drop(columns=[TARGET_COLUMN])
    y_balanced = balanced_df[TARGET_COLUMN]
    return X_balanced, y_balanced


def preprocess_data(
    df,
    test_size=0.2,
    random_state=42,
    stratify=True,
    correlation_threshold=0.05,
    balance_strategy="oversample",
):
    df = df.copy()
    columns_to_drop = [col for col in ID_COLUMNS if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    X_train_encoded, X_test_encoded = _encode_categorical(X_train, X_test)
    selected_features, correlation_scores = _select_features_by_correlation(
        X_train_encoded, y_train, threshold=correlation_threshold
    )

    X_train_selected = X_train_encoded[selected_features]
    X_test_selected = X_test_encoded[selected_features]

    X_train_balanced, y_train_balanced = _balance_training_data(
        X_train_selected, y_train, strategy=balance_strategy
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test_selected)

    metadata = {
        "selected_features": selected_features,
        "correlation_scores": correlation_scores.to_dict(),
        "class_distribution_before": y_train.value_counts(normalize=True).to_dict(),
        "class_distribution_after": y_train_balanced.value_counts(normalize=True).to_dict(),
        "test_size": test_size,
        "stratified": stratify,
    }

    return X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler, metadata


