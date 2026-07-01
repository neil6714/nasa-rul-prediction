import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# These sensors have near-zero variance in FD001 and carry no useful signal
# Dropping them is standard practice in CMAPSS literature
DROP_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                "sensor_16", "sensor_18", "sensor_19"]

DROP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"] + DROP_SENSORS

# Sensors we'll actually use
SENSOR_COLS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21"
]

# Clip RUL at this value — early life cycles have identical sensor readings
# regardless of actual RUL, so capping removes noise and is standard practice
RUL_CLIP = 125


def clip_rul(df, clip=RUL_CLIP):
    """
    Clips RUL to a max value. Engines far from failure look identical
    regardless of their true RUL, so capping at 125 is standard in literature.
    """
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=clip)
    return df


def add_rolling_features(df, window=5):
    """
    Adds rolling mean and std for each sensor column, per engine.
    This smooths sensor noise and gives the model trend information.
    """
    df = df.copy()
    for col in SENSOR_COLS:
        df[f"{col}_mean{window}"] = (
            df.groupby("engine_id")[col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"{col}_std{window}"] = (
            df.groupby("engine_id")[col]
            .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        )
    return df


def normalize_features(train, test, feature_cols):
    """
    MinMax scales features using train statistics only (no data leakage).
    """
    scaler = MinMaxScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])
    return train, test, scaler


def get_feature_cols(df):
    """Returns all feature columns (sensors + rolling features, no metadata)."""
    exclude = ["engine_id", "cycle", "RUL"]
    return [c for c in df.columns if c not in exclude]


def build_features(train, test, rolling_window=5, clip=RUL_CLIP):
    """
    Full feature engineering pipeline.
    """
    # Drop low-variance sensors and operating settings
    train = train.drop(columns=DROP_COLS, errors="ignore")
    test = test.drop(columns=DROP_COLS, errors="ignore")

    # Clip RUL
    train = clip_rul(train, clip)
    test = clip_rul(test, clip)

    # Add rolling features
    train = add_rolling_features(train, window=rolling_window)
    test = add_rolling_features(test, window=rolling_window)

    # Get feature columns
    feature_cols = get_feature_cols(train)

    # Normalize
    train, test, scaler = normalize_features(train, test, feature_cols)

    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Train shape after features: {train.shape}")
    print(f"Test shape after features:  {test.shape}")
    print(f"RUL range in train: {train['RUL'].min()} - {train['RUL'].max()}")

    return train, test, feature_cols, scaler


if __name__ == "__main__":
    train = pd.read_csv("../data/train_processed.csv")
    test = pd.read_csv("../data/test_processed.csv")

    train, test, feature_cols, scaler = build_features(train, test)

    train.to_csv("../data/train_featured.csv", index=False)
    test.to_csv("../data/test_featured.csv", index=False)
    print("\nFeature files saved to data/")