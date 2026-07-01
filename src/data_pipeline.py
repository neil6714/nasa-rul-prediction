import pandas as pd
import numpy as np
import os

# Column names for CMAPSS dataset
COLUMN_NAMES = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21"
]

def load_data(data_dir="../data"):
    """
    Loads train, test, and RUL files for FD001.
    """
    train = pd.read_csv(
        os.path.join(data_dir, "train_FD001.txt"),
        sep=r"\s+", header=None, names=COLUMN_NAMES
    )

    test = pd.read_csv(
        os.path.join(data_dir, "test_FD001.txt"),
        sep=r"\s+", header=None, names=COLUMN_NAMES
    )

    rul = pd.read_csv(
        os.path.join(data_dir, "RUL_FD001.txt"),
        sep=r"\s+", header=None, names=["RUL"]
    )

    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    print(f"RUL shape:   {rul.shape}")
    print(f"Unique engines in train: {train['engine_id'].nunique()}")
    print(f"Unique engines in test:  {test['engine_id'].nunique()}")

    return train, test, rul


def add_rul_to_train(train):
    """
    For training data: computes RUL for each row by counting
    cycles remaining until the engine's last recorded cycle (failure).
    """
    max_cycles = train.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    train = train.merge(max_cycles, on="engine_id")
    train["RUL"] = train["max_cycle"] - train["cycle"]
    train.drop(columns=["max_cycle"], inplace=True)
    return train


def add_rul_to_test(test, rul):
    """
    For test data: the RUL file gives the true RUL at the LAST cycle
    of each engine in the test set. We compute RUL for all cycles
    by counting back from that last cycle.
    """
    rul["engine_id"] = rul.index + 1  # engine IDs are 1-indexed
    max_cycles = test.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    test = test.merge(max_cycles, on="engine_id")
    test = test.merge(rul, on="engine_id")
    test["RUL"] = test["max_cycle"] - test["cycle"] + test["RUL"]
    test.drop(columns=["max_cycle"], inplace=True)
    return test


if __name__ == "__main__":
    train, test, rul = load_data()
    train = add_rul_to_train(train)
    test = add_rul_to_test(test, rul)

    print(f"\nTrain sample (with RUL):")
    print(train[["engine_id", "cycle", "RUL"]].head(10))

    print(f"\nTest sample (with RUL):")
    print(test[["engine_id", "cycle", "RUL"]].head(10))

    print(f"\nMax RUL in train: {train['RUL'].max()}")
    print(f"Min RUL in train: {train['RUL'].min()}")

    train.to_csv("../data/train_processed.csv", index=False)
    test.to_csv("../data/test_processed.csv", index=False)
    print("\nProcessed files saved to data/")