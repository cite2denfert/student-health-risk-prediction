"""데이터 로드 & train/valid 분리"""
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_train() -> pd.DataFrame:
    return pd.read_csv(config.TRAIN_PATH)


def load_test() -> pd.DataFrame:
    return pd.read_csv(config.TEST_PATH)


def split_train_valid(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """검증 전략: train 80% / valid 20%, stratify=target."""
    train_df, valid_df = train_test_split(
        df,
        test_size=config.VALID_SIZE,
        stratify=df[config.TARGET_COL],
        random_state=config.RANDOM_STATE,
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)
