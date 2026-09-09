"""
data/raw/train.csv 를 내려받은 뒤 가장 먼저 실행해서
실제 컬럼명 / dtype / 결측치 / 타깃 분포를 확인하는 스크립트.

    python -m src.inspect_data
"""
import pandas as pd

from src import config


def main() -> None:
    if not config.TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"{config.TRAIN_PATH} 가 없습니다. Kaggle에서 데이터를 내려받아 "
            f"data/raw/ 아래에 train.csv, test.csv 로 저장하세요.\n"
            f"  kaggle competitions download -c playground-series-s6e7 -p data/raw\n"
            f"  unzip -o data/raw/playground-series-s6e7.zip -d data/raw"
        )

    df = pd.read_csv(config.TRAIN_PATH)

    print("=" * 70)
    print(f"shape: {df.shape}")
    print("=" * 70)
    print("\n[dtypes]")
    print(df.dtypes)

    print("\n[결측치 비율 (상위 15)]")
    na_ratio = df.isna().mean().sort_values(ascending=False)
    print((na_ratio[na_ratio > 0] * 100).round(2).head(15).astype(str) + " %")

    if config.TARGET_COL in df.columns:
        print(f"\n[타깃 분포: {config.TARGET_COL}]")
        print(df[config.TARGET_COL].value_counts(normalize=True).round(4) * 100)
    else:
        print(f"\n주의: 타깃 컬럼 '{config.TARGET_COL}' 이 없습니다. "
              f"실제 컬럼명을 확인하고 src/config.py 의 TARGET_COL 을 수정하세요.")

    missing_expected = [c for c in config.RAW_FEATURE_COLS if c not in df.columns]
    if missing_expected:
        print(f"\n주의: config.py 에 정의된 컬럼 중 데이터에 없는 컬럼: {missing_expected}")
        print("실제 컬럼명에 맞춰 src/config.py 의 *_AXIS / RAW_*_COLS 값을 업데이트하세요.")
    else:
        print("\nOK: config.py 의 3축 원본 변수가 모두 데이터에 존재합니다.")


if __name__ == "__main__":
    main()
