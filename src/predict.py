"""
Kaggle 제출 파일 생성

    python -m src.predict
"""
import joblib
import pandas as pd

from src import config
from src.data import load_test
from src.pipeline import prepare_features


def main():
    model_path = config.MODELS_DIR / "final_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} 가 없습니다. 먼저 `python -m src.train` 을 실행하세요.")

    model = joblib.load(model_path)
    test_df = load_test()
    X_test = prepare_features(test_df)

    preds = model.predict(X_test)

    submission = pd.DataFrame({
        config.ID_COL: test_df[config.ID_COL],
        config.TARGET_COL: preds,
    })
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"제출 파일 저장 완료: {config.SUBMISSION_PATH} ({len(submission)} rows)")
    print(submission[config.TARGET_COL].value_counts(normalize=True))


if __name__ == "__main__":
    main()
