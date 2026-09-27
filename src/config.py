"""
프로젝트 전역 설정
----------------
경로, 타깃 컬럼, 축(axis)별 원본 변수, 파생변수 정의를 한 곳에서 관리합니다.

주의: RAW_NUMERIC_COLS / RAW_CATEGORICAL_COLS 는 Kaggle Playground S6E7
(https://www.kaggle.com/competitions/playground-series-s6e7) 의 실제 컬럼명을
data/raw/train.csv 를 내려받은 뒤 `python -m src.inspect_data` 로 확인하고
맞춰야 합니다.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

TRAIN_PATH = DATA_RAW_DIR / "train.csv"
TEST_PATH = DATA_RAW_DIR / "test.csv"
SUBMISSION_PATH = REPORTS_DIR / "submission.csv"

# ---------------------------------------------------------------------------
# 타깃
# ---------------------------------------------------------------------------
ID_COL = "id"
TARGET_COL = "health_condition"
TARGET_CLASSES = ["at-risk", "unhealthy", "fit"]  # 관측된 비율: 85.9% / 8.4% / 5.8%

RANDOM_STATE = 42
VALID_SIZE = 0.2  # train 80% / valid 20%, stratify=TARGET_COL

# ---------------------------------------------------------------------------
# 3축 원본 변수
# ---------------------------------------------------------------------------
RECOVERY_AXIS = ["sleep_duration", "sleep_quality", "heart_rate", "stress_level"]
ACTIVITY_AXIS = ["step_count", "exercise_duration", "calorie_expenditure", "physical_activity_level"]
NUTRITION_AXIS = ["water_intake", "diet_type", "bmi", "smoking_alcohol"]

RAW_FEATURE_COLS = RECOVERY_AXIS + ACTIVITY_AXIS + NUTRITION_AXIS

# 위 변수 중 수치형 / 범주형 분류 (실제 데이터로 확인 후 조정)
RAW_NUMERIC_COLS = [
    "sleep_duration",
    "heart_rate",
    "step_count",
    "exercise_duration",
    "calorie_expenditure",
    "water_intake",
    "bmi",
]
RAW_CATEGORICAL_COLS = [
    "stress_level",         # low / medium / high
    "sleep_quality",        # poor / average / good
    "physical_activity_level",  # sedentary / moderate / active
    "diet_type",            # veg / non-veg / balanced
    "smoking_alcohol",      # no / occasional / yes
    "gender",                # female / male / other
]

# ---------------------------------------------------------------------------
# 파생변수 (계산식은 src/features.py 참고)
# ---------------------------------------------------------------------------
# Ablation (HistGradientBoosting, valid BA):
#   전체 0.9424 / class_weight 제거 0.8663 (-0.076) / 파생변수 제거 0.9093 (-0.033)
# → 점수에 가장 크게 기여한 건 클래스 가중치이고, 파생변수도 의미 있는 이득을 줍니다.
DERIVED_FEATURE_NAMES = [
    "fatigue_index",
    "activity_index",
    "health_habit_score",
    "bmi_step_ratio",
    "sleep_x_quality",
    "hr_per_bmi",
    "cal_per_step",
    "cal_per_min",
]
