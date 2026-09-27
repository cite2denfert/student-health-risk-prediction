# Predicting Student Health Risk

웨어러블 기반 생활습관 데이터로 학생의 건강 위험도(`at-risk` / `unhealthy` / `fit`)를
사전에 예측하고, 보건·체육·영양 교사와 가정에 예방적으로 연결하는 데이터 파이프라인 &
모델링 프로젝트입니다.

> **Kaggle Playground Series — Season 6, Episode 7**
> https://www.kaggle.com/competitions/playground-series-s6e7

---

## 1. 문제 정의

학교의 학생 건강 관리는 **연 1회 건강검진 + 자기보고 설문**에 의존하는 사후 대응 체계인
경우가 많습니다. 문제가 이미 터진 뒤에야 개입하고, 회복(수면)·활동·섭식 데이터를 따로
관리하며, 정작 가정에서는 학교가 무엇을 관찰하고 있는지 알기 어렵습니다.

**SCQA**

| | 내용 |
|---|---|
| **S**ituation | 사후 대응 체계로는 예방이 안 되고, 축(회복/활동/섭식)별로 데이터가 따로 관리된다. |
| **C**omplication | 데이터가 쌓여도 85.9%가 "주의군(at-risk)"으로 뭉뚱그려져 실제 개입 지점이 안 보인다. |
| **Q**uestion | 증상이 나타나기 전에 "누가, 무엇 때문에, 어떤 도움이 필요한지" 알 수 있는가? |
| **A**nswer | 회복·활동·섭식 3축의 파생변수로 위험 근거를 분해하고, 담당 교사에게 자동 배정 + 가정에 정기 알림을 보내는 예방적 웰니스 거버넌스를 구축한다. |

**3축 설계 (MECE)**

| 축 | 원본 변수 | 담당 |
|---|---|---|
| 회복(Recovery) | `sleep_duration`, `sleep_quality`, `heart_rate`, `stress_level` | 보건교사 |
| 활동(Activity) | `step_count`, `exercise_duration`, `calorie_expenditure`, `physical_activity_level` | 체육교사 |
| 섭식/체성분(Nutrition) | `water_intake`, `diet_type`, `bmi`, `smoking_alcohol` | 영양교사 |

---

## 2. 데이터

- **train.csv**: 690,088행 (정답 있음, `health_condition`)
- **test.csv**: 295,753행 (정답 없음, Kaggle 제출용)
- **타깃 분포**: `at-risk` 85.9% / `unhealthy` 8.4% / `fit` 5.8% — 심한 클래스 불균형
- **검증 전략**: `train_test_split(stratify=target)` 으로 train 80% / valid 20% 분리, 실제 test.csv는 unseen 데이터로 최종 제출에만 사용

데이터는 리포지토리에 포함하지 않았습니다(용량, Kaggle 이용약관). 아래 "실행 방법"을
참고해 직접 내려받아 `data/raw/` 에 넣으세요.

---

## 3. 파생변수 & 통계적 근거

3축 원본 변수만으로는 "위험/정상"을 가르는 신호가 약해, 도메인 지식을 반영한
파생변수를 설계했습니다 (`src/features.py`).

| 파생변수 | 계산식 | 의미 |
|---|---|---|
| `fatigue_index` | stress − sleep | 피로 누적 정도 |
| `activity_index` | exercise + steps/1000 | 신체 활동 수준 |
| `health_habit_score` | sleep + hydration + exercise | 건강 생활 습관 점수 |
| `bmi_step_ratio` | BMI / steps | 체중 대비 활동량 |
| `sleep_x_quality` | sleep × sleep_quality | 수면의 양과 질 반영 |
| `hr_per_bmi`, `cal_per_step`, `cal_per_min` | — | 보조 파생변수 (Top-20 피처 중요도에 등장) |

실제 데이터로 **일원분산분석(ANOVA)** 을 돌려 클래스(at-risk/unhealthy/fit) 간 차이가
통계적으로 유의한지, 효과크기(η²)는 얼마인지 확인했습니다 (`src/eda.py`):

| feature | p-value | η² (효과크기) |
|---|---|---|
| fatigue_index | < 0.001 | **0.358** (매우 큼) |
| sleep_x_quality | < 0.001 | 0.049 |
| activity_index | < 0.001 | 0.045 |
| health_habit_score | < 0.001 | 0.042 |
| bmi_step_ratio | < 0.001 | 0.017 |
| hr_per_bmi | < 0.001 | 0.014 |
| cal_per_step | < 0.001 | 0.014 |
| cal_per_min | < 0.001 | 0.003 |

→ `fatigue_index`(피로도) 파생변수 하나가 세 그룹을 압도적으로 갈라놓는다는 것을
통계적으로 확인했습니다.

---

## 4. 모델링 파이프라인

```
① 데이터 로드
② Train / Valid 분리 (stratify)
③ ColumnTransformer
     · 수치형: 중앙값 대체(Median Imputation) → 표준화(StandardScaler)
     · 범주형: 최빈값 대체(Most Frequent) → 원-핫 인코딩(One-Hot)
④ 모델 학습 (7종 비교)
⑤ Balanced Accuracy 평가
⑥ 최종 모델 선정 → 확률 보정(Calibration)
⑦ Kaggle 제출 파일 생성 (submission.csv)
```

**평가지표로 Balanced Accuracy(클래스별 recall의 평균)를 쓴 이유**: 단순 정확도는 다수
클래스(`at-risk`)만 잘 맞혀도 96%가 나올 수 있어, 소수 클래스(`fit`, `unhealthy`)를 놓치는
문제를 감추기 때문입니다. "학생 한 명 한 명을 놓치지 않기" 위해 모든 클래스를 동등하게
평가합니다.

### 모델 비교 (7종)

`python -m src.train` 실행 결과 (`reports/metrics/model_comparison.csv`, valid set 기준):

| Model | Balanced Accuracy | Accuracy | 학습 시간(초) |
|---|---:|---:|---:|
| **HistGradientBoosting** | **0.9433** | 0.9387 | **4.7** |
| CatBoost | 0.9422 | 0.9407 | 16.1 |
| LightGBM | 0.9401 | 0.9471 | 9.6 |
| Logistic Regression | 0.8917 | 0.8274 | 4.8 |
| XGBoost | 0.8747 | 0.9647 | 11.8 |
| Random Forest | 0.8573 | 0.9651 | 64.8 |
| Extra Trees | 0.8513 | 0.9641 | 48.9 |

HistGradientBoosting과 CatBoost가 Balanced Accuracy 기준 사실상 동률이지만,
HistGradientBoosting의 학습 시간이 약 3배 짧아(4.7s vs 16.1s) **최종 모델로 채택**했습니다.

### 무엇이 점수를 움직였나 (Ablation)

최종 모델(HistGradientBoosting)에서 구성 요소를 하나씩 빼고 valid Balanced Accuracy를 비교했습니다.

| 설정 | Balanced Accuracy | 차이 |
|---|---:|---:|
| 전체 (class_weight + 파생변수) | 0.9424 | — |
| class_weight 제거 | 0.8663 | −0.076 |
| 파생변수 제거 | 0.9093 | −0.033 |

가장 큰 기여는 **클래스 불균형 처리**(`class_weight="balanced"`)였고, 파생변수도 의미 있는
성능 이득을 주면서 동시에 "어떤 교사가 무엇을 해야 하는지" 설명하는 근거가 됩니다.

### 확률 보정(Calibration)의 함정

`CalibratedClassifierCV(method="sigmoid")` 로 보정한 뒤 곧바로 `argmax`로 분류하면,
오히려 **Balanced Accuracy가 0.9424 → 0.8654 로 급락**하는 현상을 발견했습니다.
원인은 보정이 확률을 실제 클래스 비율(85.9/8.4/5.8%)로 되돌리면서, 학습 시
`class_weight="balanced"` 로 얻은 소수 클래스 재현율 이득이 상쇄되기 때문입니다.

이를 해결하기 위해 **사전확률 보정(prior-adjustment)** 을 적용했습니다: 보정된 확률을
학습 시 클래스 비율로 나눈 뒤 다시 argmax를 취합니다 (`src/models.py::PriorAdjustedClassifier`).
`predict_proba()`는 해석용으로 보정된 원본 확률을 그대로 반환하고, `predict()`만 이
보정을 적용해 Balanced Accuracy를 지킵니다.

### 최종 결과 (Valid set, 전체 690,088행으로 재학습)

| | Balanced Accuracy | Accuracy |
|---|---:|---:|
| 보정 전 | 0.9424 | 0.9364 |
| 보정 후 (단순 argmax, ⚠️함정) | 0.8654 | 0.9659 |
| **보정 후 (사전확률 보정)** | **0.9415** | 0.9298 |

### 클래스별 재현율 (혼동행렬, row-normalized)

| True \ Pred | at-risk | fit | unhealthy |
|---|---:|---:|---:|
| at-risk | 0.926 | 0.032 | 0.042 |
| fit | 0.056 | 0.938 | 0.006 |
| unhealthy | 0.037 | 0.003 | 0.960 |

세 클래스 모두 90% 이상의 재현율을 확보해, "다수만 맞추고 소수를 놓치는" 문제를
피했습니다.

### 피처 중요도 (Permutation Importance, Top 10)

| feature | importance |
|---|---:|
| sleep_duration | 0.213 |
| stress_level | 0.213 |
| fatigue_index | 0.141 |
| physical_activity_level | 0.074 |
| bmi | 0.037 |
| smoking_alcohol | 0.022 |
| activity_index | 0.015 |
| exercise_duration | 0.011 |
| sleep_quality | 0.011 |
| step_count | 0.009 |

→ 수면·스트레스·피로도가 가장 크게 기여하며, 곧 보건교사(회복축)가 1차 개입 지점이
되어야 함을 시사합니다.

---

## 5. 프로젝트 구조

```
student-health-risk-prediction/
├── data/
│   ├── raw/              # train.csv, test.csv (직접 다운로드, git에 포함 안 됨)
│   └── processed/
├── src/
│   ├── config.py         # 경로, 타깃, 3축 변수, 파생변수 정의
│   ├── data.py            # 데이터 로드 / train-valid 분리
│   ├── features.py        # 파생변수 생성 (fatigue_index 등)
│   ├── pipeline.py         # ColumnTransformer 전처리 파이프라인
│   ├── models.py           # 7종 후보 모델 + Calibration 보조 클래스
│   ├── train.py             # 학습 파이프라인 (모델 비교 → 선정 → 보정)
│   ├── evaluate.py          # 혼동행렬 / 피처 중요도 리포트
│   ├── predict.py           # Kaggle 제출 파일 생성
│   ├── eda.py                # EDA + ANOVA 통계검정
│   └── inspect_data.py        # 데이터 다운로드 후 컬럼/결측치 확인용
├── notebooks/
│   └── 01_full_pipeline.ipynb  # EDA → 파생변수 → 모델링 → 평가 전체 플로우
├── reports/
│   ├── figures/            # 생성된 그래프 (confusion matrix, feature importance 등)
│   ├── metrics/             # model_comparison.csv, final_metrics.json, anova 결과
│   └── submission.csv        # Kaggle 제출 파일 (예시 실행 결과)
├── models/                  # 학습된 모델(.joblib), git에는 포함 안 됨
├── requirements.txt
└── README.md
```

---

## 6. 실행 방법

```bash
# 1) 환경 설정
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Kaggle 데이터 다운로드 (Kaggle API 키 필요: ~/.kaggle/kaggle.json)
kaggle competitions download -c playground-series-s6e7 -p data/raw
unzip -o data/raw/playground-series-s6e7.zip -d data/raw

# 3) 데이터 구조 확인 (컬럼명이 예상과 다르면 src/config.py 를 맞춰 수정)
python -m src.inspect_data

# 4) EDA 리포트 생성
python -m src.eda

# 5) 학습 (7종 모델 비교 → 최종 모델 선정 → 확률 보정)
python -m src.train

# 6) 평가 리포트 (혼동행렬, 피처 중요도)
python -m src.evaluate

# 7) Kaggle 제출 파일 생성
python -m src.predict
```

또는 `notebooks/01_full_pipeline.ipynb` 를 열어 위 과정을 순서대로 셀 실행하면 됩니다.

---

## 7. 액션 플랜 (제품/운영 관점)

모델은 그 자체로 끝이 아니라, **"누가 무엇을 해야 하는가"** 로 번역되어야 합니다.

| 위험도 | 비중 | 대응 |
|---|---|---|
| 경증(주의) | ~80% | 앱 푸시 알림 + 자동 추천 루틴 |
| 중등도(위험) | ~15% | 담당 교사(보건/체육/영양) 대면 상담 10분 |
| 중증(긴급) | ~5% | 학부모 및 119/병원 즉각 연계 |

- **보건교사**: 피로도 상위 5% → 수면 상담
- **체육교사**: 활동량 하위 10% → 걷기 소그룹 매칭
- **영양교사**: 습관점수 하위 → 영양/급식 맞춤 상담
- **가정 연동**: 월 1회 정기 알림 — 등급 라벨이나 타 학생과의 비교는 낙인 방지를 위해 제외

---

## 8. 한계 및 다음 단계

- 데이터에 존재하지 않는 일부 원천 컬럼(예: `screen_time`)은 사용 가능한 컬럼으로
  근사해 파생변수를 계산했습니다 (`src/features.py` 주석 참고).
- 다음 단계로 Optuna 등을 이용한 하이퍼파라미터 튜닝, SHAP 기반 설명력 강화,
  실제 대시보드(교사별 뷰) 프로토타입 제작을 고려할 수 있습니다.

## License

MIT License — [LICENSE](LICENSE) 참고. 데이터셋은 Kaggle의 이용약관을 따릅니다.
