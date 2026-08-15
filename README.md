# 💰 소득 예측 모델 (Income Prediction) — CatBoost

> **Pattern Recognition** 최종 팀프로젝트 · Ewha Womans University
> 개인의 인구통계·근로 정보로 **연소득 $50K 초과 여부**를 예측하는 이진 분류 프로젝트

![Python](https://img.shields.io/badge/Python-3.10-blue)
![CatBoost](https://img.shields.io/badge/Model-CatBoost-yellow)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-green)
![Metric](https://img.shields.io/badge/Val%20(F1%2BAUC)%2F2-0.8339-orange)

---

## 📖 프로젝트 소개

- **주제**: 성인 인구통계 데이터(UCI Adult 계열) 기반 연소득 `>50K` 여부 이진 분류
- **강좌**: Pattern Recognition (02) — 최종 팀프로젝트
- **핵심 전략**: 단순 feature 폭증 대신, **CatBoost가 단일 행으로 학습 불가능한 신호만 선별 추가**
- **최종 성능**: Validation `(F1 + AUC) / 2 = 0.8339`

---

## 🎯 문제 정의 & 데이터

- **문제 유형**: 이진 분류 (`<=50K` → 0, `>50K` → 1)
- **평가 지표**: `(F1 score + AUC) / 2`
- **데이터 규모**
  - `train` : 39,073 rows × 15 columns
  - `test` : 9,769 rows × 14 columns (타겟 미포함, `id` 컬럼 유지 필수)
- **클래스 분포**: `<=50K` 76.09% vs `>50K` 23.91% → 약 **3:1 불균형**

### 변수 구성

| 구분 | 변수 |
| --- | --- |
| 식별자 | `id` |
| 수치형 | `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week` |
| 범주형 | `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country` |
| 타겟 | `income` (`>50K` 여부) |

---

## 🛠 Tech Stack

### 🧠 Model & ML

- **CatBoost** (Gradient Boosting, 범주형 변수 자체 처리에 강점)
- scikit-learn (KFold, train_test_split, metrics)

### 📊 Data

- pandas / numpy
- openpyxl (Excel 로드)

### ☁️ Environment

- Python 3.10+
- Google Colab (GPU/CPU 런타임)
- Jupyter Notebook

---

## 🔬 접근 방법 (Pipeline)

원본 **13개 입력 변수 → 최종 31개 변수**로 확장. 각 단계는 `fit(train) → transform(test)` 원칙을 일관 적용하여 **데이터 누수(leakage) 방지**.

### 1️⃣ EDA (탐색적 데이터 분석)

- **수치형 상관**: `education_num`(0.335) > `age`(0.230) > `hours_per_week`(0.227) > `capital_gain`(0.223)
- **강한 규칙 발견**: `capital_gain = 99,999`인 206명 → **전원 고소득**
- **범주형 편차**
  - 혼인상태: `Married-civ-spouse` 44.43% vs `Never-married` 4.59%
  - 직업군: `Exec-managerial` 47.59%, `Prof-specialty` 45.18% vs `Priv-house-serv` 1.49%
  - 성별: 남성 30.35% vs 여성 10.93%
- **결론**: 선형 관계보다 **특정 조건 충족 시 클래스가 명확히 갈리는 비선형 패턴** → 트리 기반 모델 적합

### 2️⃣ 데이터 전처리 & 피처 엔지니어링

- **클리닝**: 결측 표기(`?`) 정규화, 문자열 `strip()`, **결측을 `Unknown` 카테고리로 보존** (결측 자체가 강한 신호 — workclass 결측 그룹 고소득 비율 9.1% vs 정상 24.8%)
- **Base FE (6개)**

  | 신규 변수 | 정의 | 의도 |
  | --- | --- | --- |
  | `cap_net` | `capital_gain - capital_loss` | 순 자본 활동 |
  | `has_capital_stats` | 자본 활동 유무 (binary) | sparse 변수 보완 (91.8%가 0) |
  | `gain_per_hour` | `capital_gain / (hours+1)` | 시간 대비 자본 효율 |
  | `age_bins` | 생애주기 구간화 | 연령-소득 비선형 반영 |
  | `sex_marital` | 성별 × 결혼상태 | 강한 상호작용 |
  | `edu_occ` | 학력 × 직업 | 전문직 구분 강화 |

- **Target Encoding — KFold OOF (5개)**: `occupation`, `native_country`, `edu_occ`, `sex_marital`, `workclass`
  - 5-fold **Out-Of-Fold** 인코딩으로 누수 방지, `smoothing=10`으로 소표본 카테고리 shrinkage
  - 검증: train/test 인코딩 분포 평균 차이 **0.003 이내** → 누수 없음
- **Group Aggregation (7개)**: 그룹별 평균을 row-level feature로 추가 (CatBoost가 학습 불가한 cross-row 통계)
  - `occupation` × [`age`, `education_num`, `hours_per_week`]
  - `native_country` × [`education_num`, `hours_per_week`]
  - `workclass` × [`age`, `hours_per_week`]

- **의도적으로 배제한 기법**: StandardScaler·IsolationForest·Winsorize/log1p·단순 binary 파생 변수
  → 트리 모델 특성상 **정보 이득 0 또는 노이즈**로 확인되어 제외

### 3️⃣ 모델링 — CatBoost

```python
CatBoostClassifier(
    iterations=2500, learning_rate=0.02, depth=7,
    l2_leaf_reg=10, bagging_temperature=0.8, random_strength=1,
    border_count=254, early_stopping_rounds=200, random_seed=42
)
```

- 범주형 변수를 `cat_features`로 직접 전달 → CatBoost 내부 인코딩 활용
- 작은 learning rate + 많은 iteration + early stopping으로 점진 학습 & 과적합 억제
- `stratify=y` holdout (Train 31,258 / Val 7,815)로 불균형 비율 보존

### 4️⃣ Threshold 최적화

- 기본 0.5 대신, `(F1 + AUC) / 2` 기준으로 **0.1~0.8 정밀 탐색**
- **최적 threshold = 0.4220** 선택 후 전체 train 재학습 → `prediction.csv` 생성

---

## 📊 결과

| 지표 | 값 |
| --- | --- |
| Validation AUC | **0.9306** |
| Validation F1 | **0.7373** |
| **최종 지표 `(F1+AUC)/2`** | **0.8339** |
| 최적 Threshold | 0.4220 |

- **Test 예측 분포**: `0` (≤50K) 7,474개 / `1` (>50K) 2,295개 → 특정 클래스 편향 없는 안정적 예측
- **핵심 인사이트**: CatBoost가 자체 학습 불가능한 **cross-fold · cross-row 통계만이 실질적 성능 향상에 기여**

### 📁 제출물 (`prediction.csv`)

| 컬럼 | 설명 |
| --- | --- |
| `id` | 샘플 식별자 |
| `y_cls` | 예측 클래스 (0: ≤50K / 1: >50K) |
| `y_prob` | 클래스 1(>50K) 예측 확률 |

---

## 📂 프로젝트 구조

```
income-prediction-catboost/
├── README.md            # 프로젝트 문서 (본 파일)
├── code.ipynb           # 학습 · 예측 전체 파이프라인
├── prediction.csv       # 테스트셋 최종 예측 결과
├── report.pdf           # 최종 보고서
├── requirements.txt     # 의존성 목록
├── data/
│   └── README.md        # 데이터셋 안내 (원본 미포함)
└── _posts/
    └── 2026-05-31-income-prediction-with-catboost.md   # 깃허브 블로그 포스트
```

---

## 👥 팀원 및 역할

> Team 16 · Pattern Recognition (02)

| 팀원 | 담당 역할 |

| 유다연 | < 전처리, 피처 엔지니어링, 하이퍼파라미터 튜닝, Threshold 최적화, 보고서 작성 > 
| 조수현 | < 전처리, EDA, 모델링, 보고서 작성 > |
| 김유림 | < 모델링, 보고서 작성 > |
| 이현민 | < 전처리, 모델링 , 하이퍼파라미터 튜닝 , 보고서 작성 > |

---

## 🚀 실행 방법

```bash
# 1. 저장소 클론
git clone https://github.com/<your-username>/income-prediction-catboost.git
cd income-prediction-catboost

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 데이터 배치
#    data/ 폴더에 train.xlsx, test.xlsx 배치 (data/README.md 참고)

# 4. 노트북 실행
jupyter notebook code.ipynb
```

> ⚠️ `code.ipynb`는 Google Colab 기준으로 작성되어 있습니다.
> 로컬 실행 시 상단의 `drive.mount(...)` 및 `TRAIN_PATH`/`TEST_PATH` 경로를 로컬 경로로 수정하세요.

---

## 📄 산출물

- 📓 [`code.ipynb`](./code.ipynb) — 재현 가능한 전체 파이프라인
- 📈 [`prediction.csv`](./prediction.csv) — 테스트셋 예측 결과
- 📑 [`report.pdf`](./report.pdf) — 최종 보고서

---

<div align="center">

**Pattern Recognition · Team 16 · 2026**

</div>
