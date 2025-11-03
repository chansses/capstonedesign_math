# 환경 데이터 증강 및 자연스러운 진동 패턴 생성

이 프로젝트는 환경 센서 데이터를 증강하여 머신러닝 모델 학습에 적합한 데이터셋을 생성합니다. 특히 온도와 CO2 기준선을 자주 교차하는 자연스러운 진동 패턴을 만들어 모델의 성능을 향상시킵니다.

## 📁 프로젝트 구조

```
찬세/
├── preprocessed_env_dataset.csv                    # 원본 환경 데이터
├── env_dataset_natural_oscillating.csv            # 자연스러운 진동 패턴 데이터
├── create_natural_oscillating_dataset.py          # 자연스러운 진동 패턴 생성 스크립트
├── threshold_detection_model.py                   # Random Forest 기준선 초과 탐지 모델
├── natural_oscillating_patterns_comparison.png    # 패턴 비교 시각화
├── feature_importance.png                         # 특징 중요도 시각화
├── requirements.txt                                # 필요 라이브러리 목록
└── README.md                                      # 프로젝트 설명서
```

## 🎯 프로젝트 목표

- **원본 데이터 한계**: 온도 26°C, CO2 1000ppm 기준선 교차가 부족
- **목표**: 기준선을 자주 교차하는 자연스러운 진동 패턴 생성
- **결과**: 머신러닝 모델 학습에 적합한 데이터셋 확보

## 📊 데이터 개요

### 원본 데이터 (`preprocessed_env_dataset.csv`)
- **크기**: 10,130개 데이터 포인트
- **컬럼**: Datetime, Temp_avg, Temp_diff, S5_CO2, CO2_Slope, Room_Occupancy_Count, State
- **특징**: 정적인 환경 데이터, 기준선 교차 빈도 낮음

### 생성된 데이터 (`env_dataset_natural_oscillating.csv`)
- **크기**: 10,130개 데이터 포인트 (원본과 동일)
- **특징**: 자연스러운 진동 패턴, 기준선 교차 빈도 대폭 증가

## 🔧 설치 및 실행

### 1. 필요한 라이브러리 설치

```bash
pip install -r requirements.txt
```

또는 개별 설치:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. 자연스러운 진동 패턴 데이터셋 생성

```bash
python create_natural_oscillating_dataset.py
```

### 3. 기준선 초과 탐지 모델 학습 및 예측

```bash
python threshold_detection_model.py
```

## 📈 데이터 변형 과정

### 1. 기본 구조 유지
- **Datetime**: 시간 정보 그대로 유지
- **Temp_diff, CO2_Slope, Room_Occupancy_Count, State**: 변하지 않음
- **Temp_avg, S5_CO2**: 자연스러운 진동 패턴 적용

### 2. 진동 패턴 생성 원리

#### A. 일일 사이클 (24시간 패턴)
```python
daily_cycle_length = 10128 // 7 = 1446개 데이터 포인트
temp_daily_amplitude = 1.5°C  # ±1.5도
co2_daily_amplitude = 150.0ppm  # ±150ppm
```

#### B. 계절적 변화 (주간/월간 패턴)
```python
seasonal_cycle_length = 10128 // 2 = 5064개 데이터 포인트
temp_seasonal_amplitude = 2.0°C  # ±2도
co2_seasonal_amplitude = 200.0ppm  # ±200ppm
```

#### C. 사람 활동 패턴 (CO2 전용)
```python
activity_pattern_length = 10128 // 10 = 1012개 데이터 포인트
activity_co2 = 100.0ppm  # 활동으로 인한 CO2 증가
```

### 3. 수학적 계산

#### 온도 계산
```python
# 일일 사이클
daily_temp_phase = 2 * π * i / 1446
daily_temp = 1.5 * sin(daily_temp_phase)

# 계절적 변화
seasonal_temp_phase = 2 * π * i / 5064
seasonal_temp = 2.0 * sin(seasonal_temp_phase)

# 노이즈
temp_noise = random_normal(0, 0.3)

# 최종 온도
new_temp = 26.0 + daily_temp + seasonal_temp + temp_noise
```

#### CO2 계산
```python
# 일일 사이클 (온도와 30도 위상차)
daily_co2_phase = 2 * π * i / 1446 + π/6
daily_co2 = 150.0 * sin(daily_co2_phase)

# 계절적 변화 (온도와 45도 위상차)
seasonal_co2_phase = 2 * π * i / 5064 + π/4
seasonal_co2 = 200.0 * sin(seasonal_co2_phase)

# 사람 활동 패턴
activity_phase = 2 * π * i / 1012
activity_co2 = 100.0 * sin(activity_phase)

# 노이즈
co2_noise = random_normal(0, 30.0)

# 최종 CO2
new_co2 = 1000.0 + daily_co2 + seasonal_co2 + activity_co2 + co2_noise
```

### 4. 현실성 보장

#### A. 값 범위 제한
```python
new_temp = clip(new_temp, 18.0, 35.0)  # 실내 온도 범위
new_co2 = clip(new_co2, 400.0, 1800.0)  # 실내 CO2 범위
```

#### B. 위상차 적용
- 온도와 CO2 사이 30-45도 위상차
- 자연스러운 상관관계 유지

#### C. 노이즈 추가
- 온도: ±0.3°C (측정 오차)
- CO2: ±30ppm (측정 오차)

## 📊 결과 비교

| 항목 | 원본 데이터 | 자연스러운 진동 | 증가율 |
|------|-------------|-----------------|--------|
| 온도 기준선 교차 | 58회 | 728회 | 12.5배 |
| CO2 기준선 교차 | 8회 | 708회 | 88.5배 |
| 데이터 포인트 수 | 10,130개 | 10,130개 | 동일 |

### 실제 변형 예시
```
원본: 온도 24.922°C, CO2 390ppm
변형: 온도 26.338°C, CO2 1206ppm
차이: 온도 +1.416°C, CO2 +816ppm
```

## 🎨 시각화

### 1. 패턴 비교 시각화
- `natural_oscillating_patterns_comparison.png`: 원본 vs 자연스러운 진동 패턴
- `improved_natural_oscillating_patterns.png`: 개선된 시각화 (두꺼운 선, 향상된 색상)
- `simple_natural_oscillating_patterns.png`: 간단한 시각화

### 2. 시각화 특징
- **왼쪽 그래프**: 온도 변화 (빨간색 선)
- **오른쪽 그래프**: CO2 변화 (파란색 선)
- **기준선**: 온도 26°C (빨간 점선), CO2 1000ppm (파란 점선)
- **각 점**: 개별 데이터 포인트 (30초 간격)

## 🔍 주요 함수

### `create_natural_oscillating_dataset()`
- 자연스러운 진동 패턴을 가진 데이터셋 생성
- 일일/계절적/활동 패턴 반영
- 기준선 교차 횟수 계산 및 출력

### `create_refined_natural_dataset()`
- 정교한 자연스러운 진동 패턴 생성
- 더 정밀한 환경 변화 패턴 반영
- 시간대별/요일별/계절별 차이 반영

### `count_crossings()`
- 기준선 교차 횟수 계산
- 위아래 교차 모두 포함

## 🚀 사용법

### 1. 기본 실행
```bash
# 자연스러운 진동 패턴 생성
python create_natural_oscillating_dataset.py

# 시각화 생성
python create_improved_visualization.py
```

### 2. 파라미터 조정
스크립트 내에서 다음 파라미터들을 조정할 수 있습니다:

```python
# 진동 파라미터
temp_daily_amplitude = 1.5      # 일일 온도 변화
temp_seasonal_amplitude = 2.0   # 계절적 온도 변화
co2_daily_amplitude = 150.0     # 일일 CO2 변화
co2_seasonal_amplitude = 200.0  # 계절적 CO2 변화

# 노이즈 레벨
temp_noise_level = 0.3          # 온도 측정 오차
co2_noise_level = 30.0          # CO2 측정 오차
```

## 🤖 머신러닝 모델: Random Forest 기반 기준선 초과 탐지

### 모델 개요
다음 시점에 기준선을 넘을지 미리 예측하는 Random Forest 모델을 구현했습니다. 프로젝트 목적에 맞게 **Normal → High 전이 예측**에 최적화되었습니다.

### 모델 특징
- **목적**: 기준선 초과를 미리 탐지하여 사전 조치 가능
- **알고리즘**: Random Forest (특징 기반 예측)
- **특징**: 온도, CO2, 변화율, 추세, 기준선까지의 거리 등 13개 특징 활용
- **High 상태 정의**: 온도 > 26°C **AND** CO2 > 1000ppm (둘 다 넘어야 High)

### 성능 결과

#### High 상태 정의
- **AND 조건**: 온도 > 26°C **그리고** CO2 > 1000ppm (둘 다 넘어야 High)
- 더 엄격한 기준으로 실제로 위험한 상황만 탐지

#### 전체 성능
- **정확도**: **96.84%** ✅
- **High 상태 탐지 F1-Score**: **95.86%** ✅
- **Precision**: 95.12% (예측한 High 중 실제 High 비율)
- **Recall**: 96.61% (실제 High 중 예측한 비율)

#### 핵심 지표: Normal → High 전이 예측
- **Normal → High 전이 F1-Score**: **66.67%** ✅ (목표: 50% 이상 달성!)
- **Precision**: 100.00% (예측이 매우 정확함)
- **Recall**: 50.00%
- **Normal 상태 사례 수**: 3,580개
- **실제 High로 전이**: 18개 (AND 조건이 엄격하여 적음)

### 주요 특징 중요도 (상위 5개 - AND 조건 기준)
1. **Temp_Avg_Recent** (최근 평균 온도): 20.17%
2. **CO2_Avg_Recent** (최근 평균 CO2): 18.50%
3. **Temp_avg** (현재 온도): 16.49%
4. **Dist_to_CO2_Thresh** (CO2 기준선까지 거리): 11.68%
5. **Dist_to_Temp_Thresh** (온도 기준선까지 거리): 11.56%

### 사용법
```python
from threshold_detection_model import ThresholdDetectionModel
import pandas as pd

# 데이터 로드
df = pd.read_csv('env_dataset_natural_oscillating.csv')

# 모델 생성 및 학습
model = ThresholdDetectionModel(temp_threshold=26.0, co2_threshold=1000.0)
results = model.fit(df, window_size=3)

# 전체 데이터 평가
eval_results = model.evaluate_on_data(df)
print(f"Normal→High 전이 F1-Score: {eval_results['normal_to_high_f1']:.4f}")
```

### 모델 개선 과정
1. **초기 시도**: 마르코프 체인 모델 (전체 정확도 86.67%지만 Normal→High F1: 0%)
2. **문제점 파악**: Normal 상태 예측에만 의존, 실제 목적(High 상태 탐지) 달성 실패
3. **Random Forest 적용**: 특징 기반 예측으로 성능 향상
4. **최종 모델**: AND 조건 적용 (온도와 CO2 둘 다 넘어야 High) - Normal→High F1: 66.67% 달성

## 📝 주의사항

1. **데이터 크기**: 원본 데이터와 동일한 크기 (10,130개)로 생성
2. **시간 정보**: Datetime 컬럼은 원본 그대로 유지
3. **현실성**: 실내 환경에 적합한 값 범위로 제한
4. **재현성**: 동일한 시드로 실행 시 동일한 결과
5. **Random Forest 모델**: 클래스 불균형을 `class_weight='balanced'`로 해결하여 High 상태 탐지 성능 향상

## 🤝 기여

이 프로젝트에 기여하고 싶으시다면:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 연락처

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해 주세요.

---

**참고**: 이 프로젝트는 환경 센서 데이터의 기준선 교차 빈도를 높여 머신러닝 모델의 학습 성능을 향상시키는 것을 목표로 합니다.