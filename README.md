# 유니폼 가격 예측 모델

레사모 거래 데이터를 기반으로 축구 유니폼의 예상 거래 가격을 예측하는 웹앱입니다.
유니폼 코드와 사이즈, 등급, 마킹·패치 여부 등의 정보를 입력하면 머신러닝 모델이 예상 거래 가격과 거래 범위를 제공합니다.

## [배포 사이트](https://re4mo-price-prediction.streamlit.app/)


## 주요 기능

* 4mation 거래 API 기반 실시간 거래 데이터 수집
* 거래 데이터 자동 전처리 및 이상치 제거
* 거래경과일(days_ago) feature engineering 적용
* 5개 모델 성능 비교 후 자동 최적 모델 선택

  * RandomForestRegressor
  * GradientBoostingRegressor
  * Ridge
  * KNeighborsRegressor
  * TensorFlow Deep Learning
* MAE 기반 예상 거래 범위 출력
* Streamlit 기반 웹 배포

## 프로젝트 목적

실제 유니폼 거래 데이터를 활용해
“현재 판매 중인 매물이 적정 가격인지”를 직관적으로 판단할 수 있는 서비스를 만드는 것을 목표로 했습니다.

## 모델 개선 과정

처음 모델을 만들었을 때는
사이즈, 등급, 마킹, 패치 여부만 feature로 사용했습니다.

하지만 테스트 과정에서
오히려 새제품보다 사용감이 있는 유니폼의 가격이 더 높게 예측되는 문제가 발생했습니다.

원인을 분석해보니 축구 유니폼 시장은 일반적인 중고 거래와 다르게
시즌이 지나 생산이 종료될수록 희소성이 증가해 가격이 오르는 경우가 많았습니다.

즉 단순 상태 정보만으로는 실제 시세를 설명하기 어려웠고,
“시간”이 가격에 영향을 주는 중요한 feature라고 판단했습니다.

이를 해결하기 위해 거래 시간을 Unix timestamp 기반의
`거래경과일(days_ago)` feature로 변환해 모델에 추가했습니다.

그 결과 최근 거래와 오래된 거래의 가격 차이를 모델이 반영할 수 있게 되었고,
실제 시세와 더 유사한 예측 결과를 얻을 수 있었습니다.

## 모델 선택 과정

유니폼 거래 데이터는 데이터 수가 많지 않고,
feature 수가 제한적이며,
비선형성이 존재하는 tabular 데이터라고 판단했습니다.

따라서 딥러닝뿐 아니라
RandomForest, GradientBoosting,
KNN, Ridge 모델을 함께 비교해
validation MAE 기준으로 가장 성능이 좋은 모델을 자동 선택하도록 구성했습니다.

또한 실제 서비스에서는 단일 가격보다
예상 거래 범위를 함께 제공하는 것이 사용자 경험 측면에서 더 적절하다고 판단해
MAE 기반 가격 범위를 함께 출력하도록 구현했습니다.

## Tech Stack

* Python
* Streamlit
* scikit-learn
* TensorFlow / Keras
* Pandas / NumPy
