# 유니폼 가격 예측 모델

레사모 거래 데이터를 기반으로 축구 유니폼의 예상 거래 가격을 예측하는 웹앱입니다.
유니폼 코드와 사이즈, 등급, 마킹·패치 여부 등의 정보를 입력하면 머신러닝 모델이 예상 거래 가격과 거래 범위를 제공합니다.

## 배포 사이트

https://re4mo-price-prediction.streamlit.app/

## 주요 기능

* 4mation 거래 API 기반 실시간 거래 데이터 수집

* 거래 데이터 자동 전처리 및 이상치 제거

* 거래 경과일(days_ago) feature engineering 적용

* 5개 모델 성능 비교 후 자동 최적 모델 선택

  * RandomForestRegressor
  * GradientBoostingRegressor
  * Ridge
  * KNeighborsRegressor
  * TensorFlow Deep Learning

* MAE 기반 예상 거래 범위 출력

* Streamlit 기반 웹 배포

## Tech Stack

* Python
* Streamlit
* scikit-learn
* TensorFlow / Keras
* Pandas / NumPy

## 프로젝트 목적

실제 유니폼 거래 데이터를 활용해
“현재 판매 중인 매물이 적정 가격인지”를 직관적으로 판단할 수 있는 서비스를 만드는 것을 목표로 했습니다.
