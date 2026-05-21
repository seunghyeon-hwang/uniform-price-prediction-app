레사모 거래 데이터를 기반으로 축구 유니폼의 예상 거래 가격을 예측하는 웹앱입니다.
사용자가 입력한 유니폼 코드의 거래 데이터를 자동으로 수집하고, 사이즈·등급·마킹·패치 여부 등의 정보를 바탕으로 현재 시세를 예측합니다.

주요 기능
4mation API 기반 거래 데이터 자동 수집
거래 데이터 전처리 및 이상치 제거
거래 경과일(days_ago) feature engineering 적용
여러 머신러닝 모델 성능 비교 후 최적 모델 자동 선택
예측 가격 + 예상 거래 범위 출력
Streamlit 기반 웹 배포
사용 모델
RandomForestRegressor
GradientBoostingRegressor
Ridge
KNeighborsRegressor
TensorFlow Deep Learning Model

각 모델의 MAE를 비교하여 가장 성능이 좋은 모델을 자동으로 선택합니다.

사용 방법
https://4mation.net/kit-archive 접속
원하는 유니폼 클릭
URL 마지막 숫자 코드 복사
웹앱에 입력 후 모델 학습
사이즈·등급·마킹·패치 정보 입력 후 가격 예측
Tech Stack

Python · Pandas · Scikit-learn · TensorFlow · Streamlit

Web App

https://re4mo-price-prediction.streamlit.app/