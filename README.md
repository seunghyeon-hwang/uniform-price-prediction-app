유니폼 가격 예측 모델

축구 유니폼 거래 사이트 '레사모' 거래 데이터를 기반으로 축구 유니폼의 예상 거래 가격을 예측하는 딥러닝 기반 웹앱 프로젝트입니다.

사용자가 유니폼 코드와 사이즈, 상태, 마킹/패치 여부 등을 입력하면
실제 거래 데이터를 기반으로 예상 시세를 계산합니다.

웹앱 링크 : https://re4mo-price-prediction.streamlit.app/

Features
4mation 거래 데이터 자동 수집
유니폼 상태 및 옵션 기반 가격 예측
딥러닝(TensorFlow) 기반 회귀 모델
Streamlit 기반 웹앱 UI
실시간 사용자 입력 예측
GitHub + Streamlit Cloud 배포

Tech Stack
Python
Streamlit
TensorFlow
Scikit-learn
Pandas
NumPy
Requests
Model Input Features

현재 모델은 아래 feature를 기반으로 가격을 예측합니다.

Feature	설명
유니폼코드	유니폼 고유 코드
사이즈	S / M / L / XL 등
등급	미개봉 / S급 / A급 등
마킹번호	선수 마킹 여부
마킹오피셜	오피셜 마킹 여부
패치유무	패치 존재 여부
패치오피셜	오피셜 패치 여부
등록일	거래 등록 시간(timestamp)


주요 구현 내용
requests 기반 API 데이터 수집
JSON 데이터 파싱 및 전처리
거래 데이터를 CSV/DataFrame 형태로 변환
MinMaxScaler 기반 정규화
TensorFlow 기반 가격 예측 모델 학습
Streamlit 기반 사용자 인터페이스 구현
GitHub 및 Streamlit Cloud 배포
Limitations
일부 유니폼은 API 특성상 최근 거래 일부만 반영될 수 있습니다.
거래 데이터 수가 적은 유니폼은 예측 정확도가 낮을 수 있습니다.
현재는 단일 유니폼 기준으로 실시간 학습 후 예측하는 구조입니다.
실제 거래 시세와 차이가 발생할 수 있습니다.
Future Improvements
전체 거래 데이터 수집 자동화
거래량 기반 신뢰도 표시
LightGBM/XGBoost 기반 모델 비교
저장된 학습 모델 사용 구조로 개선
사용자 피드백 기반 모델 개선
가격 추이 시각화 기능 추가
인기 유니폼 검색 통계 기능 추가
Motivation

평소 축구 유니폼 거래 시세를 자주 확인하면서:

현재 판매 매물이 적정 가격인지
특정 시즌/마킹이 가격에 얼마나 영향을 주는지
실제 거래 기반 시세를 자동으로 계산할 수 있는지

궁금했고, 이를 AI/딥러닝 프로젝트 형태로 구현해보고자 개발했습니다.

Developer

황승현

AI / Deep Learning
Python Backend
Streamlit Deployment
Data-driven Price Prediction Project
