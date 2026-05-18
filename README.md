# 유니폼 가격 예측 웹앱

4mation 거래 데이터를 기반으로 축구 유니폼의 예상 거래 가격을 예측하는 Streamlit 웹앱입니다.

사용자가 유니폼 코드와 사이즈, 상태, 마킹/패치 여부 등을 입력하면 거래 데이터를 수집하고 딥러닝 모델을 통해 예상 시세를 계산합니다.

배포 링크:  
https://re4mo-price-prediction.streamlit.app/

## 주요 기능
- 유니폼 거래 데이터 자동 수집
- 데이터 전처리 및 정규화
- 딥러닝 기반 가격 예측
- 사이즈·등급·마킹·패치 반영
- Streamlit 기반 웹앱 배포

## 사용 방법
1. https://4mation.net/kit-archive 접속
2. 원하는 유니폼 클릭
3. URL 마지막 숫자 코드 복사
4. 웹앱에 입력 후 거래 데이터 불러오기
5. 조건 입력 후 예상 가격 확인

## 기술 스택
- Python
- Streamlit
- TensorFlow/Keras
- Pandas, NumPy
- Scikit-learn
- Requests
- GitHub

## 프로젝트 목적
실제 거래 데이터를 활용한 가격 예측 모델 구현과 웹 배포 경험을 목표로 제작한 개인 프로젝트입니다.
