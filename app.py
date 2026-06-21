import streamlit as st
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf
import requests
import json
import os
import time

st.set_page_config(
    page_title="유니폼 가격 예측기",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="auto",
)
currenttime = int(time.time())

st.markdown(
    """
    <h1 style='font-size:2.5rem;'>
    유니폼 가격 예측 모델
    </h1>
    """,
    unsafe_allow_html=True,
)
st.write("레사모 거래 데이터를 기반으로 유니폼의 예상 거래 가격을 예측하는 웹앱입니다.")

with st.expander("사용방법", expanded=True):
    st.markdown("""
        1. [4mation Kit Archive](https://4mation.net/kit-archive)에 접속합니다.
        2. 가격을 알고 싶은 유니폼을 클릭합니다.
        3. 열린 페이지 URL의 마지막 숫자 코드를 복사합니다.
        4. 아래 유니폼 코드 입력칸에 붙여넣습니다.
        5. 거래 데이터를 불러온 뒤, 사이즈·등급·마킹·패치 정보를 입력하면 예상 가격을 확인할 수 있습니다.

        예시 URL:

        `https://4mation.net/kit-archive/12345`

        입력할 유니폼 코드:

        `12345`
        """)
유니폼코드 = st.text_input("유니폼 코드 : ")

size_map = {
    "S": 0,
    "s": 0,
    "M": 1,
    "m": 1,
    "L": 2,
    "l": 2,
    "XL": 3,
    "xl": 3,
    "2XL": 4,
    "2xl": 4,
    "3XL": 5,
    "3xl": 5,
    "해외 S": 0,
    "해외 s": 0,
    "해외 M": 1,
    "해외 m": 1,
    "해외 L": 2,
    "해외 l": 2,
    "해외 XL": 3,
    "해외 xl": 3,
    "해외 2XL": 4,
    "해외 2xl": 4,
    "해외 3XL": 5,
    "해외 3xl": 5,
    "국내 S": 0,
    "국내 s": 0,
    "국내 M": 0,
    "국내 m": 0,
    "국내 L": 1,
    "국내 l": 1,
    "국내 XL": 2,
    "국내 xl": 2,
    "국내 2XL": 2.5,
    "국내 2xl": 2.5,
    "국내 3XL": 3,
    "국내 3xl": 3,
    "국내 4XL": 3.5,
    "국내 4xl": 3.5,
}

true_false_map = {True: 1, False: 0, None: 0, 0: 0, 1: 1}


def iso8601_z_to_timestamp(s: str) -> int:
    if not s:
        return 0

    original = s

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        return int(datetime.fromisoformat(s).timestamp())
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                dt = datetime.strptime(original, fmt)
                return int(dt.replace(tzinfo=timezone.utc).timestamp())
            except ValueError:
                continue

    raise ValueError(f"지원하지 않는 시간 포맷: {original}")


def 거래데이터수집(유니폼코드, price_list, size_map, true_false_map):
    filename = f"{유니폼코드}.csv"

    with open(filename, "w", encoding="utf-8-sig") as file:
        file.write(
            "유니폼코드,사이즈,등급,마킹번호,마킹오피셜,패치유무,패치오피셜,거래경과일,가격\n"
        )

        for i in price_list:
            if "size" in i and "price" in i and "grade" in i:
                size = i.get("size", "")
                price = i.get("price", "")
                grade = i.get("grade", "")
                sellingtime = i.get("datetime", "")

                timestamp = iso8601_z_to_timestamp(sellingtime)
                days_ago = (currenttime - timestamp) / 86400

                marking = i.get("marking", {}) or {}
                patch = i.get("patch", {}) or {}

                marking_number = marking.get("marking_number", 0) or 0
                marking_official = marking.get("official", 0)
                is_patch = patch.get("is_patch", 0)
                patch_official = patch.get("official", 0)

                if isinstance(size, list) and len(size) > 0:
                    size_value = size_map.get(size[0], -1)
                else:
                    size_value = size_map.get(size, -1)

                file.write(
                    f"{int(유니폼코드)},{size_value},{grade},{marking_number},{true_false_map.get(marking_official, 0)},{true_false_map.get(is_patch, 0)},{true_false_map.get(patch_official, 0)},{days_ago},{price}\n"
                )

    return filename


if st.button("거래 데이터 불러오기 및 모델 학습"):
    if not 유니폼코드:
        st.warning("유니폼 코드를 입력하세요.")
        st.stop()

    try:
        uniform_code_value = int(유니폼코드)
    except ValueError:
        st.error("유니폼 코드는 숫자로 입력해야 합니다.")
        st.stop()
    # https://4mation.net/api/price/getpricedata/completed/{유니폼코드}?is_all=true
    # url = f"https://4mation.net/api/product/detail/{유니폼코드}"
    url = (
        f"https://4mation.net/api/price/getpricedata/completed/{유니폼코드}?is_all=true"
    )

    try:
        data = requests.get(url, timeout=10)
    except requests.exceptions.RequestException:
        st.error("API 요청 중 문제가 발생했습니다.")
        st.stop()

    if data.status_code != 200:
        st.error("API 요청에 실패했습니다.")
        st.stop()

    딕셔너리 = data.json()

    if "data" not in 딕셔너리:
        st.error("거래 데이터를 찾을 수 없습니다.")
        st.stop()

    price_list = 딕셔너리["data"]

    filename = 거래데이터수집(유니폼코드, price_list, size_map, true_false_map)

    df = pd.read_csv(filename, encoding="utf-8-sig")

    if os.path.exists(filename):
        os.remove(filename)

    if len(df) < 5:
        st.warning("학습에 사용할 거래 데이터가 너무 적습니다.")
        st.dataframe(df)
        st.stop()

    st.subheader(f"수집된 거래 데이터 개수 : {len(price_list) - 1}")
    st.dataframe(df)

    X = df.drop(["가격", "유니폼코드"], axis=1)
    y = df["가격"]
    X['마킹번호'] = X['마킹번호'].astype(int).astype(str)
    X = pd.get_dummies(X, columns=['마킹번호'], prefix='등번호')
    feature_columns = X.columns
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )

    ridge_model = Ridge(alpha=1.0, random_state=42)
    knn_model = KNeighborsRegressor(
        n_neighbors=5, weights="distance", metric="minkowski", p=2
    )

    dl_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(x_train_scaled.shape[1],)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    models = {
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model,
        "Ridge": ridge_model,
        "KNN": knn_model,
    }

    model_mae = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(X_valid_scaled)
        mae = mean_absolute_error(y_valid, y_pred)
        model_mae[name] = mae
        trained_models[name] = model

    dl_model.compile(
        optimizer="adam",
        loss="mae",
        metrics=["mae"],
    )

    with st.spinner("모델 학습 중입니다."):
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=30, restore_best_weights=True
        )

        history = dl_model.fit(
            x_train_scaled,
            y_train,
            validation_data=(X_valid_scaled, y_valid),
            epochs=300,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0,
        )

    dl_pred = dl_model.predict(X_valid_scaled).flatten()
    dl_mae = mean_absolute_error(y_valid, dl_pred)
    model_mae["Deep Learning"] = dl_mae
    trained_models["Deep Learning"] = dl_model

    best_model_name = min(model_mae, key=model_mae.get)
    best_model = trained_models[best_model_name]
    best_mae = model_mae[best_model_name]

    st.session_state["model"] = best_model
    st.session_state["model_name"] = best_model_name
    st.session_state["best_mae"] = best_mae
    st.session_state["scaler"] = scaler
    st.session_state["feature_columns"] = feature_columns
    st.success(f"""
    모델 학습이 완료되었습니다.
    """)


st.subheader("가격 예측")

size = st.selectbox("US사이즈를 선택하세요", ["S", "M", "L", "XL", "2XL", "3XL"])

grade = st.number_input(
    "등급을 입력하세요",
    min_value=1,
    max_value=6,
    value=3,
    help="미개봉: 1, 개봉새제품: 2, S급: 3, A급: 4, B급: 5, C급: 6",
)

marking_number = st.number_input(
    "마킹 번호를 입력하세요. 마킹이 없다면 0", min_value=0, value=0
)

if marking_number == 0:
    marking_official = 0
else:
    marking_official = st.number_input(
        "마킹 오피셜이면 1, 아니면 0", min_value=0, max_value=1, value=1
    )

patch = st.number_input("패치 있으면 1, 없으면 0", min_value=0, max_value=1, value=0)

if patch == 0:
    patch_official = 0
else:
    patch_official = st.number_input(
        "패치 오피셜이면 1, 아니면 0", min_value=0, max_value=1, value=1
    )

if st.button("가격 예측하기"):
    if "model" not in st.session_state or "scaler" not in st.session_state:
        st.warning("먼저 거래 데이터를 불러오고 모델을 학습하세요.")
        st.stop()

    best_model = st.session_state["model"]
    best_model_name = st.session_state["model_name"]
    scaler = st.session_state["scaler"]
    
    user_input_df = pd.DataFrame([{
    "사이즈": size_map.get(size),
    "등급": grade,
    "마킹번호": str(int(marking_number)),
    "마킹오피셜": marking_official,
    "패치유무": patch,
    "패치오피셜": patch_official,
    "거래경과일": 0,
    }])

    # 학습 때와 동일하게 One-Hot Encoding
    user_input_df = pd.get_dummies(
        user_input_df,
        columns=["마킹번호"],
        prefix="등번호"
    )

    # 학습 데이터의 column 구조와 맞추기
    feature_columns = st.session_state["feature_columns"]

    user_input_df = user_input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    user_input_scaled = scaler.transform(user_input_df)

    예측값 = best_model.predict(user_input_scaled)

    predicted_price = int(round(float(np.array(예측값).flatten()[0]), -3))
    mae = st.session_state["best_mae"]
    low_price = int(round(predicted_price - mae * 0.5, -3))
    high_price = int(round(predicted_price + mae * 0.5, -3))

    st.success(f"예측 가격: {predicted_price:,} 원")

    st.caption(f"예상 거래 범위: {low_price:,} ~ {high_price:,} 원")
    st.caption(f"사용 모델: {best_model_name}")

st.markdown("---")

st.markdown(
    """
<div style='text-align: center; color: gray; font-size: 14px;'>

<b>유니폼 가격 예측 모델</b><br><br>

Developed by 황승현<br>
Python · Streamlit · TensorFlow 기반 개인 프로젝트<br><br>

레사모 거래 데이터를 기반으로 유니폼 예상 시세를 예측합니다.<br>
피드백은 모델 개선에 큰 도움이 됩니다.<br><br>

<a href="https://github.com/seunghyeon-hwang/uniform-price-prediction-app" target="_blank">
GitHub Repository
</a>

</div>
""",
    unsafe_allow_html=True,
)
