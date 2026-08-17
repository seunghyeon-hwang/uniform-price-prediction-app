import streamlit as st
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf
import requests

st.set_page_config(
    page_title="유니폼 가격 예측기",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="auto",
)

MIN_TRAINING_SAMPLES = 10
PREDICTION_INTERVAL_PERCENTILE = 90
MODEL_STATE_KEYS = (
    "model",
    "model_name",
    "best_mae",
    "prediction_interval_error",
    "scaler",
    "feature_columns",
    "trained_uniform_code",
)

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
uniform_code = st.text_input("유니폼 코드 : ")

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
    if not isinstance(s, str) or not s.strip():
        raise ValueError("거래 시간이 없습니다.")

    original = s.strip()
    s = original

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                dt = datetime.strptime(original, fmt)
                return int(dt.replace(tzinfo=timezone.utc).timestamp())
            except ValueError:
                continue

    raise ValueError(f"지원하지 않는 시간 포맷: {original}")


def parse_numeric_value(value):
    if isinstance(value, str):
        value = value.replace(",", "").strip()
    return float(value)


def build_transactions_dataframe(uniform_code, price_list, size_map, true_false_map):
    rows = []
    skipped_count = 0
    current_timestamp = int(datetime.now(timezone.utc).timestamp())

    for item in price_list:
        if not isinstance(item, dict):
            skipped_count += 1
            continue

        try:
            size = item["size"]
            price = parse_numeric_value(item["price"])
            grade = parse_numeric_value(item["grade"])
            timestamp = iso8601_z_to_timestamp(item.get("datetime"))

            if isinstance(size, list):
                size = size[0] if size else ""
            if isinstance(size, str):
                size = size.strip()
            size_value = size_map.get(size, -1)

            if (
                size_value == -1
                or not np.isfinite(price)
                or not np.isfinite(grade)
                or price <= 0
                or not 1 <= grade <= 6
                or timestamp > current_timestamp
            ):
                raise ValueError("학습에 사용할 수 없는 거래 데이터입니다.")

            marking = item.get("marking", {}) or {}
            patch = item.get("patch", {}) or {}
            if not isinstance(marking, dict) or not isinstance(patch, dict):
                raise ValueError("마킹 또는 패치 형식이 올바르지 않습니다.")

            marking_number = int(
                parse_numeric_value(marking.get("marking_number", 0) or 0)
            )
            days_ago = (current_timestamp - timestamp) / 86400

            rows.append(
                {
                    "유니폼코드": int(uniform_code),
                    "사이즈": size_value,
                    "등급": grade,
                    "마킹번호": marking_number,
                    "마킹오피셜": true_false_map.get(marking.get("official", 0), 0),
                    "패치유무": true_false_map.get(patch.get("is_patch", 0), 0),
                    "패치오피셜": true_false_map.get(patch.get("official", 0), 0),
                    "거래경과일": days_ago,
                    "가격": price,
                }
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            skipped_count += 1

    columns = [
        "유니폼코드",
        "사이즈",
        "등급",
        "마킹번호",
        "마킹오피셜",
        "패치유무",
        "패치오피셜",
        "거래경과일",
        "가격",
    ]
    return pd.DataFrame(rows, columns=columns), skipped_count


def sklearn_models(training_size):
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        ),
        "Ridge": Ridge(alpha=1.0),
        "KNN": KNeighborsRegressor(
            n_neighbors=min(5, training_size),
            weights="distance",
            metric="minkowski",
            p=2,
        ),
    }


def build_deep_learning_model(input_size):
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(input_size,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )


def cross_validate_models(X, y):
    split_count = min(5, max(3, len(X) // 5))
    kfold = KFold(n_splits=split_count, shuffle=True, random_state=42)
    model_names = [*sklearn_models(len(X)), "Deep Learning"]
    oof_predictions = {
        name: np.empty(len(X), dtype=float) for name in model_names
    }
    dl_epoch_counts = []

    for fold_number, (train_index, valid_index) in enumerate(kfold.split(X), start=1):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        fold_scaler = MinMaxScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train)
        X_valid_scaled = fold_scaler.transform(X_valid)

        for name, model in sklearn_models(len(X_train)).items():
            model.fit(X_train_scaled, y_train)
            oof_predictions[name][valid_index] = model.predict(X_valid_scaled)

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42 + fold_number)
        dl_model = build_deep_learning_model(X_train_scaled.shape[1])
        dl_model.compile(optimizer="adam", loss="mae", metrics=["mae"])
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )
        history = dl_model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_valid_scaled, y_valid),
            epochs=200,
            batch_size=min(16, len(X_train)),
            callbacks=[early_stop],
            verbose=0,
        )
        oof_predictions["Deep Learning"][valid_index] = dl_model.predict(
            X_valid_scaled, verbose=0
        ).flatten()
        dl_epoch_counts.append(len(history.history["loss"]))

    model_mae = {
        name: mean_absolute_error(y, predictions)
        for name, predictions in oof_predictions.items()
    }
    return model_mae, oof_predictions, dl_epoch_counts


def train_model_on_full_dataset(model_name, X, y, dl_epoch_counts):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    if model_name == "Deep Learning":
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        model = build_deep_learning_model(X_scaled.shape[1])
        model.compile(optimizer="adam", loss="mae", metrics=["mae"])
        final_epochs = max(20, int(np.median(dl_epoch_counts)))
        model.fit(
            X_scaled,
            y,
            epochs=final_epochs,
            batch_size=min(16, len(X)),
            verbose=0,
        )
    else:
        model = sklearn_models(len(X))[model_name]
        model.fit(X_scaled, y)

    return model, scaler


if st.button("거래 데이터 불러오기 및 모델 학습"):
    if not uniform_code:
        st.warning("유니폼 코드를 입력하세요.")
        st.stop()

    try:
        uniform_code_value = int(uniform_code)
    except ValueError:
        st.error("유니폼 코드는 숫자로 입력해야 합니다.")
        st.stop()

    for key in MODEL_STATE_KEYS:
        st.session_state.pop(key, None)

    url = (
        "https://4mation.net/api/price/getpricedata/completed/"
        f"{uniform_code_value}?is_all=true"
    )

    try:
        data = requests.get(url, timeout=10)
    except requests.exceptions.RequestException:
        st.error("API 요청 중 문제가 발생했습니다.")
        st.stop()

    if data.status_code != 200:
        st.error("API 요청에 실패했습니다.")
        st.stop()

    try:
        response_payload = data.json()
    except ValueError:
        st.error("API 응답을 해석할 수 없습니다.")
        st.stop()

    if not isinstance(response_payload, dict) or not isinstance(
        response_payload.get("data"), list
    ):
        st.error("거래 데이터를 찾을 수 없습니다.")
        st.stop()

    price_list = response_payload["data"]
    df, skipped_count = build_transactions_dataframe(
        uniform_code_value, price_list, size_map, true_false_map
    )

    if len(df) < MIN_TRAINING_SAMPLES:
        st.warning(
            "유효한 거래 데이터가 너무 적습니다. "
            f"최소 {MIN_TRAINING_SAMPLES}건이 필요합니다."
        )
        if skipped_count:
            st.info(f"형식이 올바르지 않은 거래 {skipped_count}건을 제외했습니다.")
        st.dataframe(df)
        st.stop()

    st.subheader(f"학습에 사용하는 거래 데이터 개수: {len(df)}")
    if skipped_count:
        st.info(f"형식이 올바르지 않은 거래 {skipped_count}건을 제외했습니다.")
    st.dataframe(df)

    X = df.drop(["가격", "유니폼코드"], axis=1)
    y = df["가격"].astype(float).reset_index(drop=True)
    X["마킹번호"] = X["마킹번호"].astype(int).astype(str)
    X = pd.get_dummies(X, columns=["마킹번호"], prefix="등번호")
    X = X.astype(float).reset_index(drop=True)
    feature_columns = list(X.columns)

    with st.spinner("교차 검증으로 모델을 비교하고 전체 데이터로 학습 중입니다."):
        model_mae, oof_predictions, dl_epoch_counts = cross_validate_models(X, y)
        best_model_name = min(model_mae, key=model_mae.get)
        best_mae = model_mae[best_model_name]
        residuals = np.abs(y.to_numpy() - oof_predictions[best_model_name])
        prediction_interval_error = float(
            np.percentile(residuals, PREDICTION_INTERVAL_PERCENTILE)
        )
        best_model, scaler = train_model_on_full_dataset(
            best_model_name, X, y, dl_epoch_counts
        )

    st.session_state["model"] = best_model
    st.session_state["model_name"] = best_model_name
    st.session_state["best_mae"] = best_mae
    st.session_state["prediction_interval_error"] = prediction_interval_error
    st.session_state["scaler"] = scaler
    st.session_state["feature_columns"] = feature_columns
    st.session_state["trained_uniform_code"] = uniform_code_value

    result_df = pd.DataFrame(
        {
            "모델": model_mae.keys(),
            "교차 검증 MAE": model_mae.values(),
        }
    ).sort_values("교차 검증 MAE")
    st.dataframe(result_df, hide_index=True)
    st.success(
        f"모델 학습이 완료되었습니다. 선택 모델: {best_model_name} "
        f"(교차 검증 MAE {best_mae:,.0f}원)"
    )


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

    try:
        current_uniform_code = int(uniform_code)
    except ValueError:
        st.warning("학습한 유니폼 코드를 입력한 뒤 다시 예측하세요.")
        st.stop()

    if current_uniform_code != st.session_state.get("trained_uniform_code"):
        st.warning(
            "현재 입력한 유니폼 코드와 학습한 코드가 다릅니다. "
            "거래 데이터를 다시 불러와 모델을 학습하세요."
        )
        st.stop()

    best_model = st.session_state["model"]
    best_model_name = st.session_state["model_name"]
    scaler = st.session_state["scaler"]

    user_input_df = pd.DataFrame(
        [
            {
                "사이즈": size_map.get(size),
                "등급": grade,
                "마킹번호": str(int(marking_number)),
                "마킹오피셜": marking_official,
                "패치유무": patch,
                "패치오피셜": patch_official,
                "거래경과일": 0,
            }
        ]
    )

    # 학습 때와 동일하게 One-Hot Encoding
    user_input_df = pd.get_dummies(user_input_df, columns=["마킹번호"], prefix="등번호")

    # 학습 데이터의 column 구조와 맞추기
    feature_columns = st.session_state["feature_columns"]

    user_input_df = user_input_df.reindex(
        columns=feature_columns, fill_value=0
    ).astype(float)

    user_input_scaled = scaler.transform(user_input_df)

    prediction = best_model.predict(user_input_scaled)

    predicted_price = max(
        0, int(round(float(np.array(prediction).flatten()[0]), -3))
    )
    interval_error = st.session_state["prediction_interval_error"]
    low_price = max(0, int(round(predicted_price - interval_error, -3)))
    high_price = int(round(predicted_price + interval_error, -3))

    st.success(f"예측 가격: {predicted_price:,} 원")

    st.caption(f"예상 거래 범위: {low_price:,} ~ {high_price:,} 원")
    st.caption(
        f"사용 모델: {best_model_name} · 교차 검증 잔차의 "
        f"{PREDICTION_INTERVAL_PERCENTILE}% 기준"
    )
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
