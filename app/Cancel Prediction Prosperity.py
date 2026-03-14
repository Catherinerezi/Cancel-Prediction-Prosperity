# -*- coding: utf-8 -*-
"""
TakeHomeTestDS_HotelCancellation_streamlit.py

Streamlit app for:
- Hotel booking cancellation prediction (is_canceled)
- Interactive EDA with Altair
- Room-level transformation and feature engineering
- Model training & tuning (LogReg / RandomForest / LightGBM*)
- Feature importance (intrinsic + permutation)
- Threshold tuning and prediction simulator
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Optional LightGBM
HAS_LGBM = False
try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


st.set_page_config(page_title="Hotel Cancellation Prediction", layout="wide")
st.title("🏨 Hotel Booking Cancellation Prediction")

with st.expander("🎯 Tujuan", expanded=True):
    st.markdown(
        """
Membangun app interaktif untuk:
- memahami driver pembatalan booking hotel,
- melakukan feature engineering berbasis workflow dari notebook/PDF,
- membandingkan model klasifikasi,
- memilih threshold keputusan,
- dan mensimulasikan probabilitas cancel untuk satu booking.
"""
    )

# DATA LOADING
@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    file_id = "1Qu4Q8rwWM7TN_Bqzmpw0EtlLsaaiuVtj"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return pd.read_csv(url)

# RAW CLEANING
def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "reservation_status_date" in df.columns:
        df["reservation_status_date"] = pd.to_datetime(
            df["reservation_status_date"], errors="coerce"
        )

    missing_pct = df.isna().mean() * 100
    drop_cols = missing_pct[missing_pct > 20].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    fill_candidates = ["agent", "country", "children"]
    for col in fill_candidates:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median(skipna=True))
        else:
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    if "is_canceled" in df.columns:
        df["is_canceled"] = df["is_canceled"].astype(int)

    return df

# ROOM SPLITTING (from notebook logic)
def split_rooms_cap4(row: pd.Series) -> list[dict]:
    adults = int(row.get("adults", 0) or 0)
    children = int(row.get("children", 0) or 0)
    babies = int(row.get("babies", 0) or 0)

    minors = children + babies
    rooms: list[tuple[int, int, int]] = []
    violation = False

    while minors > 0:
        a = 1 if adults > 0 else 0
        if a == 0:
            violation = True
        else:
            adults -= 1

        take = min(3, minors)
        c_take = min(children, take)
        b_take = min(babies, take - c_take)
        children -= c_take
        babies -= b_take
        minors -= c_take + b_take

        extra_adults = min(4 - a - c_take - b_take, adults)
        adults -= extra_adults
        a += extra_adults
        rooms.append((a, c_take, b_take))

    while adults > 0:
        a = min(4, adults)
        rooms.append((a, 0, 0))
        adults -= a

    if not rooms:
        rooms = [(0, 0, 0)]

    base = row.to_dict()
    out: list[dict] = []
    for a, c, b in rooms:
        r = base.copy()
        r.update(
            {
                "adults": a,
                "children": c,
                "babies": b,
                "viol_minors_without_adult": violation,
            }
        )
        out.append(r)
    return out


@st.cache_data(show_spinner=True)
def build_room_level_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_raw_data(df_raw)

    records: list[dict] = []
    for _, row in df.iterrows():
        records.extend(split_rooms_cap4(row))

    df_room = pd.DataFrame(records).reset_index(drop=True)
    df_room["Invoice_ID"] = np.arange(1, len(df_room) + 1, dtype=int)
    df_room["bookingID"] = df_room["bookingID"].astype(str)
    df_room["rooms_in_booking"] = df_room.groupby("bookingID")["Invoice_ID"].transform(
        "count"
    )
    df_room["bulk_3p_rooms"] = df_room["rooms_in_booking"] >= 3

    viol_mask = (df_room["adults"] < 1) & (
        (df_room["children"] + df_room["babies"]) > 0
    )
    bad_bookings = df_room.loc[viol_mask, "bookingID"].unique()
    if len(bad_bookings) > 0:
        df_room = df_room[~df_room["bookingID"].isin(bad_bookings)].reset_index(drop=True)

    df_room["Invoice_ID"] = np.arange(1, len(df_room) + 1, dtype=int)
    df_room["rooms_in_booking"] = df_room.groupby("bookingID")["Invoice_ID"].transform(
        "count"
    )
    df_room["bulk_3p_rooms"] = df_room["rooms_in_booking"] >= 3

    cont_cols = [
        "adr",
        "lead_time",
        "stays_in_week_nights",
        "stays_in_weekend_nights",
        "total_of_special_requests",
    ]
    cont_cols = [c for c in cont_cols if c in df_room.columns]
    for c in cont_cols:
        s = pd.to_numeric(df_room[c], errors="coerce").astype(float)
        dt = df_room[c].dtype
        for _ in range(8):
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr == 0:
                break
            lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            m = (s < lb) | (s > ub)
            if not m.any():
                break
            s = s.clip(lb, ub)
        df_room[c] = s.astype(dt)

    return df_room.reset_index(drop=True)

# FEATURE ENGINEERING
def add_features(df_room: pd.DataFrame) -> pd.DataFrame:
    df_room = df_room.copy()

    df_room["minors"] = df_room["children"] + df_room["babies"]
    df_room["party_size"] = df_room["adults"] + df_room["minors"]
    df_room["stay_nights"] = (
        df_room["stays_in_week_nights"] + df_room["stays_in_weekend_nights"]
    )
    df_room["weekend_ratio"] = np.where(
        df_room["stay_nights"] > 0,
        df_room["stays_in_weekend_nights"] / df_room["stay_nights"],
        0.0,
    )
    df_room["room_revenue"] = df_room["adr"] * df_room["stay_nights"]

    rev_booking = df_room.groupby("bookingID")["room_revenue"].sum().rename(
        "booking_revenue"
    )
    df_room = df_room.merge(rev_booking, on="bookingID", how="left")

    season_map = {
        "December": "High",
        "January": "High",
        "February": "High",
        "June": "Peak",
        "July": "Peak",
        "August": "Peak",
    }
    df_room["season"] = df_room["arrival_date_month"].map(season_map).fillna("Shoulder")

    df_room["lead_time_bin"] = pd.cut(
        df_room["lead_time"],
        bins=[-1, 7, 30, 90, 180, 9999],
        labels=["≤7d", "8–30d", "31–90d", "91–180d", ">180d"],
    )

    try:
        df_room["adr_bin"] = pd.qcut(
            df_room["adr"].rank(method="first"),
            q=5,
            labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        )
    except Exception:
        df_room["adr_bin"] = "Q3"

    df_room["family_flag"] = df_room["minors"] > 0
    df_room["couple_flag"] = (df_room["adults"].eq(2)) & (df_room["minors"].eq(0))
    df_room["solo_flag"] = (df_room["adults"].eq(1)) & (df_room["minors"].eq(0))

    df_room["bulk_flag"] = df_room["rooms_in_booking"] >= 3

    bookings_bulk = (
        df_room.drop_duplicates(["bookingID"])  # booking-level per agent
        .groupby("agent")["bulk_flag"]
        .sum()
        .rename("bulk_bookings_by_agent")
    )
    df_room = df_room.merge(bookings_bulk, on="agent", how="left")
    df_room["bulk_booker_agent_flag"] = (
        df_room["bulk_bookings_by_agent"].fillna(0).ge(3)
    )

    df_room["waiting_list_flag"] = df_room["days_in_waiting_list"] > 0
    df_room["req_flag"] = df_room["total_of_special_requests"] > 0
    df_room["car_flag"] = df_room["required_car_parking_spaces"] > 0
    df_room["seg_x_chan"] = (
        df_room["market_segment"].astype(str)
        + " | "
        + df_room["distribution_channel"].astype(str)
    )
    df_room["room_mismatch_flag"] = (
        df_room["reserved_room_type"].astype(str)
        != df_room["assigned_room_type"].astype(str)
    )
    df_room["changes_flag"] = df_room["booking_changes"] > 0

    agg = (
        df_room.groupby("bookingID")
        .agg(
            hotel=("hotel", "first"),
            is_canceled=("is_canceled", "max"),
            season=("season", "first"),
            lead_time=("lead_time", "first"),
            lead_time_bin=("lead_time_bin", "first"),
            adr_median=("adr", "median"),
            booking_revenue=("booking_revenue", "first"),
            rooms_in_booking=("rooms_in_booking", "first"),
            bulk_flag=("bulk_flag", "first"),
            agent=("agent", "first"),
            bulk_booker_agent_flag=("bulk_booker_agent_flag", "first"),
            family_any=("family_flag", "max"),
            weekend_ratio_mean=("weekend_ratio", "mean"),
            room_mismatch_any=("room_mismatch_flag", "max"),
            seg_x_chan=("seg_x_chan", "first"),
            req_any=("req_flag", "max"),
            waiting_list_any=("waiting_list_flag", "max"),
            changes_any=("changes_flag", "max"),
        )
    )

    add_cols = [
        c
        for c in [
            "adr_median",
            "booking_revenue",
            "rooms_in_booking",
            "bulk_flag",
            "bulk_booker_agent_flag",
            "family_any",
            "weekend_ratio_mean",
            "room_mismatch_any",
            "seg_x_chan",
            "req_any",
            "waiting_list_any",
            "changes_any",
            "is_canceled",
        ]
        if c in agg.columns and c not in df_room.columns
    ]

    onefile = df_room.merge(agg[add_cols], left_on="bookingID", right_index=True, how="left")
    onefile = onefile.reset_index(drop=True)

    if ("Invoice_ID" not in onefile.columns) or (not onefile["Invoice_ID"].is_unique):
        onefile["Invoice_ID"] = np.arange(1, len(onefile) + 1, dtype=int)

    return onefile

# MODEL DATA PREP
def make_time_split(df_model: pd.DataFrame):
    df = df_model.copy()
    df["reservation_status_date"] = pd.to_datetime(
        df["reservation_status_date"], errors="coerce"
    )

    train_df = df[df["reservation_status_date"] < "2019-01-01"].copy()
    test_df = df[df["reservation_status_date"] >= "2019-01-01"].copy()

    target = "is_canceled"
    drop_base = [target, "reservation_status_date", "Invoice_ID", "bookingID"]
    drop_leak = [
        "reservation_status",
        "assigned_room_type",
        "room_mismatch_flag",
        "room_mismatch_any",
        "changes_any",
        "booking_revenue",
        "adr_median",
        "rooms_in_booking",
        "bulk_flag",
        "bulk_bookings_by_agent",
        "seg_x_chan",
        "waiting_list_any",
        "req_any",
    ]
    extra_leak = [
        "booking_changes",
        "changes_flag",
        "days_in_waiting_list",
        "waiting_list_flag",
        "bulk_booker_agent_flag",
    ]

    to_drop_train = [c for c in (drop_base + drop_leak + extra_leak) if c in train_df.columns]
    to_drop_test = [c for c in (drop_base + drop_leak + extra_leak) if c in test_df.columns]

    X_train = train_df.drop(columns=to_drop_train)
    y_train = train_df[target].astype(int)
    X_test = test_df.drop(columns=to_drop_test)
    y_test = test_df[target].astype(int)

    return X_train, X_test, y_train, y_test

# PREPROCESSORS + METRICS
def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()

    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("sc", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "ohe",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def evaluate_classifier(pipe: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y_test, p),
        "PR_AUC": average_precision_score(y_test, p),
        "Brier": brier_score_loss(y_test, p),
        "LogLoss": log_loss(y_test, p),
        "Precision@0.5": precision_score(y_test, pred, zero_division=0),
        "Recall@0.5": recall_score(y_test, pred, zero_division=0),
        "F1@0.5": f1_score(y_test, pred, zero_division=0),
    }


@st.cache_resource(show_spinner=True)
def train_all_models(X_train, y_train, X_test, y_test):
    preprocessor = build_preprocessor(X_train, scale_numeric=True)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    candidates: list[tuple[str, Pipeline, dict]] = [
        (
            "Logistic Regression",
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            solver="liblinear",
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.5, 1.0, 2.0]},
        ),
        (
            "Random Forest",
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            random_state=42,
                            n_jobs=-1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            ),
            {
                "model__max_depth": [8, 12, None],
                "model__min_samples_leaf": [1, 3],
                "model__max_features": ["sqrt", "log2"],
            },
        ),
    ]

    if HAS_LGBM:
        candidates.append(
            (
                "LightGBM",
                Pipeline(
                    [
                        ("preprocess", preprocessor),
                        (
                            "model",
                            LGBMClassifier(
                                n_estimators=300,
                                learning_rate=0.05,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                random_state=42,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
                {
                    "model__num_leaves": [31, 63],
                    "model__min_child_samples": [20, 40],
                },
            )
        )

    rows = []
    best_models: dict[str, Pipeline] = {}

    for name, pipe, grid in candidates:
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        best_pipe = clone(gs.best_estimator_)
        best_pipe.fit(X_train, y_train)
        best_models[name] = best_pipe

        metrics = evaluate_classifier(best_pipe, X_train, y_train, X_test, y_test)
        rows.append(
            {
                "model": name,
                "cv_best_roc_auc": gs.best_score_,
                **metrics,
                "best_params": gs.best_params_,
            }
        )

    cmp = pd.DataFrame(rows).sort_values(
        ["ROC_AUC", "PR_AUC"], ascending=[False, False]
    ).reset_index(drop=True)
    best_name = cmp.loc[0, "model"]
    best_pipe = best_models[best_name]

    return cmp, best_name, best_pipe, best_models


def get_feature_names_from_pipe(pipe: Pipeline) -> list[str]:
    ct = pipe.named_steps["preprocess"]
    try:
        return list(ct.get_feature_names_out())
    except Exception:
        names = []
        for name, trans, cols in ct.transformers_:
            if name == "remainder":
                continue
            if name == "num":
                names.extend(list(cols))
            elif name == "cat":
                try:
                    ohe = trans.named_steps["ohe"]
                    names.extend(list(ohe.get_feature_names_out(cols)))
                except Exception:
                    names.extend(list(cols))
        return names


def intrinsic_importance(pipe: Pipeline) -> pd.DataFrame | None:
    model = pipe.named_steps["model"]
    feat_names = get_feature_names_from_pipe(pipe)

    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame(
            {"feature": feat_names, "importance": model.feature_importances_}
        )
        return imp.sort_values("importance", ascending=False).reset_index(drop=True)

    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        imp = pd.DataFrame({"feature": feat_names, "importance": np.abs(coef)})
        return imp.sort_values("importance", ascending=False).reset_index(drop=True)

    return None


def grouped_permutation_importance(
    pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    result = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring="average_precision",
    )

    feat_names = get_feature_names_from_pipe(pipe)
    k = min(len(feat_names), len(result.importances_mean))
    out = pd.DataFrame(
        {
            "feature": feat_names[:k],
            "perm_importance": result.importances_mean[:k],
        }
    )
    return out.sort_values("perm_importance", ascending=False).reset_index(drop=True)

# APP BODY
df_raw = load_data()
df_room = build_room_level_dataset(df_raw)
df_model = add_features(df_room)
X_train, X_test, y_train, y_test = make_time_split(df_model)
cmp, best_name, best_pipe, model_map = train_all_models(X_train, y_train, X_test, y_test)

# DATA UNDERSTANDING
st.header("1. Data Understanding")

c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    st.subheader("Raw shape")
    st.write(df_raw.shape)
    st.dataframe(df_raw.head(), use_container_width=True)
with c2:
    st.subheader("Room-level shape")
    st.write(df_room.shape)
    st.dataframe(df_room.head(), use_container_width=True)
with c3:
    st.subheader("Modeling shape")
    st.write(df_model.shape)
    st.dataframe(df_model.head(), use_container_width=True)

st.subheader("Missing value (%)")
missing_df = (
    (clean_raw_data(df_raw).isna().mean() * 100)
    .sort_values(ascending=False)
    .rename("missing_pct")
    .reset_index()
    .rename(columns={"index": "column"})
)

chart_missing = (
    alt.Chart(missing_df)
    .mark_bar()
    .encode(
        x=alt.X("missing_pct:Q", title="% Missing"),
        y=alt.Y("column:N", sort="-x", title="Column"),
        tooltip=["column", alt.Tooltip("missing_pct:Q", format=".2f")],
    )
    .properties(height=max(240, 22 * len(missing_df)))
    .interactive()
)
st.altair_chart(chart_missing, use_container_width=True)

# EDA
st.header("2. Interactive EDA")

st.subheader("Target distribution")
target_df = (
    df_model["is_canceled"].astype(int).value_counts().sort_index().rename_axis("is_canceled").reset_index(name="count")
)
target_df["label"] = target_df["is_canceled"].map({0: "Not Canceled", 1: "Canceled"})

chart_target = (
    alt.Chart(target_df)
    .mark_bar()
    .encode(
        x=alt.X("label:N", title="Status"),
        y=alt.Y("count:Q", title="Count"),
        tooltip=["label", "count"],
    )
    .properties(height=320)
    .interactive()
)
st.altair_chart(chart_target, use_container_width=True)


st.subheader("Cancel-rate drivers")
eda_options = {
    "Lead time bin": "lead_time_bin",
    "Market segment": "market_segment",
    "Distribution channel": "distribution_channel",
    "Deposit type": "deposit_type",
    "Bulk booking": "bulk_flag",
    "Season": "season",
    "ADR bin": "adr_bin",
}
chosen_key = st.selectbox("Pilih driver:", list(eda_options.keys()), index=0)
chosen_col = eda_options[chosen_key]

cancel_driver = (
    df_model.groupby(chosen_col, dropna=False)["is_canceled"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .rename("cancel_rate")
    .reset_index()
)
cancel_driver[chosen_col] = cancel_driver[chosen_col].astype(str)

chart_cancel = (
    alt.Chart(cancel_driver)
    .mark_bar()
    .encode(
        x=alt.X("cancel_rate:Q", title="Cancel rate (%)"),
        y=alt.Y(f"{chosen_col}:N", sort="-x", title=chosen_key),
        tooltip=[
            alt.Tooltip(f"{chosen_col}:N", title=chosen_key),
            alt.Tooltip("cancel_rate:Q", format=".1f", title="Cancel rate (%)"),
        ],
    )
    .properties(height=max(260, 28 * len(cancel_driver)))
    .interactive()
)
st.altair_chart(chart_cancel, use_container_width=True)


st.subheader("Bulk booking behavior")
bk = (
    df_model.groupby("bookingID", as_index=False)
    .agg(
        bulk=("bulk_flag", "max"),
        is_canceled=("is_canceled", "max"),
        agent=("agent", "first"),
        market_segment=("market_segment", "first"),
    )
)

bulk_cancel = (
    bk.groupby("bulk")["is_canceled"].mean().mul(100).reset_index(name="cancel_rate")
)
bulk_cancel["bulk_label"] = bulk_cancel["bulk"].map({False: "Non-bulk", True: "Bulk"})

seg_bulk = (
    bk.loc[bk["bulk"], "market_segment"].astype(str).value_counts().head(10).rename_axis("market_segment").reset_index(name="count")
)
agent_bulk = (
    bk.loc[bk["bulk"], "agent"].astype(str).value_counts().head(10).rename_axis("agent").reset_index(name="count")
)

c_b1, c_b2, c_b3 = st.columns(3)
with c_b1:
    ch1 = (
        alt.Chart(bulk_cancel)
        .mark_bar()
        .encode(
            x=alt.X("bulk_label:N", title="Booking type"),
            y=alt.Y("cancel_rate:Q", title="Cancel rate (%)"),
            tooltip=["bulk_label", alt.Tooltip("cancel_rate:Q", format=".1f")],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(ch1, use_container_width=True)
with c_b2:
    ch2 = (
        alt.Chart(seg_bulk)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("market_segment:N", sort="-x", title="Segment"),
            tooltip=["market_segment", "count"],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(ch2, use_container_width=True)
with c_b3:
    ch3 = (
        alt.Chart(agent_bulk)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("agent:N", sort="-x", title="Agent"),
            tooltip=["agent", "count"],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(ch3, use_container_width=True)


st.subheader("ADR drivers & correlation")
adr_options = {
    "Season": "season",
    "Lead time bin": "lead_time_bin",
    "Market segment": "market_segment",
    "Mismatch": "room_mismatch_flag",
}
adr_choice = st.selectbox("Pilih driver ADR:", list(adr_options.keys()), index=0)
adr_col = adr_options[adr_choice]

adr_group = (
    df_model.groupby(adr_col, dropna=False)["adr"]
    .mean()
    .sort_values(ascending=False)
    .reset_index(name="avg_adr")
)
adr_group[adr_col] = adr_group[adr_col].astype(str)

adr_chart = (
    alt.Chart(adr_group)
    .mark_bar()
    .encode(
        x=alt.X("avg_adr:Q", title="Average ADR"),
        y=alt.Y(f"{adr_col}:N", sort="-x", title=adr_choice),
        tooltip=[alt.Tooltip(f"{adr_col}:N", title=adr_choice), alt.Tooltip("avg_adr:Q", format=".2f")],
    )
    .properties(height=max(240, 26 * len(adr_group)))
    .interactive()
)

scatter_df = df_model[["stay_nights", "adr"]].copy()
scatter_chart = (
    alt.Chart(scatter_df)
    .mark_circle(opacity=0.35)
    .encode(
        x=alt.X("stay_nights:Q", title="Stay nights"),
        y=alt.Y("adr:Q", title="ADR"),
        tooltip=[alt.Tooltip("stay_nights:Q"), alt.Tooltip("adr:Q", format=".2f")],
    )
    .properties(height=320)
    .interactive()
)

c_a1, c_a2 = st.columns(2)
with c_a1:
    st.altair_chart(adr_chart, use_container_width=True)
with c_a2:
    st.altair_chart(scatter_chart, use_container_width=True)


st.subheader("Special requests & waiting list")
family_stats = (
    df_model.assign(family=lambda d: (d["children"] + d["babies"]) > 0)
    .groupby("family")
    .agg(
        req_rate=("total_of_special_requests", lambda s: (s > 0).mean() * 100),
        wait_rate=("days_in_waiting_list", lambda s: (s > 0).mean() * 100),
        cancel_rate=("is_canceled", lambda s: s.mean() * 100),
        avg_adr=("adr", "mean"),
    )
    .reset_index()
)
family_stats["group"] = family_stats["family"].map({False: "Non-family", True: "Family"})

metric_pick = st.selectbox(
    "Pilih metrik group analysis:",
    ["req_rate", "wait_rate", "cancel_rate", "avg_adr"],
    index=0,
)

family_chart = (
    alt.Chart(family_stats)
    .mark_bar()
    .encode(
        x=alt.X("group:N", title="Group"),
        y=alt.Y(f"{metric_pick}:Q", title=metric_pick),
        tooltip=["group", alt.Tooltip(f"{metric_pick}:Q", format=".2f")],
    )
    .properties(height=320)
    .interactive()
)
st.altair_chart(family_chart, use_container_width=True)

# MODELING
st.header("3. Modeling")
st.success(f"Best model by ROC-AUC: {best_name}")

show_cmp = cmp.copy()
for c in ["cv_best_roc_auc", "ROC_AUC", "PR_AUC", "Brier", "LogLoss", "Precision@0.5", "Recall@0.5", "F1@0.5"]:
    if c in show_cmp.columns:
        show_cmp[c] = show_cmp[c].astype(float).round(4)

st.dataframe(show_cmp, use_container_width=True)

metric_long = show_cmp[["model", "ROC_AUC", "PR_AUC", "Brier", "LogLoss"]].melt(
    id_vars="model", var_name="metric", value_name="value"
)

metric_selector = st.selectbox(
    "Pilih metrik perbandingan model:",
    ["ROC_AUC", "PR_AUC", "Brier", "LogLoss"],
    index=0,
)

metric_plot_df = metric_long[metric_long["metric"] == metric_selector].copy()
sort_order = "-x" if metric_selector in ["ROC_AUC", "PR_AUC"] else "x"

chart_model_cmp = (
    alt.Chart(metric_plot_df)
    .mark_bar()
    .encode(
        x=alt.X("value:Q", title=metric_selector),
        y=alt.Y("model:N", sort=sort_order, title="Model"),
        tooltip=["model", alt.Tooltip("value:Q", format=".4f")],
    )
    .properties(height=280)
    .interactive()
)
st.altair_chart(chart_model_cmp, use_container_width=True)

# FEATURE IMPORTANCE
st.header("4. Feature Importance")

fi_intrinsic = intrinsic_importance(best_pipe)
fi_perm = grouped_permutation_importance(best_pipe, X_test, y_test)

if fi_intrinsic is not None:
    st.subheader("Intrinsic importance / coefficients")
    chart_int = (
        alt.Chart(fi_intrinsic.head(20))
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".6f")],
        )
        .properties(height=440)
        .interactive()
    )
    st.altair_chart(chart_int, use_container_width=True)

st.subheader("Permutation importance")
chart_perm = (
    alt.Chart(fi_perm.head(20))
    .mark_bar()
    .encode(
        x=alt.X("perm_importance:Q", title="Δ Average Precision"),
        y=alt.Y("feature:N", sort="-x", title="Feature"),
        tooltip=["feature", alt.Tooltip("perm_importance:Q", format=".6f")],
    )
    .properties(height=440)
    .interactive()
)
st.altair_chart(chart_perm, use_container_width=True)

# THRESHOLD TUNING + DIAGNOSTICS
st.header("5. Threshold Tuning & Diagnostics")

p_test = best_pipe.predict_proba(X_test)[:, 1]
threshold = st.slider("Threshold probability", min_value=0.05, max_value=0.95, value=0.50, step=0.01)
y_pred = (p_test >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
metrics_thr = pd.DataFrame(
    {
        "metric": ["Precision", "Recall", "F1", "TP", "FP", "FN", "TN"],
        "value": [
            precision_score(y_test, y_pred, zero_division=0),
            recall_score(y_test, y_pred, zero_division=0),
            f1_score(y_test, y_pred, zero_division=0),
            tp,
            fp,
            fn,
            tn,
        ],
    }
)
st.dataframe(metrics_thr, use_container_width=True)

prob_df = pd.DataFrame({"prob": p_test, "actual": y_test.values})
prob_chart = (
    alt.Chart(prob_df)
    .mark_bar()
    .encode(
        x=alt.X("prob:Q", bin=alt.Bin(maxbins=30), title="Predicted probability"),
        y=alt.Y("count():Q", title="Count"),
        color=alt.Color("actual:N", title="Actual"),
        tooltip=[alt.Tooltip("count():Q", title="Count")],
    )
    .properties(height=320)
    .interactive()
)
st.altair_chart(prob_chart, use_container_width=True)


diag = pd.DataFrame({"actual": y_test.values, "prob": p_test})
calib_df = (
    diag.assign(bin=pd.cut(diag["prob"], bins=np.linspace(0, 1, 11), include_lowest=True))
    .groupby("bin", observed=False)
    .agg(mean_prob=("prob", "mean"), actual_rate=("actual", "mean"), n=("actual", "size"))
    .reset_index()
    .dropna()
)

line_actual = (
    alt.Chart(calib_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("mean_prob:Q", title="Mean predicted probability"),
        y=alt.Y("actual_rate:Q", title="Observed cancel rate"),
        tooltip=[
            alt.Tooltip("mean_prob:Q", format=".3f"),
            alt.Tooltip("actual_rate:Q", format=".3f"),
            "n:Q",
        ],
    )
)
ref_df = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
ref_line = alt.Chart(ref_df).mark_line(strokeDash=[6, 4]).encode(x="x:Q", y="y:Q")
st.altair_chart((line_actual + ref_line).interactive().properties(height=320), use_container_width=True)

# PDP-LITE
st.header("6. PDP Lite")

num_candidates = [
    c for c in ["lead_time", "adr", "stay_nights", "total_of_special_requests", "previous_cancellations"] if c in X_test.columns
]

if num_candidates:
    pdp_feature = st.selectbox("Pilih fitur numerik untuk PDP:", num_candidates, index=0)
    x_ref = X_test.copy()
    lo, hi = np.quantile(pd.to_numeric(x_ref[pdp_feature], errors="coerce"), [0.05, 0.95])
    grid = np.linspace(lo, hi, 30)
    pdp_vals = []
    for v in grid:
        x_tmp = x_ref.copy()
        x_tmp[pdp_feature] = v
        pdp_vals.append(best_pipe.predict_proba(x_tmp)[:, 1].mean())

    pdp_df = pd.DataFrame({pdp_feature: grid, "pred_prob": pdp_vals})
    pdp_chart = (
        alt.Chart(pdp_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{pdp_feature}:Q", title=pdp_feature),
            y=alt.Y("pred_prob:Q", title="Average predicted cancel probability"),
            tooltip=[alt.Tooltip(f"{pdp_feature}:Q", format=".3f"), alt.Tooltip("pred_prob:Q", format=".4f")],
        )
        .properties(height=340)
        .interactive()
    )
    st.altair_chart(pdp_chart, use_container_width=True)
else:
    st.info("Tidak ada fitur numerik yang cocok untuk PDP.")

# SINGLE BOOKING SIMULATOR
st.header("7. Booking Cancellation Simulator")

sim_cols = [c for c in X_train.columns if c in X_test.columns]
default_row = X_train.mode(dropna=True).iloc[0].copy()

with st.form("sim_form"):
    st.caption("Isi beberapa field utama untuk lihat probabilitas cancel.")
    user_input = {}

    picked = [
        c
        for c in [
            "hotel",
            "lead_time",
            "arrival_date_month",
            "stays_in_week_nights",
            "stays_in_weekend_nights",
            "adults",
            "children",
            "babies",
            "meal",
            "market_segment",
            "distribution_channel",
            "is_repeated_guest",
            "previous_cancellations",
            "previous_bookings_not_canceled",
            "deposit_type",
            "customer_type",
            "adr",
            "required_car_parking_spaces",
            "total_of_special_requests",
            "bulk_3p_rooms",
            "season",
            "lead_time_bin",
            "adr_bin",
            "family_flag",
            "req_flag",
            "car_flag",
        ]
        if c in sim_cols
    ]

    for col in picked:
        s = X_train[col]
        if pd.api.types.is_bool_dtype(s):
            user_input[col] = st.selectbox(col, [False, True], index=int(bool(default_row.get(col, False))))
        elif pd.api.types.is_numeric_dtype(s):
            v = float(default_row.get(col, s.median()))
            if s.dropna().empty:
                user_input[col] = st.number_input(col, value=float(v))
            else:
                user_input[col] = st.number_input(col, value=float(v))
        else:
            opts = sorted([str(x) for x in s.dropna().astype(str).unique().tolist()])
            default_val = str(default_row.get(col, opts[0] if opts else ""))
            default_idx = opts.index(default_val) if default_val in opts else 0
            user_input[col] = st.selectbox(col, opts, index=default_idx)

    submitted = st.form_submit_button("Predict")

if submitted:
    sim_row = default_row.copy()
    for k, v in user_input.items():
        sim_row[k] = v
    sim_df = pd.DataFrame([sim_row])[X_train.columns]
    prob = float(best_pipe.predict_proba(sim_df)[:, 1][0])
    pred_label = "Canceled" if prob >= threshold else "Not Canceled"

    st.metric("Predicted cancel probability", f"{prob:.2%}")
    st.write(f"Predicted class at threshold {threshold:.2f}: **{pred_label}**")

    gauge_df = pd.DataFrame(
        {
            "label": ["Cancel", "Remaining"],
            "value": [prob, 1 - prob],
        }
    )
    gauge = (
        alt.Chart(gauge_df)
        .mark_arc(innerRadius=70)
        .encode(
            theta="value:Q",
            color=alt.Color("label:N", title="Legend"),
            tooltip=["label", alt.Tooltip("value:Q", format=".2%")],
        )
        .properties(width=320, height=320)
        .interactive()
    )
    st.altair_chart(gauge, use_container_width=False)

# FINAL NOTES
st.header("8. Notes")
st.markdown(
    """
- App ini mengikuti alur utama dari file Vertopal: cleaning → split room-level → feature engineering → EDA → klasifikasi cancel.
- Karena target adalah `is_canceled`, evaluasi utama memakai metrik klasifikasi seperti ROC-AUC, PR-AUC, Brier, dan LogLoss.
- Semua visual utama sudah diubah ke Altair supaya interaktif.
"""
)
