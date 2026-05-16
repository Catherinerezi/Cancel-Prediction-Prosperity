# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# PAGE CONFIG
st.set_page_config(page_title="Hotel Cancellation Prediction", layout="wide")
alt.data_transformers.disable_max_rows()

def _apply_altair_theme():
    theme = {
        "config": {
            "view": {"strokeOpacity": 0},
            "axis": {
                "labelLimit": 260,
                "titlePadding": 10,
                "labelPadding": 6,
                "tickSize": 3,
            },
            "title": {"fontSize": 16, "anchor": "start"},
            "legend": {"labelLimit": 260},
        }
    }
    try:
        alt.themes.register("hotel_manual_app_theme_v2", lambda: theme)
    except Exception:
        pass
    alt.themes.enable("hotel_manual_app_theme_v2")

_apply_altair_theme()

st.title("🏨 Hotel Booking Cancellation Prediction")
with st.expander("🎯 Tujuan", expanded=True):
    st.markdown(
        """
App ini mengikuti aturan eksekusi manual penuh dengan semua proses berat hanya jalan setelah tombol diklik.
"""
    )

# HELPERS
def reset_pipeline_state(clear_data: bool = False) -> None:
    keys = [
        "df_clean",
        "df_room",
        "df_model",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "model_pipe",
        "model_metrics",
        "fi_intrinsic",
        "pdp_candidates",
        "split_preview_before",
    ]
    if clear_data:
        keys += ["df_raw"]
    for key in keys:
        st.session_state.pop(key, None)


@st.cache_data(show_spinner=False)
def read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

def get_categorical_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if str(df[c].dtype) in ["object", "string", "category", "bool"]:
            cols.append(c)
    return cols

def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()

@st.cache_data(show_spinner=False)
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]

    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = (
            out[c]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    if "reservation_status_date" in out.columns:
        out["reservation_status_date"] = pd.to_datetime(
            out["reservation_status_date"], errors="coerce"
        )

    # drop kolom dengan missing > 20%
    missing_percentage = out.isna().mean() * 100
    cols_to_drop = missing_percentage[missing_percentage > 20].index.tolist()
    if cols_to_drop:
        out = out.drop(columns=cols_to_drop, errors="ignore")

    for c in ["agent", "country", "children"]:
        if c in out.columns:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = pd.to_numeric(out[c], errors="coerce")
                out[c] = out[c].fillna(out[c].median(skipna=True))
            else:
                m = out[c].mode(dropna=True)
                if not m.empty:
                    out[c] = out[c].fillna(m.iat[0])

    if "is_canceled" in out.columns:
        out["is_canceled"] = (
            pd.to_numeric(out["is_canceled"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    return out

def apply_filters(df: pd.DataFrame, filter_cols: list[str], selections: dict[str, list]) -> pd.DataFrame:
    out = df.copy()
    for c in filter_cols:
        vals = selections.get(c, [])
        if vals:
            out = out[out[c].astype(str).isin([str(v) for v in vals])]
    return out

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
    out = []
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

@st.cache_data(show_spinner=False)
def build_room_level_dataset(df: pd.DataFrame, max_rows: int = 300) -> pd.DataFrame:
    work = (
        df.head(int(max_rows))
        .copy()
        .reset_index(drop=False)
        .rename(columns={"index": "source_row_id"}))
    
    records: list[dict] = []

    for _, row in work.iterrows():
        split_result = split_rooms_cap4(row)
        for i, rec in enumerate(split_result, start=1):
            rec["source_row_id"] = row["source_row_id"]
            rec["room_no"] = i
            rec["split_room_count"] = len(split_result)
            rec["was_split"] = len(split_result) > 1
            records.append(rec)

    out = pd.DataFrame(records).reset_index(drop=True)

    if "bookingID" in out.columns:
        out["bookingID"] = out["bookingID"].astype(str)
        out["Invoice_ID"] = np.arange(1, len(out) + 1, dtype=int)
        out["rooms_in_booking"] = out.groupby("bookingID")["Invoice_ID"].transform("count")
        out["bulk_3p_rooms"] = out["rooms_in_booking"] >= 3

    return out

@st.cache_data(show_spinner=False)
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in [
        "children", "babies", "adults",
        "stays_in_week_nights", "stays_in_weekend_nights",
        "adr", "lead_time"
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if {"children", "babies"}.issubset(out.columns):
        out["minors"] = out["children"].fillna(0) + out["babies"].fillna(0)
    if {"adults", "minors"}.issubset(out.columns):
        out["party_size"] = out["adults"].fillna(0) + out["minors"].fillna(0)
    if {"stays_in_week_nights", "stays_in_weekend_nights"}.issubset(out.columns):
        out["stay_nights"] = out["stays_in_week_nights"].fillna(0) + out["stays_in_weekend_nights"].fillna(0)
    if {"stays_in_weekend_nights", "stay_nights"}.issubset(out.columns):
        out["weekend_ratio"] = np.where(
            out["stay_nights"] > 0,
            out["stays_in_weekend_nights"] / out["stay_nights"],
            0.0
        )
    if {"adr", "stay_nights"}.issubset(out.columns):
        out["room_revenue"] = out["adr"].fillna(0) * out["stay_nights"].fillna(0)
    if {"bookingID", "room_revenue"}.issubset(out.columns):
        booking_rev = out.groupby("bookingID")["room_revenue"].sum().rename("booking_revenue")
        out = out.merge(booking_rev, on="bookingID", how="left")

    if "arrival_date_month" in out.columns:
        season_map = {
            "December": "High",
            "January": "High",
            "February": "High",
            "June": "Peak",
            "July": "Peak",
            "August": "Peak",
        }
        out["season"] = out["arrival_date_month"].map(season_map).fillna("Shoulder")

    if "lead_time" in out.columns:
        out["lead_time_bin"] = pd.cut(
            out["lead_time"],
            bins=[-1, 7, 30, 90, 180, 9999],
            labels=["<=7d", "8-30d", "31-90d", "91-180d", ">180d"],
        )

    if "adr" in out.columns:
        try:
            out["adr_bin"] = pd.qcut(
                out["adr"].rank(method="first"),
                q=5,
                labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
            )
        except Exception:
            out["adr_bin"] = "Q3"

    if {"children", "babies"}.issubset(out.columns):
        out["family_flag"] = (out["children"].fillna(0) + out["babies"].fillna(0)) > 0
    if "total_of_special_requests" in out.columns:
        out["req_flag"] = pd.to_numeric(out["total_of_special_requests"], errors="coerce").fillna(0) > 0
    if "required_car_parking_spaces" in out.columns:
        out["car_flag"] = pd.to_numeric(out["required_car_parking_spaces"], errors="coerce").fillna(0) > 0

    if {"adults", "minors"}.issubset(out.columns):
        out["couple_flag"] = (out["adults"].fillna(0).eq(2)) & (out["minors"].fillna(0).eq(0))
        out["solo_flag"] = (out["adults"].fillna(0).eq(1)) & (out["minors"].fillna(0).eq(0))
    if "rooms_in_booking" in out.columns:
        out["bulk_flag"] = out["rooms_in_booking"] >= 3
    
    if {"agent", "bookingID", "rooms_in_booking"}.issubset(out.columns):
        booking_bulk = (
            out.drop_duplicates("bookingID")
            .assign(_bulk=lambda d: d["rooms_in_booking"] >= 3)
            .groupby("agent")["_bulk"]
            .sum()
            .rename("bulk_bookings_by_agent")
        )
        out = out.merge(booking_bulk, on="agent", how="left")
        out["bulk_booker_agent_flag"] = out["bulk_bookings_by_agent"].fillna(0).ge(3)
    
    if "days_in_waiting_list" in out.columns:
        out["waiting_list_flag"] = pd.to_numeric(out["days_in_waiting_list"], errors="coerce").fillna(0) > 0
    
    if {"market_segment", "distribution_channel"}.issubset(out.columns):
        out["seg_x_chan"] = out["market_segment"].astype(str) + " | " + out["distribution_channel"].astype(str)
    
    if {"reserved_room_type", "assigned_room_type"}.issubset(out.columns):
        out["room_mismatch_flag"] = out["reserved_room_type"].astype(str) != out["assigned_room_type"].astype(str)
    
    if "booking_changes" in out.columns:
        out["changes_flag"] = pd.to_numeric(out["booking_changes"], errors="coerce").fillna(0) > 0

    return out

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = get_numeric_cols(X)
    cat_cols = get_categorical_cols(X)

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", ohe),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

def prepare_model_data(df: pd.DataFrame, max_rows: int = 1000):
    target = "is_canceled"
    if target not in df.columns:
        raise ValueError("Kolom is_canceled tidak ditemukan.")

    work = df.copy()
    if len(work) > max_rows:
        work = work.sample(max_rows, random_state=42).copy()

    if "reservation_status_date" in work.columns:
        train_df = work[work["reservation_status_date"] < "2019-01-01"].copy()
        test_df = work[work["reservation_status_date"] >= "2019-01-01"].copy()
        if len(train_df) < 50 or len(test_df) < 20:
            split_ix = int(len(work) * 0.8)
            train_df = work.iloc[:split_ix].copy()
            test_df = work.iloc[split_ix:].copy()
    else:
        split_ix = int(len(work) * 0.8)
        train_df = work.iloc[:split_ix].copy()
        test_df = work.iloc[split_ix:].copy()

    drop_cols = [target, "Invoice_ID", "bookingID", "reservation_status_date", "reservation_status"]
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")
    y_train = train_df[target].astype(int)
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors="ignore")
    y_test = test_df[target].astype(int)
    return X_train, X_test, y_train, y_test

@st.cache_resource(show_spinner=False)
def train_fast_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    preprocessor = make_preprocessor(X_train)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1200, solver="liblinear", class_weight="balanced", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe

def evaluate_model(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict:
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_test, prob) if len(np.unique(y_test)) > 1 else np.nan,
        "brier": brier_score_loss(y_test, prob),
        "logloss": log_loss(y_test, prob),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }

def get_feature_names(pipe: Pipeline) -> list[str]:
    ct = pipe.named_steps["preprocess"]
    try:
        return list(ct.get_feature_names_out())
    except Exception:
        names = []
        for name, trans, cols in ct.transformers_:
            if name == "num":
                names.extend(list(cols))
            elif name == "cat":
                try:
                    ohe = trans.named_steps["ohe"]
                    names.extend(list(ohe.get_feature_names_out(cols)))
                except Exception:
                    names.extend(list(cols))
        return names

def get_intrinsic_importance(pipe: Pipeline) -> pd.DataFrame | None:
    model = pipe.named_steps["model"]
    feat_names = get_feature_names(pipe)
    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        fi = pd.DataFrame({"feature": feat_names, "importance": np.abs(coef)})
        return fi.sort_values("importance", ascending=False).reset_index(drop=True)
    return None

# no cache here because sklearn Pipeline is unhashable for cache_data
def compute_pdp_1d(pipe: Pipeline, X_ref: pd.DataFrame, feat: str, grid_size: int = 20) -> pd.DataFrame:
    s = pd.to_numeric(X_ref[feat], errors="coerce")
    vals = np.linspace(s.quantile(0.05), s.quantile(0.95), grid_size)
    pdp_vals = []
    for v in vals:
        temp = X_ref.copy()
        temp[feat] = v
        pdp_vals.append(pipe.predict_proba(temp)[:, 1].mean())
    return pd.DataFrame({feat: vals, "pred_prob": pdp_vals})

# SIDEBAR
with st.sidebar:
    st.header("Data source")
    with st.form("upload_form"):
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="manual_upload_csv")
        load_btn = st.form_submit_button("Load data")

    st.header("Manual limits")
    total_rows_loaded = len(st.session_state["df_raw"]) if "df_raw" in st.session_state else 5000
    preview_rows = st.number_input(
        "Preview rows",
        min_value=20,
        max_value=min(total_rows_loaded, 5000),
        value=min(300, total_rows_loaded),
        step=20)

    split_rows = st.number_input(
        "Rows for split room",
        min_value=20,
        max_value=total_rows_loaded,
        value=total_rows_loaded,
        step=1000)

    model_rows = st.number_input(
        "Rows for modeling",
        min_value=100,
        max_value=total_rows_loaded,
        value=min(10000, total_rows_loaded),
        step=1000)

    st.header("Global filters")

    filter_source = st.session_state.get("df_clean", st.session_state.get("df_raw"))

    if filter_source is not None:
        filter_candidates = get_categorical_cols(filter_source)

        default_filter_cols = [
            c for c in [
                "hotel",
                "arrival_date_month",
                "market_segment",
                "distribution_channel",
                "deposit_type",
                "customer_type",
            ] if c in filter_candidates
        ]

        with st.form("global_filter_form"):
            selected_filter_cols = st.multiselect(
                "Pilih kolom filter",
                options=filter_candidates,
                default=default_filter_cols[:2]
            )

            global_filter_selections = {}
            for c in selected_filter_cols:
                opts = sorted(filter_source[c].dropna().astype(str).unique().tolist())
                global_filter_selections[c] = st.multiselect(
                    f"Value untuk {c}",
                    options=opts,
                    default=[]
                )

            apply_global_filter_btn = st.form_submit_button("Apply global filters")
            reset_global_filter_btn = st.form_submit_button("Reset global filters")

        if apply_global_filter_btn:
            st.session_state["active_filters"] = {
                k: v for k, v in global_filter_selections.items() if v
            }

        if reset_global_filter_btn:
            st.session_state["active_filters"] = {}

if load_btn:
    reset_pipeline_state(clear_data=True)
    if uploaded_file is None:
        st.sidebar.warning("Upload CSV dulu.")
    else:
        try:
            st.session_state["df_raw"] = read_uploaded_csv(uploaded_file.getvalue())
            st.sidebar.success("Data loaded.")
        except Exception as e:
            st.sidebar.error(f"Gagal load data: {e}")

if "df_raw" not in st.session_state:
    st.info("Upload file CSV kamu sendiri dari sidebar, lalu klik **Load data**.")
    st.stop()

df_raw = st.session_state["df_raw"]
st.caption(f"Rows loaded: **{len(df_raw):,}**")

df_base = st.session_state.get("df_clean", df_raw)
active_filters = st.session_state.get("active_filters", {})

if active_filters:
    df_active = apply_filters(df_base, list(active_filters.keys()), active_filters)
else:
    df_active = df_base.copy()

if active_filters:
    active_filter_text = " | ".join(
        [f"{k}: {', '.join(map(str, v))}" for k, v in active_filters.items()]
    )
    st.caption(f"Active filters → {active_filter_text}")
else:
    st.caption("Active filters → none")

st.caption(f"Rows in active dataset: **{len(df_active):,}**")

# TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Overview",
        "Cleaning",
        "Split Room",
        "EDA",
        "Feature Engineering",
        "Modeling",
        "Importance + PDP + Simulator",
    ]
)

# TAB 1 - OVERVIEW / DATA UNDERSTANDING
with tab1:
    st.subheader("Data Understanding")
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        st.write("**Shape & Preview**")
        st.write(df_raw.shape)
        st.dataframe(df_raw.head(int(preview_rows)), use_container_width=True)

    with c2:
        st.write("**Dtypes**")
        dtype_df = pd.DataFrame(
            {"column": df_raw.columns, "dtype": [str(df_raw[c].dtype) for c in df_raw.columns]}
        )
        st.dataframe(dtype_df, use_container_width=True)

    with c3:
        st.write("**Basic Summary**")
        summary_df = pd.DataFrame(
            {
                "metric": ["rows", "columns", "duplicates"],
                "value": [len(df_raw), df_raw.shape[1], int(df_raw.duplicated().sum())],
            }
        )
        st.dataframe(summary_df, use_container_width=True)

    st.markdown("### Missing Value (%)")
    missing_df = (df_raw.isna().mean() * 100).sort_values(ascending=False).round(2).reset_index()
    missing_df.columns = ["column", "missing_pct"]

    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.dataframe(missing_df, use_container_width=True)
    with cc2:
        miss_chart = (
            alt.Chart(missing_df)
            .mark_bar()
            .encode(
                x=alt.X("missing_pct:Q", title="% Missing"),
                y=alt.Y("column:N", sort="-x", title="Column"),
                tooltip=["column", alt.Tooltip("missing_pct:Q", format=".2f")],
            )
            .properties(height=max(220, 20 * len(missing_df)))
            .interactive()
        )
        st.altair_chart(miss_chart, use_container_width=True)

    num_cols = get_numeric_cols(df_raw)
    if num_cols:
        st.markdown("### Describe (Numeric)")
        st.dataframe(df_raw[num_cols].describe().T, use_container_width=True)

    st.markdown("### Unique Count per Column")
    uniq_df = pd.DataFrame(
        {
            "column": df_raw.columns,
            "n_unique": [df_raw[c].nunique(dropna=True) for c in df_raw.columns],
        }
    ).sort_values("n_unique", ascending=False)
    st.dataframe(uniq_df, use_container_width=True)

    if "is_canceled" in df_raw.columns:
        st.markdown("### Target Distribution")
        target_df = df_raw["is_canceled"].value_counts().sort_index().reset_index()
        target_df.columns = ["is_canceled", "count"]
        target_df["label"] = target_df["is_canceled"].map({0: "Not Canceled", 1: "Canceled"})
        target_chart = (
            alt.Chart(target_df)
            .mark_bar()
            .encode(
                x=alt.X("label:N", title="Status"),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["label", "count"],
            )
            .properties(height=260)
        )
        st.altair_chart(target_chart, use_container_width=True)

# TAB 2 - CLEANING
with tab2:
    st.subheader("Cleaning")
    st.caption("Menampilkan kondisi sebelum cleaning dan sesudah cleaning.")
    before_clean_df = pd.DataFrame(
        {
            "metric": ["rows", "columns", "duplicates"],
            "value": [len(df_raw), df_raw.shape[1], int(df_raw.duplicated().sum())],
        }
    )
    before_missing = (df_raw.isna().sum()).sort_values(ascending=False).reset_index()
    before_missing.columns = ["column", "missing_count"]

    st.markdown("### Before Cleaning")
    bc1, bc2 = st.columns([1, 1])
    with bc1:
        st.dataframe(before_clean_df, use_container_width=True)
    with bc2:
        st.dataframe(before_missing, use_container_width=True)

    if st.button("Run cleaning", key="btn_clean"):
        cleaned = basic_clean(df_raw)
        reset_pipeline_state(clear_data=False)
        st.session_state["df_clean"] = cleaned
        st.session_state["active_filters"] = {}
        st.success("Cleaning selesai.")

    if "df_clean" in st.session_state:
        df_clean = st.session_state["df_clean"]

        after_clean_df = pd.DataFrame(
            {
                "metric": ["rows", "columns", "duplicates"],
                "value": [len(df_clean), df_clean.shape[1], int(df_clean.duplicated().sum())],
            }
        )
        after_missing = (df_clean.isna().sum()).sort_values(ascending=False).reset_index()
        after_missing.columns = ["column", "missing_count"]

        st.markdown("### After Cleaning")
        ac1, ac2 = st.columns([1, 1])
        with ac1:
            st.dataframe(after_clean_df, use_container_width=True)
        with ac2:
            st.dataframe(after_missing, use_container_width=True)

        st.markdown("### Preview After Cleaning")
        st.dataframe(df_clean.head(20), use_container_width=True)

# TAB 3 - SPLIT ROOM
with tab3:
    st.subheader("Split Room")
    st.markdown(
        """
Aturan split yang dipakai:
- maksimal **4 orang per room**,
- kalau ada **children/babies**, sistem berusaha menempatkan minimal **1 adult** di room itu,
- data yang diproses hanya sebanyak limit **Rows for split room** di sidebar.
"""
    )

    if st.button("Run split room", key="btn_split"):
        source_df = df_active.copy()
        rows_to_process = min(int(split_rows), len(source_df))
        st.session_state["split_preview_before"] = (
            source_df
            .head(rows_to_process)
            .copy()
            .reset_index(drop=False)
            .rename(columns={"index": "source_row_id"}))
        st.session_state["df_room"] = build_room_level_dataset(
            source_df,
            max_rows=rows_to_process
        )
        
        st.session_state["df_model"] = add_features(st.session_state["df_room"]) 
        st.success(f"Split room + feature engineering selesai. Rows diproses: {rows_to_process:,}")

    if "df_room" in st.session_state:
        df_room = st.session_state["df_room"]
        before_split = st.session_state.get("split_preview_before")

        st.markdown("### Summary Split Room")
        s1, s2, s3 = st.columns(3)
        s1.metric("Rows before split (processed)", len(before_split) if before_split is not None else 0)
        s2.metric("Rows after split", len(df_room))
        rows_processed = len(before_split) if before_split is not None else 0
        s3.metric("Rows not processed", max(len(df_active) - rows_processed, 0))

        check_df = before_split.copy()
        check_df["total_guests"] = (
            pd.to_numeric(check_df["adults"], errors="coerce").fillna(0)
            + pd.to_numeric(check_df["children"], errors="coerce").fillna(0)
            + pd.to_numeric(check_df["babies"], errors="coerce").fillna(0)
        )

        candidate_split = check_df[check_df["total_guests"] > 4].copy()

        st.markdown("### Kandidat Row yang Berpotensi Ter-Split")
        if not candidate_split.empty:
            cols_show = [c for c in [
                "source_row_id", "bookingID", "adults", "children", "babies", "total_guests"
            ] if c in candidate_split.columns]
            st.dataframe(candidate_split[cols_show].head(20), use_container_width=True)
        else:
            st.info("Pada row yang diproses sekarang, tidak ada total guest > 4.")
        
        st.markdown("### Bukti Row yang Benar-Benar Ter-Split")
        if not isinstance(df_room, pd.DataFrame):
            st.warning("df_room belum berbentuk DataFrame.")
        elif "was_split" not in df_room.columns:
            st.warning("Kolom was_split belum tersedia di hasil split.")
        else:
            source_df_used = st.session_state.get("split_preview_before")
            if not isinstance(source_df_used, pd.DataFrame):
                st.warning("Data mentah sebelum split belum tersedia.")
            elif "source_row_id" not in source_df_used.columns:
                st.warning("Kolom source_row_id tidak ditemukan pada data sebelum split.")
            else:
                before_raw = source_df_used.copy()
        
                before_raw["total_guests"] = (
                    pd.to_numeric(before_raw["adults"], errors="coerce").fillna(0)
                    + pd.to_numeric(before_raw["children"], errors="coerce").fillna(0)
                    + pd.to_numeric(before_raw["babies"], errors="coerce").fillna(0)
                )
        
                before_raw["estimated_rooms_before"] = np.ceil(
                    before_raw["total_guests"] / 4
                ).astype(int)
        
                before_raw.loc[
                    before_raw["estimated_rooms_before"] < 1,
                    "estimated_rooms_before"
                ] = 1
        
                before_proof = before_raw.copy()
                split_only = df_room[df_room["was_split"] == True].copy()
                if before_proof.empty:
                    st.info("Pada data mentah sebelum split, tidak ada row yang membutuhkan lebih dari 1 room.")
        
                else:
                    st.markdown("#### Before Split - Row asli dari data mentah")
        
                    cols_before = [
                        c for c in [
                            "source_row_id",
                            "bookingID",
                            "adults",
                            "children",
                            "babies",
                            "total_guests",
                            "estimated_rooms_before",
                        ]
                        if c in before_proof.columns
                    ]
        
                    st.dataframe(
                        before_proof[cols_before].head(50),
                        use_container_width=True
                    )

                    st.markdown("##### Chart Before Split")
        
                    before_dist_df = pd.DataFrame({
                        "room_before_split": [1],
                        "booking_count": [before_raw["bookingID"].nunique()]
                    })
        
                    before_chart = (
                        alt.Chart(before_dist_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("room_before_split:O", title="Jumlah room sebelum split"),
                            y=alt.Y("booking_count:Q", title="Jumlah booking"),
                            tooltip=["room_before_split", "booking_count"],
                        )
                        .properties(height=300)
                        .interactive()
                    )
                    
                    st.altair_chart(before_chart, use_container_width=True)
                    st.dataframe(before_dist_df, use_container_width=True)
        
                if split_only.empty:
                    st.info("Pada hasil after split, tidak ada row yang benar-benar ter-split.")
        
                else:
                    st.markdown("#### After Split - Hasil pecahan per room")
                    cols_after = [
                        c for c in [
                            "source_row_id",
                            "bookingID",
                            "room_no",
                            "split_room_count",
                            "adults",
                            "children",
                            "babies",
                            "viol_minors_without_adult",
                        ]
                        if c in split_only.columns
                    ]
        
                    st.dataframe(
                        df_room[cols_after].head(100),
                        use_container_width=True
                    )
    
                    st.markdown("##### Chart After Split")

                    after_room_count_per_booking = df_room.groupby("bookingID").size()

                    after_dist_df = (
                        after_room_count_per_booking
                        .value_counts()
                        .sort_index()
                        .rename_axis("rooms_after_split")
                        .reset_index(name="booking_count")
                    )
                    
                    after_chart = (
                        alt.Chart(after_dist_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("rooms_after_split:O", title="Jumlah kamar setelah split"),
                            y=alt.Y("booking_count:Q", title="Jumlah bookingID"),
                            tooltip=["rooms_after_split", "booking_count"],
                        )
                        .properties(height=300)
                        .interactive()
                    )
                    
                    st.altair_chart(after_chart, use_container_width=True)
                    st.dataframe(after_dist_df, use_container_width=True)

                    st.markdown("##### Chart After Split - Hanya Booking yang Jadi Lebih dari 1 Kamar")
                    
                    after_dist_split_only = after_dist_df[after_dist_df["rooms_after_split"] > 1].copy()
                    
                    if after_dist_split_only.empty:
                        st.info("Tidak ada booking yang menjadi lebih dari 1 kamar.")
                    else:
                        after_split_only_chart = (
                            alt.Chart(after_dist_split_only)
                            .mark_bar()
                            .encode(
                                x=alt.X("rooms_after_split:O", title="Jumlah kamar setelah split"),
                                y=alt.Y("booking_count:Q", title="Jumlah bookingID"),
                                tooltip=["rooms_after_split", "booking_count"],
                            )
                            .properties(height=300)
                            .interactive()
                        )
                    
                        st.altair_chart(after_split_only_chart, use_container_width=True)
                        st.dataframe(after_dist_split_only, use_container_width=True)

# TAB 4 - EDA
with tab4:
    st.subheader("EDA")
    st.caption("EDA memakai data terakhir yang kamu proses manual.")

    if "df_model" in st.session_state:
        eda_source = st.session_state["df_model"]
    elif "df_room" in st.session_state:
        eda_source = st.session_state["df_room"]
    else:
        eda_source = df_active.copy()

    if "is_canceled" in eda_source.columns:
        target_df = eda_source["is_canceled"].value_counts().sort_index().reset_index()
        target_df.columns = ["is_canceled", "count"]
        target_df["label"] = target_df["is_canceled"].map({0: "Not Canceled", 1: "Canceled"})
        target_chart = (
            alt.Chart(target_df)
            .mark_bar()
            .encode(
                x=alt.X("label:N", title="Status"),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["label", "count"],
            )
            .properties(height=260)
        )
        st.altair_chart(target_chart, use_container_width=True)

    cat_candidates = [
        c for c in [
            "hotel",
            "arrival_date_month",
            "market_segment",
            "distribution_channel",
            "deposit_type",
            "customer_type",
            "season",
            "lead_time_bin",
            "adr_bin",
            "family_flag",
            "couple_flag",
            "solo_flag",
            "bulk_3p_rooms",
            "bulk_flag",
            "bulk_booker_agent_flag",
            "waiting_list_flag",
            "req_flag",
            "car_flag",
            "room_mismatch_flag",
            "changes_flag",
            "seg_x_chan",
        ] if c in eda_source.columns
    ]
    if cat_candidates and "is_canceled" in eda_source.columns:
        cat_sel = st.selectbox("Pilih kolom kategori", options=cat_candidates, key="eda_cat")
        rate_df = (
            eda_source.groupby(cat_sel)["is_canceled"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .reset_index(name="cancel_rate")
        )
        rate_df[cat_sel] = rate_df[cat_sel].astype(str)
        rate_chart = (
            alt.Chart(rate_df)
            .mark_bar()
            .encode(
                x=alt.X("cancel_rate:Q", title="Cancel rate (%)"),
                y=alt.Y(f"{cat_sel}:N", sort="-x", title=cat_sel),
                tooltip=[cat_sel, alt.Tooltip("cancel_rate:Q", format=".2f")],
            )
            .properties(height=max(220, 24 * len(rate_df)))
            .interactive()
        )
        st.altair_chart(rate_chart, use_container_width=True)

# TAB 5 - FEATURE ENGINEERING
with tab5:
    st.subheader("Feature Engineering")
    st.caption("Feature engineering hanya jalan setelah klik tombol.")

    if st.button("Run feature engineering", key="btn_fe"):
        source_df = st.session_state.get("df_room")
        if source_df is None:
            st.warning("Run split room dulu.")
        else:
            st.session_state["df_model"] = add_features(source_df)
            st.success("Feature engineering selesai.")

    if "df_model" in st.session_state:
        df_model = st.session_state["df_model"]
        st.dataframe(df_model.head(20), use_container_width=True)
        feature_examples = [
            c for c in [
                "minors", "party_size", "stay_nights", "weekend_ratio", "room_revenue",
                "booking_revenue", "season", "lead_time_bin", "adr_bin", "family_flag",
                "req_flag", "car_flag",
            ] if c in df_model.columns
        ]
        st.write("Contoh fitur baru:", feature_examples)

# TAB 6 - MODELING
with tab6:
    st.subheader("Modeling")
    st.caption("Modeling tidak jalan sebelum tombol diklik. Model cepat: Logistic Regression.")

    if st.button("Run model", key="btn_model"):
        source_df = st.session_state.get("df_model")
        if source_df is None:
            st.warning("Run feature engineering dulu.")
        else:
            X_train, X_test, y_train, y_test = prepare_model_data(source_df, max_rows=int(model_rows))
            pipe = train_fast_model(X_train, y_train)
            metrics = evaluate_model(pipe, X_test, y_test)
            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test
            st.session_state["model_pipe"] = pipe
            st.session_state["model_metrics"] = metrics
            st.success("Model selesai dijalankan.")

    if "model_metrics" in st.session_state:
        m = st.session_state["model_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", "N/A" if pd.isna(m["roc_auc"]) else f"{m['roc_auc']:.4f}")
        c2.metric("PR-AUC", "N/A" if pd.isna(m["pr_auc"]) else f"{m['pr_auc']:.4f}")
        c3.metric("Precision", f"{m['precision']:.4f}")
        c4.metric("Recall", f"{m['recall']:.4f}")

        cmp = pd.DataFrame([{"model": "Logistic Regression", **m}])
        st.dataframe(cmp, use_container_width=True)

# TAB 7 - IMPORTANCE + PDP + SIMULATOR
with tab7:
    st.subheader("Importance + PDP + Simulator")

    left, right = st.columns(2)
    with left:
        if st.button("Generate importance", key="btn_imp"):
            if "model_pipe" not in st.session_state:
                st.warning("Run model dulu.")
            else:
                st.session_state["fi_intrinsic"] = get_intrinsic_importance(st.session_state["model_pipe"])
                st.success("Importance selesai.")

    with right:
        if st.button("Prepare PDP", key="btn_pdp"):
            if "model_pipe" not in st.session_state:
                st.warning("Run model dulu.")
            else:
                st.session_state["pdp_candidates"] = st.session_state["X_test"].select_dtypes(include=[np.number]).columns.tolist()
                st.success("PDP siap.")

    if "fi_intrinsic" in st.session_state and st.session_state["fi_intrinsic"] is not None:
        fi = st.session_state["fi_intrinsic"].head(20)
        fi_chart = (
            alt.Chart(fi)
            .mark_bar()
            .encode(
                x=alt.X("importance:Q", title="Importance / |Coefficient|"),
                y=alt.Y("feature:N", sort="-x"),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".6f")],
            )
            .properties(height=380)
            .interactive()
        )
        st.altair_chart(fi_chart, use_container_width=True)

    if "pdp_candidates" in st.session_state and st.session_state["pdp_candidates"]:
        feat = st.selectbox("Pilih fitur numerik untuk PDP", options=st.session_state["pdp_candidates"], key="pdp_feat")
        pdp_df = compute_pdp_1d(st.session_state["model_pipe"], st.session_state["X_test"].copy(), feat, grid_size=20)
        pdp_chart = (
            alt.Chart(pdp_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(f"{feat}:Q", title=feat),
                y=alt.Y("pred_prob:Q", title="Predicted cancel probability"),
                tooltip=[alt.Tooltip(feat, format=".3f"), alt.Tooltip("pred_prob:Q", format=".4f")],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(pdp_chart, use_container_width=True)

    st.markdown("### Simulator")
    if "model_pipe" not in st.session_state:
        st.info("Run model dulu.")
    else:
        X_train = st.session_state["X_train"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        pipe = st.session_state["model_pipe"]
        default_row = X_train.mode(dropna=True).iloc[0].copy()

        with st.form("sim_form"):
            user_input = {}
            picked = [
                c for c in [
                    "hotel", "lead_time", "arrival_date_month", "stays_in_week_nights",
                    "stays_in_weekend_nights", "adults", "children", "babies", "meal",
                    "market_segment", "distribution_channel", "is_repeated_guest",
                    "previous_cancellations", "previous_bookings_not_canceled", "deposit_type",
                    "customer_type", "adr", "required_car_parking_spaces",
                    "total_of_special_requests", "bulk_3p_rooms", "season", "lead_time_bin",
                    "adr_bin", "family_flag", "req_flag", "car_flag",
                ] if c in X_train.columns
            ]

            for col in picked:
                s = X_train[col]
                if pd.api.types.is_bool_dtype(s):
                    user_input[col] = st.selectbox(col, [False, True], index=int(bool(default_row.get(col, False))))
                elif pd.api.types.is_numeric_dtype(s):
                    user_input[col] = st.number_input(col, value=float(default_row.get(col, s.median())))
                else:
                    opts = sorted([str(x) for x in s.dropna().astype(str).unique().tolist()])
                    default_val = str(default_row.get(col, opts[0] if opts else ""))
                    default_idx = opts.index(default_val) if default_val in opts else 0
                    user_input[col] = st.selectbox(col, opts, index=default_idx)

            threshold = st.slider("Threshold probability", 0.05, 0.95, 0.50, 0.01, key="threshold_sim")
            submitted = st.form_submit_button("Predict")

        prob_test = pipe.predict_proba(X_test)[:, 1]
        pred_test = (prob_test >= threshold).astype(int)
        cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        diag_df = pd.DataFrame(
            {
                "metric": ["Precision", "Recall", "F1", "TP", "FP", "FN", "TN"],
                "value": [
                    precision_score(y_test, pred_test, zero_division=0),
                    recall_score(y_test, pred_test, zero_division=0),
                    f1_score(y_test, pred_test, zero_division=0),
                    tp, fp, fn, tn,
                ],
            }
        )
        st.dataframe(diag_df, use_container_width=True)

        if submitted:
            sim_row = default_row.copy()
            for k, v in user_input.items():
                sim_row[k] = v
            sim_df = pd.DataFrame([sim_row])[X_train.columns]
            prob = float(pipe.predict_proba(sim_df)[:, 1][0])
            pred_label = "Canceled" if prob >= threshold else "Not Canceled"

            c1, c2 = st.columns(2)
            c1.metric("Predicted cancel probability", f"{prob:.2%}")
            c2.metric("Predicted class", pred_label)

            donut_df = pd.DataFrame({"label": ["Cancel", "Remaining"], "value": [prob, 1 - prob]})
            donut_chart = (
                alt.Chart(donut_df)
                .mark_arc(innerRadius=70)
                .encode(
                    theta="value:Q",
                    color="label:N",
                    tooltip=["label", alt.Tooltip("value:Q", format=".2%")],
                )
                .properties(width=300, height=300)
                .interactive()
            )
            st.altair_chart(donut_chart, use_container_width=False)
