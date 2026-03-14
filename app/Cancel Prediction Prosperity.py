# -*- coding: utf-8 -*-
"""
app.py
Strict manual-execution Hotel Booking Cancellation app

Rules implemented:
- App startup: UI only, no dataset processing
- User must manually load dataset from sidebar
- After load: still no cleaning / filtering / split / FE / modeling automatically
- Every heavy step runs only when its button is clicked
- Altair only
"""

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


# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
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
    alt.themes.register("streamlit_safe_hotel_strict", lambda: theme)
    alt.themes.enable("streamlit_safe_hotel_strict")


_apply_altair_theme()

st.title("🏨 Hotel Booking Cancellation Prediction")
with st.expander("🎯 Tujuan", expanded=True):
    st.markdown(
        """
App ini dibuat dengan eksekusi manual penuh:
- startup hanya menampilkan UI,
- data dimasukkan manual dari sidebar,
- cleaning, filtering, split room, feature engineering, modeling, importance, PDP, dan simulator hanya jalan saat tombolnya diklik.
"""
    )


# ---------------------------------------------------------------------
# STATE HELPERS
# ---------------------------------------------------------------------
def clear_runtime(clear_data: bool = False):
    keys = [
        "df_clean",
        "df_filtered",
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
        "filter_applied",
    ]
    if clear_data:
        keys += ["df_raw", "data_source"]
    for k in keys:
        st.session_state.pop(k, None)


# ---------------------------------------------------------------------
# CACHED READERS / TRANSFORMS
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes=None, repo_path=None, source="upload"):
    if source == "upload":
        return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    return pd.read_csv(repo_path, low_memory=False)


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
        out["reservation_status_date"] = pd.to_datetime(out["reservation_status_date"], errors="coerce")

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
        out["is_canceled"] = pd.to_numeric(out["is_canceled"], errors="coerce").fillna(0).astype(int)

    return out


def apply_filters(df: pd.DataFrame, filter_cols: list[str], selections: dict[str, list]) -> pd.DataFrame:
    out = df.copy()
    for c in filter_cols:
        chosen = selections.get(c, [])
        if chosen:
            out = out[out[c].astype(str).isin([str(v) for v in chosen])]
    return out


def split_rooms_cap4(row: pd.Series) -> list[dict]:
    adults = int(row.get("adults", 0) or 0)
    children = int(row.get("children", 0) or 0)
    babies = int(row.get("babies", 0) or 0)

    minors = children + babies
    rooms = []
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
def build_room_level_dataset(df: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame:
    work = df.head(max_rows).copy()
    records = []
    for _, row in work.iterrows():
        records.extend(split_rooms_cap4(row))

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

    for c in ["children", "babies", "adults", "stays_in_week_nights", "stays_in_weekend_nights", "adr", "lead_time"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if {"children", "babies"}.issubset(out.columns):
        out["minors"] = out["children"].fillna(0) + out["babies"].fillna(0)
    if {"adults", "minors"}.issubset(out.columns):
        out["party_size"] = out["adults"].fillna(0) + out["minors"].fillna(0)
    if {"stays_in_week_nights", "stays_in_weekend_nights"}.issubset(out.columns):
        out["stay_nights"] = out["stays_in_week_nights"].fillna(0) + out["stays_in_weekend_nights"].fillna(0)
    if {"stays_in_weekend_nights", "stay_nights"}.issubset(out.columns):
        out["weekend_ratio"] = np.where(out["stay_nights"] > 0, out["stays_in_weekend_nights"] / out["stay_nights"], 0.0)
    if {"adr", "stay_nights"}.issubset(out.columns):
        out["room_revenue"] = out["adr"].fillna(0) * out["stay_nights"].fillna(0)
    if {"bookingID", "room_revenue"}.issubset(out.columns):
        rev = out.groupby("bookingID")["room_revenue"].sum().rename("booking_revenue")
        out = out.merge(rev, on="bookingID", how="left")

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
            out["adr_bin"] = pd.qcut(out["adr"].rank(method="first"), q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        except Exception:
            out["adr_bin"] = "Q3"

    if {"children", "babies"}.issubset(out.columns):
        out["family_flag"] = (out["children"].fillna(0) + out["babies"].fillna(0)) > 0
    if "total_of_special_requests" in out.columns:
        out["req_flag"] = pd.to_numeric(out["total_of_special_requests"], errors="coerce").fillna(0) > 0
    if "required_car_parking_spaces" in out.columns:
        out["car_flag"] = pd.to_numeric(out["required_car_parking_spaces"], errors="coerce").fillna(0) > 0

    return out


def make_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)]), cat_cols),
        ],
        remainder="drop",
    )


def prepare_model_data(df: pd.DataFrame):
    target = "is_canceled"
    if target not in df.columns:
        raise ValueError("Kolom is_canceled tidak ditemukan.")

    work = df.copy()
    if len(work) > 2000:
        work = work.sample(2000, random_state=42).copy()

    if "reservation_status_date" in work.columns:
        train_df = work[work["reservation_status_date"] < "2019-01-01"].copy()
        test_df = work[work["reservation_status_date"] >= "2019-01-01"].copy()
        if len(train_df) < 50 or len(test_df) < 20:
            split = int(len(work) * 0.8)
            train_df = work.iloc[:split].copy()
            test_df = work.iloc[split:].copy()
    else:
        split = int(len(work) * 0.8)
        train_df = work.iloc[:split].copy()
        test_df = work.iloc[split:].copy()

    drop_cols = [target, "Invoice_ID", "bookingID", "reservation_status_date", "reservation_status"]
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")
    y_train = train_df[target].astype(int)
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors="ignore")
    y_test = test_df[target].astype(int)
    return X_train, X_test, y_train, y_test


@st.cache_resource(show_spinner=False)
def train_fast_model(X_train, y_train):
    preprocessor = make_preprocessor(X_train)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1200, solver="liblinear", class_weight="balanced", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(pipe, X_test, y_test, threshold=0.5):
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


def get_intrinsic_importance(pipe: Pipeline):
    model = pipe.named_steps["model"]
    feat_names = get_feature_names(pipe)
    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        fi = pd.DataFrame({"feature": feat_names, "importance": np.abs(coef)})
        return fi.sort_values("importance", ascending=False).reset_index(drop=True)
    return None


@st.cache_data(show_spinner=False)
def compute_pdp_1d(pipe: Pipeline, X_ref: pd.DataFrame, feat: str, grid_size: int = 20):
    s = pd.to_numeric(X_ref[feat], errors="coerce")
    vals = np.linspace(s.quantile(0.05), s.quantile(0.95), grid_size)
    pdp_vals = []
    for v in vals:
        temp = X_ref.copy()
        temp[feat] = v
        pdp_vals.append(pipe.predict_proba(temp)[:, 1].mean())
    return pd.DataFrame({feat: vals, "pred_prob": pdp_vals})


# ---------------------------------------------------------------------
# SIDEBAR - LOAD ONLY
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Data source")
    with st.form("load_form"):
        source_choice = st.radio("Choose", ["Upload file (CSV)", "Repo file (path)"], key="src_choice")
        uploaded = None
        repo_path = None

        if source_choice == "Upload file (CSV)":
            uploaded = st.file_uploader("Upload CSV", type=["csv"], key="uploader_csv")
        else:
            repo_path = st.text_input("Path file", value="", key="repo_path")

        load_btn = st.form_submit_button("Load data")

    st.header("Manual limits")
    preview_rows = st.number_input("Preview rows", min_value=50, max_value=20000, value=1000, step=50)
    split_rows = st.number_input("Rows for split room", min_value=50, max_value=5000, value=500, step=50)

if load_btn:
    clear_runtime(clear_data=True)
    try:
        if source_choice == "Upload file (CSV)":
            if uploaded is None:
                st.sidebar.warning("Upload CSV dulu.")
            else:
                st.session_state["df_raw"] = read_csv_cached(file_bytes=uploaded.getvalue(), source="upload")
                st.session_state["data_source"] = "upload"
                st.sidebar.success("Data loaded.")
        else:
            if not repo_path.strip():
                st.sidebar.warning("Isi path dulu.")
            else:
                st.session_state["df_raw"] = read_csv_cached(repo_path=repo_path.strip(), source="repo")
                st.session_state["data_source"] = "repo"
                st.sidebar.success("Data loaded.")
    except Exception as e:
        st.sidebar.error(f"Gagal load data: {e}")

if "df_raw" not in st.session_state:
    st.info("Masukkan dataset dari sidebar lalu klik **Load data**.")
    st.stop()

st.caption(f"Sumber data: **{st.session_state.get('data_source', 'unknown')}**")
st.write(f"Rows loaded: **{len(st.session_state['df_raw']):,}**")


# ---------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Overview",
        "Cleaning",
        "Filtering",
        "Split Room",
        "EDA",
        "Feature Engineering",
        "Modeling",
        "Importance + PDP + Simulator",
    ]
)


# ---------------------------------------------------------------------
# TAB 1 - OVERVIEW
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Raw preview only")
    raw_preview = st.session_state["df_raw"].head(int(preview_rows)).copy()
    st.dataframe(raw_preview.head(20), use_container_width=True)
    summary = pd.DataFrame(
        {
            "metric": ["rows loaded", "columns", "duplicates"],
            "value": [len(st.session_state["df_raw"]), st.session_state["df_raw"].shape[1], int(st.session_state["df_raw"].duplicated().sum())],
        }
    )
    st.dataframe(summary, use_container_width=True)


# ---------------------------------------------------------------------
# TAB 2 - CLEANING
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Cleaning")
    st.caption("Tidak ada cleaning otomatis. Klik tombol ini kalau memang ingin cleaning.")
    if st.button("Run cleaning", key="btn_clean"):
        st.session_state["df_clean"] = basic_clean(st.session_state["df_raw"])
        clear_runtime(clear_data=False)
        st.session_state["df_clean"] = basic_clean(st.session_state["df_raw"])
        st.success("Cleaning selesai.")

    if "df_clean" in st.session_state:
        st.dataframe(st.session_state["df_clean"].head(20), use_container_width=True)


# ---------------------------------------------------------------------
# TAB 3 - FILTERING
# ---------------------------------------------------------------------
with tab3:
    st.subheader("Filtering")
    st.caption("Filtering juga manual. Tidak ada filter yang jalan otomatis.")

    source_df = st.session_state.get("df_clean", st.session_state["df_raw"])

    filter_candidates = []
    preferred = [
        "hotel", "arrival_date_month", "market_segment", "distribution_channel",
        "deposit_type", "customer_type", "country", "meal",
    ]
    for c in preferred:
        if c in source_df.columns:
            filter_candidates.append(c)
    for c in source_df.columns:
        if source_df[c].dtype == "object" and c not in filter_candidates:
            filter_candidates.append(c)

    with st.form("filter_form"):
        filter_cols = st.multiselect("Pilih kolom filter", options=filter_candidates, default=[])
        filter_selections = {}
        for c in filter_cols:
            opts = sorted(source_df[c].dropna().astype(str).unique().tolist())
            filter_selections[c] = st.multiselect(c, options=opts, default=[])
        apply_filter_btn = st.form_submit_button("Apply filters")

    if apply_filter_btn:
        st.session_state["df_filtered"] = apply_filters(source_df, filter_cols, filter_selections)
        clear_runtime(clear_data=False)
        st.session_state["df_filtered"] = apply_filters(source_df, filter_cols, filter_selections)
        st.session_state["filter_applied"] = True
        st.success("Filter diterapkan.")

    if "df_filtered" in st.session_state:
        st.write(f"Rows after filter: **{len(st.session_state['df_filtered']):,}**")
        st.dataframe(st.session_state["df_filtered"].head(20), use_container_width=True)


# ---------------------------------------------------------------------
# TAB 4 - SPLIT ROOM
# ---------------------------------------------------------------------
with tab4:
    st.subheader("Split Room")
    st.caption("Split room hanya jalan saat tombol diklik, dan dibatasi oleh rows limit.")

    if st.button("Run split room", key="btn_split"):
        source_df = st.session_state.get("df_filtered", st.session_state.get("df_clean", st.session_state["df_raw"]))
        st.session_state["df_room"] = build_room_level_dataset(source_df, max_rows=int(split_rows))
        st.success("Split room selesai.")

    if "df_room" in st.session_state:
        df_room = st.session_state["df_room"]
        st.write(f"Rows produced by split room: **{len(df_room):,}**")
        st.dataframe(df_room.head(20), use_container_width=True)


# ---------------------------------------------------------------------
# TAB 5 - EDA
# ---------------------------------------------------------------------
with tab5:
    st.subheader("EDA")
    st.caption("EDA memakai data terakhir yang sudah kamu proses manual.")

    eda_source = (
        st.session_state.get("df_room")
        or st.session_state.get("df_filtered")
        or st.session_state.get("df_clean")
        or st.session_state["df_raw"]
    )

    if "is_canceled" in eda_source.columns:
        target_df = eda_source["is_canceled"].value_counts().sort_index().reset_index()
        target_df.columns = ["is_canceled", "count"]
        target_df["label"] = target_df["is_canceled"].map({0: "Not Canceled", 1: "Canceled"})
        chart_target = (
            alt.Chart(target_df)
            .mark_bar()
            .encode(
                x=alt.X("label:N", title="Status"),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["label", "count"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart_target, use_container_width=True)

    cat_candidates = [c for c in ["hotel", "arrival_date_month", "market_segment", "distribution_channel", "deposit_type", "customer_type", "bulk_3p_rooms"] if c in eda_source.columns]
    if cat_candidates and "is_canceled" in eda_source.columns:
        cat_sel = st.selectbox("Pilih kolom kategori", options=cat_candidates, key="eda_cat")
        rate_df = eda_source.groupby(cat_sel)["is_canceled"].mean().mul(100).sort_values(ascending=False).reset_index(name="cancel_rate")
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


# ---------------------------------------------------------------------
# TAB 6 - FEATURE ENGINEERING
# ---------------------------------------------------------------------
with tab6:
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
        st.write("Contoh fitur baru:", [c for c in ["minors", "party_size", "stay_nights", "weekend_ratio", "booking_revenue", "season", "lead_time_bin", "adr_bin", "family_flag", "req_flag", "car_flag"] if c in df_model.columns])


# ---------------------------------------------------------------------
# TAB 7 - MODELING
# ---------------------------------------------------------------------
with tab7:
    st.subheader("Modeling")
    st.caption("Modeling hanya jalan saat tombol diklik. Tidak ada training otomatis.")

    if st.button("Run model", key="btn_model"):
        source_df = st.session_state.get("df_model")
        if source_df is None:
            st.warning("Run feature engineering dulu.")
        else:
            X_train, X_test, y_train, y_test = prepare_model_data(source_df)
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


# ---------------------------------------------------------------------
# TAB 8 - IMPORTANCE + PDP + SIMULATOR
# ---------------------------------------------------------------------
with tab8:
    st.subheader("Importance + PDP + Simulator")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Generate importance", key="btn_imp"):
            if "model_pipe" not in st.session_state:
                st.warning("Run model dulu.")
            else:
                st.session_state["fi_intrinsic"] = get_intrinsic_importance(st.session_state["model_pipe"])
                st.success("Importance selesai.")
    with c2:
        if st.button("Prepare PDP", key="btn_pdp"):
            if "model_pipe" not in st.session_state:
                st.warning("Run model dulu.")
            else:
                st.session_state["pdp_candidates"] = st.session_state["X_test"].select_dtypes(include=[np.number]).columns.tolist()
                st.success("PDP siap.")

    if "fi_intrinsic" in st.session_state and st.session_state["fi_intrinsic"] is not None:
        fi = st.session_state["fi_intrinsic"].head(20)
        ch = (
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
        st.altair_chart(ch, use_container_width=True)

    if "pdp_candidates" in st.session_state and st.session_state["pdp_candidates"]:
        feat = st.selectbox("Pilih fitur numerik untuk PDP", options=st.session_state["pdp_candidates"], key="pdp_feat")
        pdp_df = compute_pdp_1d(st.session_state["model_pipe"], st.session_state["X_test"].copy(), feat, grid_size=20)
        ch = (
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
        st.altair_chart(ch, use_container_width=True)

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
                    "adr_bin", "family_flag", "req_flag", "car_flag"
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
        tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
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

            a, b = st.columns(2)
            a.metric("Predicted cancel probability", f"{prob:.2%}")
            b.metric("Predicted class", pred_label)

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
