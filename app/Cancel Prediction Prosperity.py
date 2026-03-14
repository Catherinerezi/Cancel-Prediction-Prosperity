# -*- coding: utf-8 -*-
"""
app.py
Hotel Booking Cancellation Prediction

Altair-only Streamlit app with:
- Sidebar data source (upload CSV / repo path)
- Data understanding
- Cleaning + room split
- EDA
- Feature engineering
- Modeling + tuning
- Feature importance
- PDP
- Booking simulator

Designed to follow the interaction style of the user's Food ETA app:
- user provides data first
- tabs organize workflow
- heavy computation runs only when user clicks buttons
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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

HAS_LGBM = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# ---------------------------------------------------------------------
# PAGE
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
    alt.themes.register("streamlit_safe_hotel", lambda: theme)
    alt.themes.enable("streamlit_safe_hotel")


_apply_altair_theme()

st.title("🏨 Hotel Booking Cancellation Prediction")
with st.expander("🎯 Tujuan", expanded=True):
    st.markdown(
        """
App ini dibuat untuk:
- memahami pola pembatalan booking hotel,
- mengubah data booking ke room-level,
- melakukan feature engineering,
- membandingkan model klasifikasi,
- melihat feature importance dan PDP,
- serta mensimulasikan probabilitas cancel untuk booking baru.
"""
    )


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def load_data(uploaded_file, repo_path: str | None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source = "upload"
    else:
        if not repo_path:
            return None, None
        df = pd.read_csv(repo_path)
        source = "repo"
    return df, source


def _signature(df_in: pd.DataFrame):
    idx = df_in.index.to_numpy()
    head = tuple(idx[:5].tolist()) if len(idx) else ()
    tail = tuple(idx[-5:].tolist()) if len(idx) else ()
    return (len(df_in), head, tail)


def _safe_metric(x, pct=False):
    if x is None:
        return "N/A"
    try:
        v = float(x)
        if np.isnan(v):
            return "N/A"
        return f"{v*100:.2f}%" if pct else f"{v:.4f}"
    except Exception:
        return "N/A"


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


def apply_global_filters(df: pd.DataFrame, filter_cols: list[str], selections: dict[str, list]):
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


def build_room_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        records.extend(split_rooms_cap4(row))

    out = pd.DataFrame(records).reset_index(drop=True)
    if "bookingID" in out.columns:
        out["bookingID"] = out["bookingID"].astype(str)
        out["Invoice_ID"] = np.arange(1, len(out) + 1, dtype=int)
        out["rooms_in_booking"] = out.groupby("bookingID")["Invoice_ID"].transform("count")
        out["bulk_3p_rooms"] = out["rooms_in_booking"] >= 3

        viol = (out["adults"] < 1) & ((out["children"] + out["babies"]) > 0)
        bad_ids = out.loc[viol, "bookingID"].unique()
        if len(bad_ids) > 0:
            out = out[~out["bookingID"].isin(bad_ids)].reset_index(drop=True)
            out["Invoice_ID"] = np.arange(1, len(out) + 1, dtype=int)
            out["rooms_in_booking"] = out.groupby("bookingID")["Invoice_ID"].transform("count")
            out["bulk_3p_rooms"] = out["rooms_in_booking"] >= 3
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in ["children", "babies", "adults", "stays_in_week_nights", "stays_in_weekend_nights", "adr"]:
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
            0.0,
        )
    if {"adr", "stay_nights"}.issubset(out.columns):
        out["room_revenue"] = out["adr"].fillna(0) * out["stay_nights"].fillna(0)

    if {"bookingID", "room_revenue"}.issubset(out.columns):
        revenue = out.groupby("bookingID")["room_revenue"].sum().rename("booking_revenue")
        out = out.merge(revenue, on="bookingID", how="left")

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
        out["lead_time"] = pd.to_numeric(out["lead_time"], errors="coerce")
        out["lead_time_bin"] = pd.cut(
            out["lead_time"],
            bins=[-1, 7, 30, 90, 180, 9999],
            labels=["≤7d", "8–30d", "31–90d", "91–180d", ">180d"],
        )

    if "adr" in out.columns:
        try:
            out["adr_bin"] = pd.qcut(
                out["adr"].rank(method="first"),
                q=5,
                labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
            )
        except Exception:
            out["adr_bin"] = "Q3"

    if {"children", "babies"}.issubset(out.columns):
        out["family_flag"] = (out["children"].fillna(0) + out["babies"].fillna(0)) > 0
    if {"adults", "children", "babies"}.issubset(out.columns):
        out["couple_flag"] = (out["adults"].fillna(0) == 2) & ((out["children"].fillna(0) + out["babies"].fillna(0)) == 0)
        out["solo_flag"] = (out["adults"].fillna(0) == 1) & ((out["children"].fillna(0) + out["babies"].fillna(0)) == 0)

    if "days_in_waiting_list" in out.columns:
        out["waiting_list_flag"] = pd.to_numeric(out["days_in_waiting_list"], errors="coerce").fillna(0) > 0
    if "total_of_special_requests" in out.columns:
        out["req_flag"] = pd.to_numeric(out["total_of_special_requests"], errors="coerce").fillna(0) > 0
    if "required_car_parking_spaces" in out.columns:
        out["car_flag"] = pd.to_numeric(out["required_car_parking_spaces"], errors="coerce").fillna(0) > 0
    if {"reserved_room_type", "assigned_room_type"}.issubset(out.columns):
        out["room_mismatch_flag"] = out["reserved_room_type"].astype(str) != out["assigned_room_type"].astype(str)
    if "booking_changes" in out.columns:
        out["changes_flag"] = pd.to_numeric(out["booking_changes"], errors="coerce").fillna(0) > 0

    return out


def prepare_model_data(df: pd.DataFrame):
    target = "is_canceled"
    if target not in df.columns:
        raise ValueError("Kolom is_canceled tidak ditemukan.")
    if "reservation_status_date" not in df.columns:
        raise ValueError("Kolom reservation_status_date tidak ditemukan.")

    train_df = df[df["reservation_status_date"] < "2019-01-01"].copy()
    test_df = df[df["reservation_status_date"] >= "2019-01-01"].copy()

    drop_cols = [
        target,
        "Invoice_ID",
        "bookingID",
        "reservation_status_date",
        "reservation_status",
    ]

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")
    y_train = train_df[target].astype(int)
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors="ignore")
    y_test = test_df[target].astype(int)
    return X_train, X_test, y_train, y_test


def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", _make_ohe()),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def evaluate_classifier(pipe, X_test, y_test, threshold=0.5):
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, prob),
        "pr_auc": average_precision_score(y_test, prob),
        "brier": brier_score_loss(y_test, prob),
        "logloss": log_loss(y_test, prob),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }


def train_all_models(X_train, y_train, X_test, y_test):
    preprocessor = make_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    candidates = [
        (
            "Logistic Regression",
            Pipeline([
                ("preprocess", preprocessor),
                ("model", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=42)),
            ]),
            {"model__C": [0.5, 1.0, 2.0]},
        ),
        (
            "Random Forest",
            Pipeline([
                ("preprocess", preprocessor),
                ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced_subsample")),
            ]),
            {"model__max_depth": [8, 12, None], "model__min_samples_leaf": [1, 3]},
        ),
    ]

    if HAS_LGBM:
        candidates.append(
            (
                "LightGBM",
                Pipeline([
                    ("preprocess", preprocessor),
                    ("model", LGBMClassifier(n_estimators=200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)),
                ]),
                {"model__num_leaves": [31, 63], "model__min_child_samples": [20, 40]},
            )
        )

    rows = []
    model_map = {}

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
        model_map[name] = best_pipe

        metrics = evaluate_classifier(best_pipe, X_test, y_test)
        rows.append(
            {
                "model": name,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "brier": metrics["brier"],
                "logloss": metrics["logloss"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "best_params": gs.best_params_,
            }
        )

    cmp = pd.DataFrame(rows).sort_values(["roc_auc", "pr_auc"], ascending=[False, False]).reset_index(drop=True)
    best_name = cmp.loc[0, "model"]
    best_pipe = model_map[best_name]
    return cmp, best_name, best_pipe, model_map


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

    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({"feature": feat_names, "importance": model.feature_importances_})
        return fi.sort_values("importance", ascending=False).reset_index(drop=True)

    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        fi = pd.DataFrame({"feature": feat_names, "importance": np.abs(coef)})
        return fi.sort_values("importance", ascending=False).reset_index(drop=True)

    return None


def get_permutation_importance(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    r = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring="average_precision",
    )
    feat_names = get_feature_names(pipe)
    k = min(len(feat_names), len(r.importances_mean))
    out = pd.DataFrame({"feature": feat_names[:k], "perm_importance": r.importances_mean[:k]})
    return out.sort_values("perm_importance", ascending=False).reset_index(drop=True)


def compute_pdp_1d(pipe: Pipeline, X_ref: pd.DataFrame, feat: str, grid_size: int = 30):
    s = pd.to_numeric(X_ref[feat], errors="coerce")
    vals = np.linspace(s.quantile(0.05), s.quantile(0.95), grid_size)
    pdp_vals = []
    for v in vals:
        temp = X_ref.copy()
        temp[feat] = v
        pdp_vals.append(pipe.predict_proba(temp)[:, 1].mean())
    return pd.DataFrame({feat: vals, "pred_prob": pdp_vals})


def ensure_processed_data(df_filtered: pd.DataFrame):
    sig = _signature(df_filtered)
    if st.session_state.get("proc_sig") != sig:
        for k in [
            "df_room", "df_model", "X_train", "X_test", "y_train", "y_test", "cmp",
            "best_name", "best_pipe", "model_map", "fi_intrinsic", "fi_perm", "pdp_candidates"
        ]:
            st.session_state.pop(k, None)
        st.session_state["proc_sig"] = sig

    if "df_room" not in st.session_state:
        st.session_state["df_room"] = build_room_level_dataset(df_filtered)
    if "df_model" not in st.session_state:
        st.session_state["df_model"] = add_features(st.session_state["df_room"])
    if "X_train" not in st.session_state:
        X_train, X_test, y_train, y_test = prepare_model_data(st.session_state["df_model"])
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test


def ensure_modeling(df_filtered: pd.DataFrame):
    ensure_processed_data(df_filtered)
    if "best_pipe" not in st.session_state:
        cmp, best_name, best_pipe, model_map = train_all_models(
            st.session_state["X_train"],
            st.session_state["y_train"],
            st.session_state["X_test"],
            st.session_state["y_test"],
        )
        st.session_state["cmp"] = cmp
        st.session_state["best_name"] = best_name
        st.session_state["best_pipe"] = best_pipe
        st.session_state["model_map"] = model_map


# ---------------------------------------------------------------------
# SIDEBAR - DATA SOURCE
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Data source")
    source_choice = st.radio("Choose", ["Upload file (CSV)", "Repo file (path)"], key="src_choice")

    uploaded = None
    repo_path = None
    if source_choice == "Upload file (CSV)":
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="uploader_csv_hotel")
    else:
        repo_path = st.text_input("Path (contoh: raw_data/train.csv)", value="", key="repo_path_hotel")


df_raw, src = load_data(uploaded, repo_path)
if df_raw is None:
    st.info("Upload CSV atau isi path repo dulu.")
    st.stop()

df = basic_clean(df_raw)


# ---------------------------------------------------------------------
# SIDEBAR - FILTERS
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Global filters")
    filter_candidates = []
    preferred = [
        "hotel", "arrival_date_month", "market_segment", "distribution_channel",
        "deposit_type", "customer_type", "country", "meal", "reserved_room_type"
    ]
    for c in preferred:
        if c in df.columns:
            filter_candidates.append(c)
    for c in df.columns:
        if df[c].dtype == "object" and c not in filter_candidates:
            filter_candidates.append(c)

    default_filter_cols = [c for c in ["hotel", "arrival_date_month", "market_segment"] if c in filter_candidates]
    filter_cols = st.multiselect("Pilih kolom filter", options=filter_candidates, default=default_filter_cols, key="hotel_filter_cols")

    filter_selections = {}
    for c in filter_cols:
        opts = sorted(df[c].dropna().astype(str).unique().tolist())
        filter_selections[c] = st.multiselect(c, options=opts, default=[], key=f"hotel_filter_{c}")


df_f = apply_global_filters(df, filter_cols, filter_selections)

st.caption(f"Sumber data: **{src}** | Rows aktif setelah filter: **{len(df_f):,}**")

sig_f = _signature(df_f)
if st.session_state.get("sig_f_prev") != sig_f:
    st.session_state["sig_f_prev"] = sig_f
    for k in [
        "df_room", "df_model", "X_train", "X_test", "y_train", "y_test", "cmp",
        "best_name", "best_pipe", "model_map", "fi_intrinsic", "fi_perm", "pdp_candidates"
    ]:
        st.session_state.pop(k, None)


# ---------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------
tab_overview, tab_understanding, tab_clean_split, tab_eda, tab_fe, tab_model, tab_imp, tab_sim = st.tabs(
    [
        "Overview",
        "Data Understanding",
        "Cleaning & Split Room",
        "EDA",
        "Feature Engineering",
        "Modeling",
        "Importance + PDP",
        "Simulator",
    ]
)


# ---------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------
with tab_overview:
    st.subheader("Project goal alignment")
    st.write(
        """
Proyek ini menjawab beberapa hal utama:
1. memahami pola pembatalan booking,
2. mengubah booking-level menjadi room-level,
3. membuat feature engineering yang relevan,
4. membandingkan beberapa model klasifikasi,
5. dan memberi alat simulasi probabilitas cancel untuk keputusan bisnis.
"""
    )

    st.subheader("Data snapshot (setelah filter)")
    st.dataframe(df_f.head(30), use_container_width=True)

    if "is_canceled" in df_f.columns:
        target_df = df_f["is_canceled"].value_counts().sort_index().reset_index()
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
            .properties(height=280, title="Distribusi target")
        )
        st.altair_chart(chart_target, use_container_width=True)


# ---------------------------------------------------------------------
# DATA UNDERSTANDING
# ---------------------------------------------------------------------
with tab_understanding:
    st.subheader("Data Understanding")
    c1, c2, c3 = st.columns([2, 2, 3])

    with c1:
        st.write("**Shape**")
        st.write(df_f.shape)
        st.dataframe(df_f.head(), use_container_width=True)

    with c2:
        st.write("**Dtypes**")
        st.json({col: str(tp) for col, tp in df_f.dtypes.items()})

    with c3:
        st.write("**Quick summary**")
        summary_df = pd.DataFrame(
            {
                "metric": ["rows", "columns", "duplicates"],
                "value": [len(df_f), df_f.shape[1], int(df_f.duplicated().sum())],
            }
        )
        st.dataframe(summary_df, use_container_width=True)

    st.subheader("Missing value (%)")
    missing_df = (df_f.isna().mean() * 100).sort_values(ascending=False).round(2).reset_index()
    missing_df.columns = ["column", "missing_pct"]
    cc1, cc2 = st.columns(2)
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


# ---------------------------------------------------------------------
# CLEANING & SPLIT ROOM
# ---------------------------------------------------------------------
with tab_clean_split:
    st.subheader("Cleaning & Split Room")
    run_split = st.button("Run cleaning + split room", key="btn_split_hotel")

    if run_split:
        ensure_processed_data(df_f)
        st.success("Cleaning + split room selesai.")

    if "df_room" in st.session_state:
        df_room = st.session_state["df_room"]
        a, b = st.columns(2)
        with a:
            st.write("**Room-level preview**")
            st.write(df_room.shape)
            st.dataframe(df_room.head(20), use_container_width=True)
        with b:
            split_summary = pd.DataFrame(
                {
                    "metric": [
                        "rows after split",
                        "unique bookingID",
                        "avg rooms per booking",
                        "bulk 3+ rooms",
                    ],
                    "value": [
                        len(df_room),
                        df_room["bookingID"].nunique() if "bookingID" in df_room.columns else np.nan,
                        round(df_room.groupby("bookingID").size().mean(), 2) if "bookingID" in df_room.columns else np.nan,
                        int(df_room["bulk_3p_rooms"].sum()) if "bulk_3p_rooms" in df_room.columns else np.nan,
                    ],
                }
            )
            st.dataframe(split_summary, use_container_width=True)

        if {"bulk_3p_rooms", "is_canceled"}.issubset(df_room.columns):
            bulk_df = (
                df_room.groupby("bulk_3p_rooms")["is_canceled"]
                .mean()
                .mul(100)
                .reset_index(name="cancel_rate")
            )
            bulk_df["type"] = bulk_df["bulk_3p_rooms"].map({False: "Non-bulk", True: "Bulk"})
            bulk_chart = (
                alt.Chart(bulk_df)
                .mark_bar()
                .encode(
                    x=alt.X("type:N", title="Booking type"),
                    y=alt.Y("cancel_rate:Q", title="Cancel rate (%)"),
                    tooltip=["type", alt.Tooltip("cancel_rate:Q", format=".2f")],
                )
                .properties(height=300, title="Cancel rate by split-room grouping")
                .interactive()
            )
            st.altair_chart(bulk_chart, use_container_width=True)
    else:
        st.info("Klik tombol untuk menjalankan cleaning + split room.")


# ---------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------
with tab_eda:
    st.subheader("EDA interaktif")
    run_eda_prep = st.button("Prepare data for EDA", key="btn_eda_hotel")
    if run_eda_prep:
        ensure_processed_data(df_f)
        st.success("Data EDA siap.")

    if "df_model" in st.session_state:
        eda_df = st.session_state["df_model"]

        if "is_canceled" in eda_df.columns:
            st.markdown("### Distribusi target")
            target_df = eda_df["is_canceled"].value_counts().sort_index().reset_index()
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
                .properties(height=300)
                .interactive()
            )
            st.altair_chart(target_chart, use_container_width=True)

        st.markdown("### Cancel rate by category")
        cat_candidates = [
            c for c in [
                "hotel", "arrival_date_month", "market_segment", "distribution_channel",
                "deposit_type", "customer_type", "season", "lead_time_bin",
                "adr_bin", "bulk_3p_rooms", "family_flag"
            ] if c in eda_df.columns
        ]
        if cat_candidates and "is_canceled" in eda_df.columns:
            cat_sel = st.selectbox("Pilih kolom kategori", options=cat_candidates, key="eda_cat_sel_hotel")
            rate_df = (
                eda_df.groupby(cat_sel)["is_canceled"]
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

        st.markdown("### ADR vs selected numeric feature")
        num_candidates = [c for c in ["lead_time", "stay_nights", "adults", "children", "booking_revenue", "room_revenue"] if c in eda_df.columns]
        if num_candidates and "adr" in eda_df.columns and "is_canceled" in eda_df.columns:
            num_sel = st.selectbox("Pilih fitur numerik", options=num_candidates, key="eda_num_sel_hotel")
            scatter_df = eda_df[[num_sel, "adr", "is_canceled"]].dropna().copy()
            scatter_chart = (
                alt.Chart(scatter_df)
                .mark_circle(opacity=0.45)
                .encode(
                    x=alt.X(f"{num_sel}:Q", title=num_sel),
                    y=alt.Y("adr:Q", title="ADR"),
                    color=alt.Color("is_canceled:N", title="Canceled"),
                    tooltip=[
                        alt.Tooltip(f"{num_sel}:Q", format=".2f"),
                        alt.Tooltip("adr:Q", format=".2f"),
                        "is_canceled:N",
                    ],
                )
                .properties(height=350)
                .interactive()
            )
            st.altair_chart(scatter_chart, use_container_width=True)
    else:
        st.info("Klik 'Prepare data for EDA' dulu.")


# ---------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------
with tab_fe:
    st.subheader("Feature Engineering")
    run_fe = st.button("Run feature engineering", key="btn_fe_hotel")
    if run_fe:
        ensure_processed_data(df_f)
        st.success("Feature engineering selesai.")

    if "df_model" in st.session_state:
        df_model = st.session_state["df_model"]
        num_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_model.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

        st.write(
            "**Contoh fitur numerik baru**",
            [c for c in ["minors", "party_size", "stay_nights", "weekend_ratio", "room_revenue", "booking_revenue"] if c in num_cols],
        )
        st.write(
            "**Contoh fitur kategori/flag baru**",
            [c for c in ["season", "lead_time_bin", "adr_bin", "family_flag", "couple_flag", "solo_flag", "waiting_list_flag", "req_flag", "car_flag", "room_mismatch_flag", "changes_flag"] if c in df_model.columns],
        )

        fe_summary = pd.DataFrame(
            {
                "type": ["numeric features", "categorical/bool features", "total columns"],
                "value": [len(num_cols), len(cat_cols), df_model.shape[1]],
            }
        )
        st.dataframe(fe_summary, use_container_width=True)
    else:
        st.info("Klik tombol FE dulu.")


# ---------------------------------------------------------------------
# MODELING
# ---------------------------------------------------------------------
with tab_model:
    st.subheader("Modeling + Tuning via CV")
    run_modeling = st.button("Run modeling", key="btn_model_hotel")
    if run_modeling:
        ensure_modeling(df_f)
        st.success("Modeling selesai.")

    if "cmp" in st.session_state:
        cmp = st.session_state["cmp"].copy()
        best_name = st.session_state["best_name"]
        st.success(f"Best model by ROC-AUC: **{best_name}**")

        show_cmp = cmp.copy()
        for c in ["roc_auc", "pr_auc", "brier", "logloss", "precision", "recall", "f1"]:
            if c in show_cmp.columns:
                show_cmp[c] = show_cmp[c].round(4)
        st.dataframe(show_cmp, use_container_width=True)

        metric_pick = st.selectbox(
            "Pilih metrik model comparison",
            options=["roc_auc", "pr_auc", "brier", "logloss", "precision", "recall", "f1"],
            key="metric_pick_hotel",
        )
        plot_df = show_cmp[["model", metric_pick]].copy()
        sort_rule = "-x" if metric_pick in ["roc_auc", "pr_auc", "precision", "recall", "f1"] else "x"
        cmp_chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                x=alt.X(f"{metric_pick}:Q", title=metric_pick),
                y=alt.Y("model:N", sort=sort_rule, title="Model"),
                tooltip=["model", alt.Tooltip(f"{metric_pick}:Q", format=".4f")],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(cmp_chart, use_container_width=True)
    else:
        st.info("Klik tombol Run modeling dulu.")


# ---------------------------------------------------------------------
# IMPORTANCE + PDP
# ---------------------------------------------------------------------
with tab_imp:
    st.subheader("Feature Importance + PDP")
    c1, c2 = st.columns(2)

    with c1:
        gen_imp = st.button("Generate feature importance", key="btn_imp_hotel")
    with c2:
        gen_pdp = st.button("Generate PDP candidates", key="btn_pdp_hotel")

    if gen_imp:
        ensure_modeling(df_f)
        pipe = st.session_state["best_pipe"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        st.session_state["fi_intrinsic"] = get_intrinsic_importance(pipe)
        st.session_state["fi_perm"] = get_permutation_importance(pipe, X_test, y_test)
        st.success("Feature importance selesai.")

    if gen_pdp:
        ensure_modeling(df_f)
        X_test = st.session_state["X_test"]
        st.session_state["pdp_candidates"] = X_test.select_dtypes(include=[np.number]).columns.tolist()
        st.success("PDP siap.")

    if "fi_intrinsic" in st.session_state or "fi_perm" in st.session_state:
        st.markdown("### Intrinsic importance / coefficients")
        fi_intrinsic = st.session_state.get("fi_intrinsic")
        if fi_intrinsic is not None:
            fi_chart = (
                alt.Chart(fi_intrinsic.head(20))
                .mark_bar()
                .encode(
                    x=alt.X("importance:Q", title="Importance / |Coefficient|"),
                    y=alt.Y("feature:N", sort="-x"),
                    tooltip=["feature", alt.Tooltip("importance:Q", format=".6f")],
                )
                .properties(height=400)
                .interactive()
            )
            st.altair_chart(fi_chart, use_container_width=True)
        else:
            st.info("Model ini tidak expose intrinsic importance.")

        st.markdown("### Permutation importance")
        fi_perm = st.session_state.get("fi_perm")
        if fi_perm is not None:
            perm_chart = (
                alt.Chart(fi_perm.head(20))
                .mark_bar()
                .encode(
                    x=alt.X("perm_importance:Q", title="Δ Average Precision"),
                    y=alt.Y("feature:N", sort="-x"),
                    tooltip=["feature", alt.Tooltip("perm_importance:Q", format=".6f")],
                )
                .properties(height=400)
                .interactive()
            )
            st.altair_chart(perm_chart, use_container_width=True)

    if "pdp_candidates" in st.session_state and st.session_state["pdp_candidates"]:
        st.markdown("### PDP")
        ensure_modeling(df_f)
        feat_pdp = st.selectbox("Pilih fitur numerik untuk PDP", options=st.session_state["pdp_candidates"], key="pdp_feature_hotel")
        pdp_df = compute_pdp_1d(st.session_state["best_pipe"], st.session_state["X_test"].copy(), feat_pdp, grid_size=40)
        pdp_chart = (
            alt.Chart(pdp_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(f"{feat_pdp}:Q", title=feat_pdp),
                y=alt.Y("pred_prob:Q", title="Predicted cancel probability"),
                tooltip=[alt.Tooltip(feat_pdp, format=".3f"), alt.Tooltip("pred_prob:Q", format=".4f")],
            )
            .properties(height=350, title=f"PDP — {feat_pdp}")
            .interactive()
        )
        st.altair_chart(pdp_chart, use_container_width=True)

    if "best_pipe" not in st.session_state:
        st.info("Jalankan modeling dulu, lalu generate importance / PDP.")


# ---------------------------------------------------------------------
# SIMULATOR
# ---------------------------------------------------------------------
with tab_sim:
    st.subheader("Booking Cancellation Simulator")
    prep_sim = st.button("Prepare simulator", key="btn_sim_hotel")
    if prep_sim:
        ensure_modeling(df_f)
        st.success("Simulator siap.")

    if "best_pipe" in st.session_state:
        X_train = st.session_state["X_train"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        pipe = st.session_state["best_pipe"]
        default_row = X_train.mode(dropna=True).iloc[0].copy()

        with st.form("hotel_sim_form"):
            st.caption("Isi beberapa field utama untuk melihat probabilitas cancel.")
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
                    v = float(default_row.get(col, s.median()))
                    user_input[col] = st.number_input(col, value=float(v))
                else:
                    opts = sorted([str(x) for x in s.dropna().astype(str).unique().tolist()])
                    default_val = str(default_row.get(col, opts[0] if opts else ""))
                    default_idx = opts.index(default_val) if default_val in opts else 0
                    user_input[col] = st.selectbox(col, opts, index=default_idx)

            threshold = st.slider("Threshold probability", 0.05, 0.95, 0.50, 0.01, key="threshold_sim_hotel")
            submitted = st.form_submit_button("Predict")

        st.markdown("### Threshold diagnostics on test set")
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

            m1, m2 = st.columns(2)
            m1.metric("Predicted cancel probability", f"{prob:.2%}")
            m2.metric("Predicted class", pred_label)

            donut_df = pd.DataFrame({"label": ["Cancel", "Remaining"], "value": [prob, 1 - prob]})
            donut_chart = (
                alt.Chart(donut_df)
                .mark_arc(innerRadius=70)
                .encode(
                    theta="value:Q",
                    color="label:N",
                    tooltip=["label", alt.Tooltip("value:Q", format=".2%")],
                )
                .properties(width=300, height=300, title="Prediction donut")
                .interactive()
            )
            st.altair_chart(donut_chart, use_container_width=False)
    else:
        st.info("Klik Prepare simulator dulu.")
