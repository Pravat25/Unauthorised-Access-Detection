import datetime
import json
import os
import re
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from flask_wtf import CSRFProtect
from flask_wtf.csrf import generate_csrf

# -------- Optional model libs ----------
try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# -------- Optional validation/QA libs ----------
try:
    from great_expectations.dataset import PandasDataset

    HAS_GE = True
except Exception:
    HAS_GE = False

try:
    from evidently.report import Report
    from evidently.metric_preset import (
        DataDriftPreset,
        DataQualityPreset,
        ClassificationPreset,
    )
    from evidently import ColumnMapping

    HAS_EVIDENTLY = True
except Exception:
    HAS_EVIDENTLY = False

try:
    from deepchecks.tabular import Dataset as DC_Dataset
    from deepchecks.tabular.suites import train_test_validation

    HAS_DEEPCHECKS = True
except Exception:
    HAS_DEEPCHECKS = False

# ---------------- Configuration ----------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "img")
REPORT_FOLDER = os.path.join(BASE_DIR, "static", "reports")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

ALLOWED_EXT = {"csv"}

app = Flask(__name__)
@app.context_processor
def inject_csrf_token():
    # make {{ csrf_token() }} available in Jinja templates
    return dict(csrf_token=generate_csrf)

app.config["SECRET_KEY"] = "replace-this-with-a-random-secret"
csrf = CSRFProtect(app)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "site.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MODEL_FOLDER"] = MODEL_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER


# Secure cookie settings (set SESSION_COOKIE_SECURE=True in production over HTTPS)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
)

@app.after_request
def add_security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"  # or "SAMEORIGIN"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'"
    )
    return resp



db: Any = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"


# ---------------- Models ----------------
class ModelRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    trained_on = db.Column(
        db.DateTime, default=lambda: datetime.datetime.now(datetime.UTC)
    )


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(150))
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(
        db.String(50), default="user"
    )  # 'user' | 'researcher' | 'admin' | 'superadmin'
    approved = db.Column(db.Boolean, default=False)
    registered_on = db.Column(
        db.DateTime, default=lambda: datetime.datetime.now(datetime.UTC)
    )

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)


@login_manager.user_loader
def load_user(user_id: str):
    # SQLAlchemy 2.x style (no LegacyAPIWarning)
    return db.session.get(User, int(user_id))


# ---------------- Helpers ----------------
def is_strong_password(pw: str) -> bool:
    """≥6 chars, with upper, lower, digit, special."""
    if not isinstance(pw, str) or len(pw) < 6:
        return False
    return (
        re.search(r"[A-Z]", pw)
        and re.search(r"[a-z]", pw)
        and re.search(r"\d", pw)
        and re.search(r"[^A-Za-z0-9]", pw)
    ) is not None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def load_model_wrapper(model_path):
    """Load joblib model; support {'model':..., 'meta':...} bundles."""
    loaded = joblib.load(model_path)
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded["model"], loaded.get("meta", {})
    else:
        return loaded, {}


def _save_fig(fig, filename_base) -> str:
    """Save fig under static/img and return a web path like 'img/file.png'."""
    out_name = f"{filename_base}.png"
    out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_name)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return f"img/{out_name}"


def save_confusion_matrix_basic(y_true, y_pred, filename="conf_matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            int(v),
            ha="center",
            va="center",
            color="white" if cm.max() > 50 else "black",
            fontsize=10,
        )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save_fig(fig, filename)


def save_confusion_matrix_normalized(y_true, y_pred, filename="conf_matrix_norm"):
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(all="ignore"):
        cmn = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cmn = np.nan_to_num(cmn)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cmn, cmap="Purples", vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cmn):
        ax.text(
            j,
            i,
            f"{v * 100:.0f}%",
            ha="center",
            va="center",
            color="white" if v > 0.50 else "black",
            fontsize=10,
        )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save_fig(fig, filename)


def save_prf_bars(y_true, y_pred, labels, filename="prf_bars"):
    pr, rc, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(x - width, pr, width, label="Precision")
    ax.bar(x, rc, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([str(lbl) for lbl in labels], rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.legend(frameon=False)
    for i, s in enumerate(support):
        ax.text(i, 0.02, f"n={int(s)}", ha="center", va="bottom", fontsize=8, color="#444")
    return _save_fig(fig, filename)


def save_top_misclass_bars(y_true, y_pred, filename="top_misclass"):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    err_pairs = []
    for i, a in enumerate(labels):
        for j, p in enumerate(labels):
            if i != j and cm[i, j] > 0:
                err_pairs.append((f"{a}→{p}", int(cm[i, j])))
    if not err_pairs:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No misclassifications", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return _save_fig(fig, filename)
    err_pairs.sort(key=lambda x: x[1], reverse=True)
    err_pairs = err_pairs[:10]
    names = [n for n, _ in err_pairs]
    vals = [v for _, v in err_pairs]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(names, vals)
    ax.set_title("Top Misclassifications (count)")
    ax.set_ylabel("Count")
    ax.set_xticklabels(names, rotation=45, ha="right")
    return _save_fig(fig, filename)


def save_roc_pr_curves(y_true, y_score, classes, prefix="roc_pr"):
    y_bin = label_binarize(y_true, classes=classes)
    # ROC (macro)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    fpr_dict, tpr_dict = {}, {}
    for i, _ in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        fpr_dict[i], tpr_dict[i] = fpr, tpr
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in fpr_dict]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in tpr_dict:
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= len(tpr_dict)
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, label=f"macro-ROC (AUC={macro_auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="#888")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC (macro)")
    ax.legend(frameon=False)
    roc_path = _save_fig(fig, f"{prefix}_roc")

    # PR macro (text summary, fast)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    aps = []
    for i, _ in enumerate(classes):
        aps.append(average_precision_score(y_bin[:, i], y_score[:, i]))
    macro_ap = float(np.mean(aps)) if aps else 0.0
    ax.text(
        0.5, 0.5, f"Macro Average Precision = {macro_ap:.2f}", ha="center", va="center", fontsize=12
    )
    ax.set_axis_off()
    pr_path = _save_fig(fig, f"{prefix}_pr")
    return roc_path, pr_path, macro_ap


def save_calibration_and_confidence(y_true, y_pred, y_proba, filename_prefix="calib_conf"):
    max_conf = y_proba.max(axis=1)
    correct = y_pred == y_true

    bins = np.linspace(0.0, 1.0, 11)
    idx = np.digitize(max_conf, bins) - 1
    bin_conf, bin_acc = [], []
    for b in range(10):
        sel = idx == b
        if sel.any():
            bin_conf.append(max_conf[sel].mean())
            bin_acc.append(correct[sel].mean())
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="#aaa", label="Perfect")
    ax.plot(bin_conf, bin_acc, marker="o", label="Observed")
    ax.set_title("Calibration (Reliability)")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.legend(frameon=False)
    calib_path = _save_fig(fig, f"{filename_prefix}_reliability")

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.hist(max_conf[correct], bins=15, alpha=0.7, label="Correct")
    ax.hist(max_conf[~correct], bins=15, alpha=0.7, label="Incorrect")
    ax.set_title("Prediction Confidence Histogram")
    ax.set_xlabel("Max predicted probability")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    confhist_path = _save_fig(fig, f"{filename_prefix}_confhist")

    bands = {
        "high": int((max_conf >= 0.9).sum()),
        "medium": int(((max_conf >= 0.6) & (max_conf < 0.9)).sum()),
        "low": int((max_conf < 0.6).sum()),
    }
    return calib_path, confhist_path, bands


def save_unlabeled_confidence(y_proba, filename_prefix="unlabeled_conf"):
    max_conf = y_proba.max(axis=1)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.hist(max_conf, bins=15)
    ax.set_title("Prediction Confidence (Unlabeled)")
    ax.set_xlabel("Max predicted probability")
    ax.set_ylabel("Count")
    confhist_path = _save_fig(fig, f"{filename_prefix}_hist")
    bands = {
        "high": int((max_conf >= 0.9).sum()),
        "medium": int(((max_conf >= 0.6) & (max_conf < 0.9)).sum()),
        "low": int((max_conf < 0.6).sum()),
    }
    return confhist_path, bands


def save_feature_importance(model, feature_names, filename="feat_importance"):
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            importances = np.mean(np.abs(coef), axis=0)
    if importances is None or not feature_names:
        fig, ax = plt.subplots(figsize=(4.2, 2.4))
        ax.text(0.5, 0.5, "Feature importance not available", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return _save_fig(fig, filename)

    idx = np.argsort(importances)[::-1][:15]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.barh(top_names[::-1], top_vals[::-1])
    ax.set_title("Top Feature Importance")
    ax.set_xlabel("Importance")
    return _save_fig(fig, filename)


# -------- NEW: extra charts --------
def save_cumulative_gains_binary(y_true, y_score, filename="gains"):
    """Cumulative gains curve (binary only)."""
    classes = np.unique(y_true)
    if len(classes) != 2:
        return None
    pos_label = classes[1]
    y_bin = (y_true == pos_label).astype(int)
    idx = np.argsort(-y_score)
    y_sorted = y_bin[idx]
    cum_pos = np.cumsum(y_sorted)
    total_pos = y_bin.sum()
    n = len(y_true)
    perc_samples = np.arange(1, n + 1) / n
    gains = cum_pos / (total_pos if total_pos > 0 else 1)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(perc_samples, gains, label="Model")
    ax.plot([0, 1], [0, 1], "--", color="#888", label="Random")
    ax.set_title("Cumulative Gains (positive class)")
    ax.set_xlabel("Fraction of samples (sorted by score)")
    ax.set_ylabel("Fraction of positives captured")
    ax.legend(frameon=False)
    return _save_fig(fig, filename)


def save_lift_curve_binary(y_true, y_score, filename="lift"):
    classes = np.unique(y_true)
    if len(classes) != 2:
        return None
    pos_label = classes[1]
    y_bin = (y_true == pos_label).astype(int)
    idx = np.argsort(-y_score)
    y_sorted = y_bin[idx]
    cum_pos = np.cumsum(y_sorted)
    n = len(y_true)
    overall_rate = y_bin.mean() if n > 0 else 0
    if overall_rate == 0:
        return None
    perc_samples = np.arange(1, n + 1) / n
    lift = (cum_pos / np.arange(1, n + 1)) / overall_rate

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(perc_samples, lift)
    ax.set_title("Lift Curve")
    ax.set_xlabel("Fraction of samples (sorted by score)")
    ax.set_ylabel("Lift")
    return _save_fig(fig, filename)


def save_ks_curve_binary(y_true, y_score, filename="ks"):
    classes = np.unique(y_true)
    if len(classes) != 2:
        return None
    pos_label = classes[1]
    y_bin = (y_true == pos_label).astype(int)
    pos_scores = y_score[y_bin == 1]
    neg_scores = y_score[y_bin == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None

    grid = np.linspace(0, 1, 101)
    F_pos = np.array([(pos_scores <= t).mean() for t in grid])
    F_neg = np.array([(neg_scores <= t).mean() for t in grid])
    ks = np.max(np.abs(F_pos - F_neg))
    ks_t = grid[np.argmax(np.abs(F_pos - F_neg))]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(grid, F_pos, label="Pos CDF")
    ax.plot(grid, F_neg, label="Neg CDF")
    ax.vlines(
        ks_t,
        F_neg[np.argmax(np.abs(F_pos - F_neg))],
        F_pos[np.argmax(np.abs(F_pos - F_neg))],
        linestyles="--",
        colors="red",
        label=f"KS={ks:.2f}",
    )
    ax.set_title("KS Statistic Curve")
    ax.set_xlabel("Score (probability of positive)")
    ax.set_ylabel("CDF")
    ax.legend(frameon=False)
    return _save_fig(fig, filename)


def save_support_vs_recall(y_true, y_pred, filename="support_recall"):
    labels = np.unique(y_true)
    _, recall, _, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(support, recall, s=np.clip(support, 20, 400), alpha=0.8)
    for i, lab in enumerate(labels):
        ax.text(support[i], recall[i], str(lab), fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("Support (true count)")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)
    ax.set_title("Support vs Recall (size≈support)")
    return _save_fig(fig, filename)


def save_trueclass_proba_hists(y_true, y_proba, classes, filename="trueproba"):
    # overlay up to 4 most frequent classes
    counts = pd.Series(y_true).value_counts()
    top = list(counts.index[:4])
    if len(top) == 0:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for cls in top:
        idx = np.where(classes == cls)[0]
        if len(idx) == 0:
            continue
        col = idx[0]
        vals = y_proba[y_true == cls, col]
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=15, alpha=0.6, label=str(cls))
    if not ax.has_data():
        plt.close(fig)
        return None
    ax.set_title("True-class Probability (by true label)")
    ax.set_xlabel("P(true class)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    return _save_fig(fig, filename)


def save_distribution_pie(distr: dict, filename="dist_pie"):
    if not distr:
        return None
    labels = list(distr.keys())
    sizes = list(distr.values())
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, counterclock=False)
    ax.axis("equal")
    ax.set_title("Class Distribution")
    return _save_fig(fig, filename)


# -------- Validation / QA helpers (optional libs) --------
def _evidently_colmap(df: pd.DataFrame, target: str | None, pred: str | None,
                      proba_cols: list[str] | None):
    if not HAS_EVIDENTLY:
        return None
    nums = list(df.select_dtypes(include=[np.number]).columns)
    cats = [c for c in df.columns if c not in nums]
    return ColumnMapping(
        target=target,
        prediction=pred,
        prediction_probas=proba_cols or None,
        numerical_features=nums,
        categorical_features=cats,
    )


def save_ge_data_quality_html(df: pd.DataFrame, out_name: str) -> str | None:
    """Great Expectations: simple expectations -> HTML-ish page."""
    if not HAS_GE:
        return None
    try:
        ds = PandasDataset(df.copy())
        ds.expect_table_row_count_to_be_between(min_value=1)
        ds.expect_table_columns_to_match_set(set(df.columns))
        for col in df.columns:
            ds.expect_column_values_to_not_be_null(col)
        result = ds.validate()

        out_path = os.path.join(app.config["REPORT_FOLDER"], out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        html = (
            "<html><head><meta charset='utf-8'>"
            "<title>Great Expectations - Data Quality</title></head><body>"
            "<h2>Great Expectations - Data Quality</h2>"
            f"<pre>{json.dumps(result, indent=2, default=str)}</pre>"
            "</body></html>"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        return out_name
    except Exception:
        return None


def save_evidently_data_drift_html(current: pd.DataFrame, reference: pd.DataFrame,
                                   out_name: str,
                                   target_col: str | None = None,
                                   pred_col: str | None = None,
                                   proba_cols: list[str] | None = None) -> str | None:
    if not HAS_EVIDENTLY:
        return None
    try:
        mapping = _evidently_colmap(current, target_col, pred_col, proba_cols)
        report = Report(metrics=[DataQualityPreset(), DataDriftPreset()])
        report.run(current_data=current, reference_data=reference, column_mapping=mapping)
        out_path = os.path.join(app.config["REPORT_FOLDER"], out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        report.save_html(out_path)
        return out_name
    except Exception:
        return None


def save_evidently_classification_html(y_true: np.ndarray, y_pred: np.ndarray,
                                       y_proba: np.ndarray | None,
                                       class_labels: np.ndarray | list,
                                       out_name: str) -> str | None:
    if not HAS_EVIDENTLY:
        return None
    try:
        df = pd.DataFrame({"target": y_true, "prediction": y_pred})
        proba_cols = None
        if y_proba is not None:
            proba_cols = []
            for i, cls in enumerate(class_labels):
                col = f"proba_{cls}"
                df[col] = y_proba[:, i]
                proba_cols.append(col)
        mapping = _evidently_colmap(df, target="target", pred="prediction",
                                    proba_cols=proba_cols)
        report = Report(metrics=[ClassificationPreset()])
        report.run(current_data=df, reference_data=None, column_mapping=mapping)
        out_path = os.path.join(app.config["REPORT_FOLDER"], out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        report.save_html(out_path)
        return out_name
    except Exception:
        return None


def save_deepchecks_train_test_html(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                    label_col: str, out_name: str) -> str | None:
    if not HAS_DEEPCHECKS:
        return None
    try:
        cats = [
            c for c in train_df.columns
            if c != label_col and
            (train_df[c].dtype == "object" or train_df[c].nunique() <= 20)
        ]
        ds_train = DC_Dataset(
            train_df.drop(columns=[label_col]),
            label=train_df[label_col],
            cat_features=cats or None,
        )
        ds_test = DC_Dataset(
            test_df.drop(columns=[label_col]),
            label=test_df[label_col],
            cat_features=cats or None,
        )
        suite = train_test_validation()
        result = suite.run(train_dataset=ds_train, test_dataset=ds_test)
        out_path = os.path.join(app.config["REPORT_FOLDER"], out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result.save_as_html(out_path)
        return out_name
    except Exception:
        return None


def get_models():
    return {
        "logistic-regression": LogisticRegression(max_iter=1000),
        "random-forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "svm": SVC(probability=True, kernel="rbf", gamma="scale", C=1.0),
        "neural-network": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42),
        "gradient-boosting": GradientBoostingClassifier(random_state=42),
        **(
            {
                "xgboost": XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                )
            }
            if HAS_XGB
            else {}
        ),
        **({"lightgbm": LGBMClassifier(random_state=42)} if HAS_LGBM else {}),
    }


def pick_label_column(df: pd.DataFrame) -> str:
    for name in ["class", "label", "target"]:
        if name in df.columns:
            return name
    return df.columns[-1]


def trim_to_10k_stratified(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if len(df) <= 10000:
        return df.reset_index(drop=True)
    y = df[label_col]
    sss = StratifiedShuffleSplit(n_splits=1, train_size=10000, random_state=42)
    idx, _ = next(sss.split(np.zeros(len(y)), y))
    return df.iloc[idx].reset_index(drop=True)


# ---------------- Routes ----------------
@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return render_template("home.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            if not user.approved:
                msg = "awaiting superadmin approval"
                flash(msg, "warning")
                return render_template("login.html", immediate_msg=msg), 200
            login_user(user)
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid credentials.", "danger")
        return render_template("login.html"), 200

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        fullname = (request.form.get("fullname") or "").strip()
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""
        role = request.form.get("role") or "user"

        if not is_strong_password(password):
            msg = (
                "Password must be ≥6 chars and include upper, lower, number, and special character."
            )
            flash(msg, "warning")
            return render_template(
                "register.html",
                immediate_msg=msg,
                preset_fullname=fullname,
                preset_email=email,
                preset_role=role,
            ), 200

        if User.query.filter_by(email=email).first():
            msg = "Email already registered."
            flash(msg, "warning")
            return render_template(
                "register.html",
                immediate_msg=msg,
                preset_fullname=fullname,
                preset_email=email,
                preset_role=role,
            ), 200

        u = User(
            fullname=fullname,
            email=email,
            password_hash=generate_password_hash(password),
            role=role,
            approved=False,
        )
        db.session.add(u)
        db.session.commit()
        flash(
            "Registration submitted — superadmin will approve your account shortly.",
            "info",
        )
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/dashboard")
@login_required
def dashboard():
    total_users = User.query.count()
    pending = User.query.filter_by(approved=False).count()
    approved = User.query.filter_by(approved=True).count()
    model_files = [f for f in os.listdir(app.config["MODEL_FOLDER"]) if f.endswith(".pkl")]
    model_count = len(model_files)
    models = ModelRecord.query.all()
    return render_template(
        "dashboard.html",
        total_users=total_users,
        pending=pending,
        approved=approved,
        model_count=model_count,
        models=models,
    )


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("home"))


# ---------------- Admin pages ----------------
@app.route("/admin/users")
@login_required
def admin_users():
    if current_user.role != "superadmin":
        flash("Superadmin access required.", "danger")
        return redirect(url_for("dashboard"))
    users = User.query.order_by(User.registered_on.desc()).all()
    return render_template("admin_users.html", users=users)


@app.route("/admin/approve/<int:user_id>")
@login_required
def approve(user_id):
    if current_user.role != "superadmin":
        flash("Superadmin access required.", "danger")
        return redirect(url_for("dashboard"))
    u = User.query.get_or_404(user_id)
    u.approved = True
    db.session.commit()
    flash(f"User {u.email} approved.", "success")
    return redirect(url_for("admin_users"))


@app.route("/admin/reject/<int:user_id>")
@login_required
def reject(user_id):
    if current_user.role != "superadmin":
        flash("Superadmin access required.", "danger")
        return redirect(url_for("dashboard"))
    u = User.query.get_or_404(user_id)
    db.session.delete(u)
    db.session.commit()
    flash(f"User {u.email} rejected and removed.", "info")
    return redirect(url_for("admin_users"))


# ---- Training (admins + superadmin only) ----
@app.route("/admin/train", methods=["GET", "POST"])
@login_required
def admin_train():
    if current_user.role not in ("admin", "superadmin"):
        flash("Admin access required.", "danger")
        return redirect(url_for("dashboard"))

    models_available = get_models()
    preselected = request.args.get("model")  # for retrain links

    if request.method == "POST":
        if "trainfile" not in request.files:
            flash("No file uploaded.", "danger")
            return redirect(request.url)

        file = request.files["trainfile"]
        model_type = request.form.get("model_type")

        if file.filename == "" or not allowed_file(file.filename):
            flash("Please upload a CSV training file.", "warning")
            return redirect(request.url)
        if model_type not in models_available:
            flash("Unsupported model type.", "warning")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        tmp_path = os.path.join(BASE_DIR, "tmptrain_" + filename)
        file.save(tmp_path)

        try:
            df = pd.read_csv(tmp_path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            flash("Failed to read CSV: " + str(e), "danger")
            return redirect(request.url)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        label_col = pick_label_column(df)
        if label_col not in df.columns:
            flash("Could not find label/target column.", "danger")
            return redirect(request.url)

        df = trim_to_10k_stratified(df, label_col)

        y = df[label_col]
        X = df.drop(columns=[label_col])
        X = X.fillna(0)
        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = models_available[model_type]
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            flash("Training failed: " + str(e), "danger")
            return redirect(request.url)

        save_path = os.path.join(app.config["MODEL_FOLDER"], f"{model_type}.pkl")
        meta = {"columns": list(X.columns)}
        joblib.dump({"model": model, "meta": meta}, save_path)

        try:
            y_pred_test = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred_test)

            _ = save_confusion_matrix_basic(
                y_test, y_pred_test, filename=f"conf_train_{model_type}"
            )
            try:
                _ = save_feature_importance(
                    model, meta["columns"], filename=f"featimp_{model_type}"
                )
            except Exception:
                pass

            # ---- Save reference sample for future drift checks ----
            try:
                ref_df = X_train.copy()
                ref_df[label_col] = y_train.values
                ref_sample = ref_df.sample(
                    n=min(5000, len(ref_df)), random_state=42
                )
                ref_path = os.path.join(
                    app.config["REPORT_FOLDER"], f"ref_{model_type}.csv"
                )
                ref_sample.to_csv(ref_path, index=False)
            except Exception:
                pass

            # ---- Deepchecks train/test validation report (optional) ----
            try:
                dc_html = save_deepchecks_train_test_html(
                    train_df=pd.concat(
                        [X_train.reset_index(drop=True),
                         y_train.reset_index(drop=True).rename(label_col)],
                        axis=1,
                    ),
                    test_df=pd.concat(
                        [X_test.reset_index(drop=True),
                         y_test.reset_index(drop=True).rename(label_col)],
                        axis=1,
                    ),
                    label_col=label_col,
                    out_name=(
                        f"deepchecks_train_test_{model_type}_"
                        f"{int(datetime.datetime.now(datetime.UTC).timestamp())}.html"
                    ),
                )
                if dc_html:
                    flash(f"Deepchecks report saved: {dc_html}", "info")
            except Exception:
                pass

            rec = ModelRecord.query.filter_by(name=model_type).first()
            if not rec:
                rec = ModelRecord(name=model_type, accuracy=acc)
                db.session.add(rec)
            else:
                rec.accuracy = acc
                rec.trained_on = datetime.datetime.now(datetime.UTC)
            db.session.commit()

            flash(f'Model "{model_type}" trained. Accuracy: {acc * 100:.2f}%', "success")
        except Exception as e:
            flash(f"Trained but evaluation failed: {e}", "warning")

        return redirect(url_for("admin_models"))

    return render_template("train.html", models=list(models_available.keys()), preselected=preselected)


@app.route("/admin/models")
@login_required
def admin_models():
    if current_user.role not in ("admin", "superadmin"):
        flash("Admin access required.", "danger")
        return redirect(url_for("dashboard"))

    items = []
    for rec in ModelRecord.query.all():
        conf_abs = os.path.join(app.config["UPLOAD_FOLDER"], f"conf_train_{rec.name}.png")
        feat_abs = os.path.join(app.config["UPLOAD_FOLDER"], f"featimp_{rec.name}.png")
        items.append(
            {
                "rec": rec,
                "conf_img": f"img/conf_train_{rec.name}.png" if os.path.exists(conf_abs) else None,
                "feat_img": f"img/featimp_{rec.name}.png" if os.path.exists(feat_abs) else None,
            }
        )
    return render_template("admin_models.html", items=items)


# ---- Detection (all approved users) ----
@app.route("/detect", methods=["GET", "POST"])
@login_required
def detect():
    # Only show trained models present on disk
    models_available = [f[:-4] for f in os.listdir(app.config["MODEL_FOLDER"]) if f.endswith(".pkl")]

    if request.method == "POST":
        if "testfile" not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)

        file = request.files["testfile"]
        model_name = request.form.get("model_name")

        if file.filename == "" or not allowed_file(file.filename):
            flash("Please upload a CSV file.", "warning")
            return redirect(request.url)
        if not model_name:
            flash("Please choose a model.", "warning")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        tmp_path = os.path.join(BASE_DIR, "tmp_" + filename)
        file.save(tmp_path)
        try:
            df = pd.read_csv(tmp_path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            flash("Failed to read CSV: " + str(e), "danger")
            return redirect(request.url)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Try to infer label column if present
        label_col = None
        for c in ["class", "label", "target"]:
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            last = df.columns[-1]
            if df[last].nunique() <= max(10, int(0.01 * len(df))):
                label_col = last

        if label_col and label_col in df.columns:
            y = df[label_col].values
            X = df.drop(columns=[label_col])
        else:
            y = None
            X = df

        model_file = os.path.join(app.config["MODEL_FOLDER"], f"{model_name}.pkl")
        if not os.path.exists(model_file):
            flash("Selected model not found. Ask admin to train the model first.", "warning")
            return redirect(request.url)

        model, meta = load_model_wrapper(model_file)

        X = X.fillna(0)
        X = pd.get_dummies(X)
        if meta.get("columns"):
            for c in meta["columns"]:
                if c not in X.columns:
                    X[c] = 0
            X = X[meta["columns"]]

        try:
            y_pred = model.predict(X)
        except Exception as e:
            flash("Model prediction failed: " + str(e), "danger")
            return redirect(request.url)

        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X)
            except Exception:
                y_proba = None

        # timezone-aware timestamp for filenames
        ts = int(datetime.datetime.now(datetime.UTC).timestamp())

        # --- Great Expectations: basic data quality on uploaded data ---
        try:
            ge_html = save_ge_data_quality_html(df, out_name=f"ge_quality_{model_name}_{ts}.html")
        except Exception:
            ge_html = None

        # --- labeled case ---
        if y is not None and set(pd.unique(y)).issubset(set(pd.unique(y_pred))) and len(y) == len(y_pred):
            classes = np.unique(y)
            charts = {}

            # Core charts (confusion matrices only if ≥2 classes)
            if len(classes) >= 2:
                try:
                    charts["cm"] = save_confusion_matrix_basic(y, y_pred, filename=f"cm_{ts}")
                except Exception:
                    charts["cm"] = None
                try:
                    charts["cm_norm"] = save_confusion_matrix_normalized(
                        y, y_pred, filename=f"cm_norm_{ts}"
                    )
                except Exception:
                    charts["cm_norm"] = None
            else:
                charts["cm"] = None
                charts["cm_norm"] = None

            try:
                charts["prf"] = save_prf_bars(y, y_pred, labels=classes, filename=f"prf_{ts}")
            except Exception:
                charts["prf"] = None
            try:
                charts["misclass"] = save_top_misclass_bars(y, y_pred, filename=f"misclass_{ts}")
            except Exception:
                charts["misclass"] = None
            try:
                pred_dist = pd.Series(y_pred).value_counts().to_dict()
                charts["pred_pie"] = save_distribution_pie(pred_dist, filename=f"predpie_{ts}")
            except Exception:
                charts["pred_pie"] = None

            # Probability-based charts only with proba AND ≥2 classes
            macro_ap = None
            conf_bands = None
            if y_proba is not None and len(classes) >= 2:
                try:
                    charts["roc"], charts["pr"], macro_ap = save_roc_pr_curves(
                        y, y_proba, classes=classes, prefix=f"rocpr_{ts}"
                    )
                except Exception:
                    charts["roc"] = charts.get("roc")
                    charts["pr"] = charts.get("pr")
                try:
                    charts["calib"], charts["confhist"], conf_bands = (
                        save_calibration_and_confidence(
                            y_true=y, y_pred=y_pred, y_proba=y_proba,
                            filename_prefix=f"calib_{ts}"
                        )
                    )
                except Exception:
                    charts["calib"] = charts.get("calib")
                    charts["confhist"] = charts.get("confhist")
                try:
                    charts["supp_recall"] = save_support_vs_recall(
                        y, y_pred, filename=f"supprec_{ts}"
                    )
                except Exception:
                    charts["supp_recall"] = None
                try:
                    charts["trueproba"] = save_trueclass_proba_hists(
                        y, y_proba, classes=classes, filename=f"trueproba_{ts}"
                    )
                except Exception:
                    charts["trueproba"] = None
            else:
                charts["roc"] = None
                charts["pr"] = None
                charts["calib"] = charts.get("calib")
                charts["confhist"] = charts.get("confhist")

            # Binary-only curves (need probabilities)
            if y_proba is not None and len(classes) == 2:
                try:
                    pos_idx = np.where(classes == classes[1])[0][0]
                    score = y_proba[:, pos_idx]
                    charts["gains"] = save_cumulative_gains_binary(
                        y, score, filename=f"gains_{ts}"
                    )
                    charts["lift"] = save_lift_curve_binary(y, score, filename=f"lift_{ts}")
                    charts["ks"] = save_ks_curve_binary(y, score, filename=f"ks_{ts}")
                except Exception:
                    charts["gains"] = charts.get("gains")
                    charts["lift"] = charts.get("lift")
                    charts["ks"] = charts.get("ks")

            # ---- Evidently classification performance (optional) ----
            try:
                ev_cls_html = save_evidently_classification_html(
                    y_true=y, y_pred=y_pred, y_proba=y_proba,
                    class_labels=classes, out_name=f"evidently_cls_{model_name}_{ts}.html"
                )
            except Exception:
                ev_cls_html = None

            # ---- Evidently drift vs. saved reference (optional) ----
            try:
                ev_drift_html = None
                ref_path = os.path.join(app.config["REPORT_FOLDER"], f"ref_{model_name}.csv")
                if os.path.exists(ref_path):
                    ref_df = pd.read_csv(ref_path)
                    # compare on feature space only
                    ref_feat = ref_df.drop(
                        columns=[c for c in ["class", "label", "target"] if c in ref_df.columns],
                        errors="ignore",
                    )
                    cur = X.copy()
                    for c in ref_feat.columns:
                        if c not in cur.columns:
                            cur[c] = 0
                    for c in cur.columns:
                        if c not in ref_feat.columns:
                            ref_feat[c] = 0
                    cur = cur[sorted(cur.columns)]
                    ref_feat = ref_feat[sorted(ref_feat.columns)]
                    ev_drift_html = save_evidently_data_drift_html(
                        current=cur, reference=ref_feat,
                        out_name=f"evidently_drift_{model_name}_{ts}.html"
                    )
            except Exception:
                ev_drift_html = None

            validation_reports = [p for p in [ge_html, ev_cls_html, ev_drift_html] if p]

            accuracy = accuracy_score(y, y_pred)
            cls_report_text = classification_report(y, y_pred)

            return render_template(
                "results.html",
                accuracy=accuracy,
                report=cls_report_text,
                charts=charts,
                macro_ap=macro_ap,
                conf_bands=conf_bands,
                feat_img=None,
                validation_reports=validation_reports,
            )

        # --- unlabeled case ---
        else:
            out_name = f"predictions_{ts}.csv"
            out_path = os.path.join(app.config["REPORT_FOLDER"], out_name)
            out_df = df.copy()
            out_df["prediction"] = y_pred
            out_df.to_csv(out_path, index=False)

            dist = pd.Series(y_pred).value_counts().to_dict()

            confhist_path, bands = (None, None)
            if y_proba is not None:
                try:
                    confhist_path, bands = save_unlabeled_confidence(
                        y_proba, filename_prefix=f"unlabeled_{ts}"
                    )
                except Exception:
                    confhist_path, bands = (None, None)

            try:
                dist_pie = save_distribution_pie(dist, filename=f"unlabeled_pie_{ts}")
            except Exception:
                dist_pie = None

            # ---- Evidently drift vs reference (optional) ----
            try:
                ev_drift_html = None
                ref_path = os.path.join(app.config["REPORT_FOLDER"], f"ref_{model_name}.csv")
                if os.path.exists(ref_path):
                    ref_df = pd.read_csv(ref_path).drop(
                        columns=[c for c in ["class", "label", "target"] if c in ref_df.columns],
                        errors="ignore",
                    )
                    cur = X.copy()
                    for c in ref_df.columns:
                        if c not in cur.columns:
                            cur[c] = 0
                    for c in cur.columns:
                        if c not in ref_df.columns:
                            ref_df[c] = 0
                    cur = cur[sorted(cur.columns)]
                    ref_df = ref_df[sorted(ref_df.columns)]
                    ev_drift_html = save_evidently_data_drift_html(
                        current=cur, reference=ref_df,
                        out_name=f"evidently_drift_{model_name}_{ts}.html"
                    )
            except Exception:
                ev_drift_html = None

            extra_reports = [p for p in [ge_html, ev_drift_html] if p]
            if extra_reports:
                flash("Validation reports: " + ", ".join(extra_reports), "info")

            flash("Predictions generated (no labels detected).", "info")
            return render_template(
                "results_unlabeled.html",
                download_file=out_name,
                distribution=dist,
                confhist_img=confhist_path,
                conf_bands=bands,
                dist_pie=dist_pie,
            )

    return render_template("detect.html", models=models_available)


# Static routes
@app.route("/models/<path:filename>")
def download_model(filename):
    return send_from_directory(app.config["MODEL_FOLDER"], filename, as_attachment=True)


@app.route("/reports/<path:filename>")
def download_report(filename):
    return send_from_directory(app.config["REPORT_FOLDER"], filename, as_attachment=True)


# Context processor to safely check existence of views in templates
@app.context_processor
def inject_endpoint_flags():
    from flask import current_app

    return {"has_admin_models": "admin_models" in current_app.view_functions}


# DB init
def prepare_db():
    db.create_all()
    print("✅ Database prepared")
    if not User.query.filter_by(email="admin@localhost").first():
        superadmin = User(
            fullname="Super Administrator",
            email="admin@localhost",
            password_hash=generate_password_hash("admin123"),
            role="superadmin",
            approved=True,
        )
        db.session.add(superadmin)
        db.session.commit()
        print("✅ Superadmin created (admin@localhost / admin123)")
    else:
        print("ℹ️ Superadmin already exists")


@app.route("/admin/delete_model/<string:model_name>", methods=["POST", "GET"])
@login_required
def delete_model(model_name):
    if current_user.role not in ["admin", "superadmin"]:
        flash("Unauthorized", "danger")
        return redirect(url_for("dashboard"))

    model = ModelRecord.query.filter_by(name=model_name).first()
    if model:
        filepath = os.path.join(app.config["MODEL_FOLDER"], f"{model.name}.pkl")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except (PermissionError, OSError):
                pass
        db.session.delete(model)
        db.session.commit()
        flash(f'Model "{model_name}" deleted.', "success")
    else:
        flash(f'Model "{model_name}" not found.', "warning")

    return redirect(url_for("admin_models"))


if __name__ == "__main__":
    with app.app_context():
        prepare_db()
    app.run(debug=True)
