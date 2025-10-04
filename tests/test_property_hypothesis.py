# tests/test_property_hypothesis.py
import string

import pandas as pd
from hypothesis import HealthCheck, given, settings, strategies as st

from app import (
    allowed_file,
    is_strong_password,
    pick_label_column,
    trim_to_10k_stratified,
)

# ----------------------------
# Strategies (generators)
# ----------------------------

SPECIALS = "!@#$%^&*()-_=+[]{};:',.<>/?\\|`~"
ALPHABET = string.ascii_letters + string.digits + SPECIALS


@st.composite
def strong_passwords(draw) -> str:
    """Generate strings that MUST satisfy is_strong_password:
    ≥6 chars, has upper/lower/digit/special."""
    u = draw(st.sampled_from(string.ascii_uppercase))
    lower_ch = draw(st.sampled_from(string.ascii_lowercase))
    d = draw(st.sampled_from(string.digits))
    s = draw(st.sampled_from(SPECIALS))
    # ensure total length ≥ 6
    rest_len = draw(st.integers(min_value=2, max_value=24))
    rest = "".join(
        draw(
            st.lists(
                st.sampled_from(ALPHABET),
                min_size=rest_len,
                max_size=rest_len,
            )
        )
    )
    # Build a list so we can shuffle deterministically via Hypothesis RNG
    chars = [u, lower_ch, d, s] + list(rest)
    rnd = draw(st.randoms())
    rnd.shuffle(chars)
    return "".join(chars)


@st.composite
def csv_filenames(draw) -> str:
    base = draw(
        st.text(
            alphabet=string.ascii_letters + string.digits + "_-",
            min_size=1,
            max_size=40,
        )
    )
    ext = draw(st.sampled_from([".csv", ".CSV", ".CsV"]))
    return base + ext


@st.composite
def non_csv_filenames(draw) -> str:
    base = draw(
        st.text(
            alphabet=string.ascii_letters + string.digits + "_-",
            min_size=1,
            max_size=40,
        )
    )
    ext = draw(st.sampled_from([".txt", ".json", ".xlsx", ".png", ".jpg", "", ".csv.bak"]))
    return base + ext


@st.composite
def df_with_keyword_labels(draw) -> pd.DataFrame:
    """Build a DataFrame that includes at least one of
    {'class','label','target'}."""
    keywords = ["class", "label", "target"]
    included = draw(
        st.lists(
            st.sampled_from(keywords),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    others = draw(
        st.lists(
            st.text(
                alphabet=string.ascii_lowercase,
                min_size=1,
                max_size=6,
            ).filter(lambda s: s not in keywords),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    cols: list[str] = included + others
    n = draw(st.integers(min_value=3, max_value=20))
    data = {
        c: draw(
            st.lists(
                st.integers(min_value=0, max_value=5),
                min_size=n,
                max_size=n,
            )
        )
        for c in cols
    }
    return pd.DataFrame(data)


@st.composite
def df_without_keyword_labels(draw) -> pd.DataFrame:
    """DataFrame with NO {'class','label','target'} so pick_label_column
    returns the LAST column."""
    keywords = {"class", "label", "target"}
    cols = draw(
        st.lists(
            st.text(
                alphabet=string.ascii_lowercase,
                min_size=1,
                max_size=6,
            ).filter(lambda s: s not in keywords),
            min_size=2,
            max_size=6,
            unique=True,
        )
    )
    n = draw(st.integers(min_value=3, max_value=20))
    data = {
        c: draw(
            st.lists(
                st.integers(min_value=0, max_value=5),
                min_size=n,
                max_size=n,
            )
        )
        for c in cols
    }
    return pd.DataFrame(data)


# ---------- Big frames generated deterministically (avoid Hypothesis buffer limits) ----------


@st.composite
def df_for_trimming_small(draw) -> pd.DataFrame:
    """≤ 10,000 rows; create features/labels deterministically."""
    n_classes = draw(st.integers(min_value=2, max_value=5))
    n = draw(st.integers(min_value=100, max_value=10_000))
    y = [i % n_classes for i in range(n)]
    f1 = list(range(n))
    f2 = [(i * 7) % 1001 for i in range(n)]
    return pd.DataFrame({"f1": f1, "f2": f2, "label": y})


@st.composite
def df_for_trimming_large(draw) -> pd.DataFrame:
    """> 10,000 rows; ensure n - 10_000 >= n_classes so SSS has enough test rows."""
    n_classes = draw(st.integers(min_value=2, max_value=5))
    # test_size >= n_classes (n - 10_000 plays the role of test_size for SSS)
    extra = draw(st.integers(min_value=n_classes, max_value=n_classes + 200))
    n = 10_000 + extra
    y = [i % n_classes for i in range(n)]
    f1 = list(range(n))
    f2 = [(i * 11) % 1001 for i in range(n)]
    return pd.DataFrame({"f1": f1, "f2": f2, "label": y})


# ----------------------------
# Properties
# ----------------------------


# 1) Password strength
@given(
    st.text(alphabet=ALPHABET, min_size=0, max_size=30).filter(
        lambda pw: (len(pw) < 6)
        or (not any(c.isupper() for c in pw))
        or (not any(c.islower() for c in pw))
        or (not any(c.isdigit() for c in pw))
        or (not any(c in SPECIALS for c in pw))
    )
)
@settings(deadline=None, max_examples=60)
def test_is_strong_password_rejects_weak(pw: str) -> None:
    assert is_strong_password(pw) is False


@given(strong_passwords())
@settings(deadline=None, max_examples=40)
def test_is_strong_password_accepts_strong(pw: str) -> None:
    assert is_strong_password(pw) is True


# 2) allowed_file
@given(csv_filenames())
def test_allowed_file_accepts_csv(fname: str) -> None:
    assert allowed_file(fname) is True


@given(non_csv_filenames())
def test_allowed_file_rejects_non_csv(fname: str) -> None:
    assert allowed_file(fname) is False


# 3) pick_label_column
@given(df_with_keyword_labels())
def test_pick_label_column_prefers_class_then_label_then_target(
    df: pd.DataFrame,
) -> None:
    chosen = pick_label_column(df)
    for key in ["class", "label", "target"]:
        if key in df.columns:
            assert chosen == key
            break


@given(df_without_keyword_labels())
def test_pick_label_column_falls_back_to_last_column(df: pd.DataFrame) -> None:
    assert pick_label_column(df) == df.columns[-1]


# 4) trim_to_10k_stratified (split)
@given(df_for_trimming_small())
@settings(deadline=None, max_examples=8)
def test_trim_to_10k_stratified_small(df: pd.DataFrame) -> None:
    out = trim_to_10k_stratified(df, "label")
    assert len(out) == len(df)
    assert set(out["label"].unique()) == set(df["label"].unique())


@given(df_for_trimming_large())
@settings(
    deadline=None,
    max_examples=4,
    suppress_health_check=[
        HealthCheck.large_base_example,
    ],
)
def test_trim_to_10k_stratified_large(df: pd.DataFrame) -> None:
    out = trim_to_10k_stratified(df, "label")
    assert len(out) == 10_000
    assert set(out["label"].unique()) == set(df["label"].unique())
