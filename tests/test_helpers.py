import pandas as pd

from app import allowed_file, pick_label_column, trim_to_10k_stratified


def test_allowed_file():
    assert allowed_file("a.csv")
    assert not allowed_file("b.txt")


def test_pick_label_column():
    df = pd.DataFrame({"x": [1, 2], "label": [0, 1]})
    assert pick_label_column(df) == "label"
    df2 = pd.DataFrame({"x": [1, 2], "target": [0, 1]})
    assert pick_label_column(df2) == "target"


def test_trim_to_10k_stratified():
    df = pd.DataFrame({"x": range(12000), "label": [i % 2 for i in range(12000)]})
    small = trim_to_10k_stratified(df, "label")
    assert len(small) == 10000
