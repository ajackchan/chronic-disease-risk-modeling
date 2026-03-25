import pandas as pd

from chronic_disease_risk.preprocessing.nhanes_merge import merge_nhanes_tables


def test_merge_nhanes_tables_merges_on_seqn_with_inner_join() -> None:
    demo = pd.DataFrame({"seqn": [1, 2], "age": [45, 60]})
    lab = pd.DataFrame({"seqn": [1, 3], "glu": [90.0, 110.0]})

    merged = merge_nhanes_tables({"demo": demo, "lab": lab})

    assert list(merged.columns) == ["seqn", "age", "glu"]
    assert merged.to_dict(orient="records") == [{"seqn": 1, "age": 45, "glu": 90.0}]
