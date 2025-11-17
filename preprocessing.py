import pandas as pd
import numpy as np
from clean_fct import filter_cols
from sklearn.preprocessing import PowerTransformer, StandardScaler
from impute_by_date import impute_numeric_by_time
from copy import deepcopy

def preprocess_for_air_quality(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    date_col: str = "Date",
    place_col: str = "Place_ID",
    target_col: str = "target",
    impute_method: str = "weekly",
    scale: bool = True
):
    """
    Preprocess train and test without leakage.
    Returns: df_train_proc, df_test_proc, fitted_objects
    """

    # deep copies for safety
    train = deepcopy(df_train)
    test = deepcopy(df_test)

    # filter columns
    train = filter_cols(train)
    test = filter_cols(test)

    # target-related columns (these must be untouched)
    target_related = [target_col, 'target_min', 'target_max', 'target_variance', 'target_count']

    # numeric predictor cols
    numeric_cols = [
        c for c in train.select_dtypes(include=np.number).columns
        if c not in target_related
    ]

    # ---------- 1. IMPUTATION (fit on train only)
    train = impute_numeric_by_time(train, date_col, place_col, method=impute_method)
    test = impute_numeric_by_time(test, date_col, place_col, method=impute_method)

    # ---------- 2. Remove impossible negative values
    for col in numeric_cols:
        if "column" in col or col.endswith("_density"):
            train = train[train[col] >= -0.001]
            test = test[test[col] >= -0.001]

    # ---------- 3. Cap cloud values (fit on train only)
    cloud_cols = [c for c in numeric_cols if "cloud" in c.lower()]
    for col in cloud_cols:
        upper = train[col].quantile(0.99)
        train[col] = train[col].clip(upper=upper)
        test[col] = test[col].clip(upper=upper)

    # ---------- 4. Winsorize outliers (fit on train only)
    def cap_outliers(tr_series, te_series):
        low = tr_series.quantile(0.01)
        high = tr_series.quantile(0.99)
        return tr_series.clip(low, high), te_series.clip(low, high)

    for col in numeric_cols:
        train[col], test[col] = cap_outliers(train[col], test[col])

    # ---------- 5. Transform skewed features (>2) (fit on train only)
    skew_vals = train[numeric_cols].skew()
    skewed = skew_vals[abs(skew_vals) > 2].index.tolist()

    pt = None
    if len(skewed) > 0:
        pt = PowerTransformer(method="yeo-johnson")
        train[skewed] = pt.fit_transform(train[skewed])
        test[skewed] = pt.transform(test[skewed])

    # ---------- 6. Scaling (fit on train only)
    scaler = None
    if scale:
        scaler = StandardScaler()
        train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
        test[numeric_cols] = scaler.transform(test[numeric_cols])

    # ---------- 7. Drop remaining NAs
    train = train.dropna()
    test = test.dropna()

    # return everything
    return train, test, {"scaler": scaler, "pt": pt}
