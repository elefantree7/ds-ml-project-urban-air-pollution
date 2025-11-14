import pandas as pd
import numpy as np
from clean_fct import filter_cols
from sklearn.preprocessing import PowerTransformer, StandardScaler
from impute_by_date import impute_numeric_by_time
from copy import deepcopy

def preprocess_for_air_quality(
    df: pd.DataFrame,
    df_impute: pd.DataFrame,
    date_col: str = "Date",
    place_col: str = "Place_ID",
    target_col: str = "target",
    impute_method: str = "weekly",
    scale: bool = True
) -> pd.DataFrame:
    """
    Preprocess predictor columns for the Air Pollution Challenge.
    
    Steps:
    0) Filter only the columns you want
    1) Time-aware imputation of missing numeric predictor values
    2) Remove impossible vertical column values (< -0.001 mol/m²)
    3) Cap extremely high cloud values (99th percentile)
    4) Winsorize extreme values (1st-99th percentile)
    5) Transform highly skewed features
    6) Optional scaling
    Target and target-derived columns are untouched.
    """


    df_proc = deepcopy(df)
    #df_impute_proc = deepcopy(df_impute)

    # 0) Filter only the columns you want
    df_proc = filter_cols(df_proc)
    #df_impute_proc = filter_cols(df_impute_proc)

    # Identify numeric columns excluding target and target-related
    target_related_cols = [target_col, 'target_min', 'target_max', 'target_variance', 'target_count']
    numeric_cols = [c for c in df_proc.select_dtypes(include=np.number).columns if c not in target_related_cols]

    # 1) Impute missing values using the time-aware function
    df_proc = impute_numeric_by_time(df_impute, date_col=date_col, place_col=place_col, method=impute_method)

    # 2) Remove impossible negative vertical column densities
    for col in numeric_cols:
        if "column" in col or col.endswith("_density"):
            df_proc = df_proc[df_proc[col] >= -0.001]

    # 3) Cap extremely high cloud values at 99th percentile
    cloud_cols = [c for c in numeric_cols if "cloud" in c.lower()]
    for col in cloud_cols:
        upper_limit = df_proc[col].quantile(0.99)
        df_proc[col] = df_proc[col].clip(lower=None, upper=upper_limit)

    # 4) Winsorize all numeric columns (1st-99th percentile)
    def cap_outliers(series, lower_quantile=0.01, upper_quantile=0.99):
        lower = series.quantile(lower_quantile)
        upper = series.quantile(upper_quantile)
        return series.clip(lower, upper)

    for col in numeric_cols:
        df_proc[col] = cap_outliers(df_proc[col])

    # 5) Transform highly skewed features (>2)
    skewed_features = df_proc[numeric_cols].skew().sort_values(ascending=False)
    high_skew = skewed_features[abs(skewed_features) > 2].index.tolist()
    if high_skew:
        pt = PowerTransformer(method='yeo-johnson')
        df_proc[high_skew] = pt.fit_transform(df_proc[high_skew])

    # 6) Optional scaling
    if scale:
        scaler = StandardScaler()
        df_proc[numeric_cols] = scaler.fit_transform(df_proc[numeric_cols])
    
    # 7) drop last na
    df_proc = df_proc.dropna()

    return df_proc
