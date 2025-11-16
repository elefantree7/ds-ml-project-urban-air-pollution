import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_minimal(
    df: pd.DataFrame,
    y: pd.Series = None,
    fit_data: pd.DataFrame = None,
    scale: bool = True
) -> tuple:
    """
    MINIMAL preprocessing that preserves signal.
    Only does: imputation + scaling
    """
    
    df_proc = deepcopy(df)
    y_proc = deepcopy(y) if y is not None else None
    
    if fit_data is None:
        fit_data = df_proc
    else:
        fit_data = deepcopy(fit_data)
    
    # 1. Drop non-numeric columns
    non_numeric_cols = ['Date', 'Place_ID', 'Place_ID X Date']
    df_proc = df_proc.drop(columns=non_numeric_cols, errors='ignore')
    fit_data = fit_data.drop(columns=non_numeric_cols, errors='ignore')
    
    # 2. Drop target-related columns
    target_cols = ['target', 'target_min', 'target_max', 'target_variance', 'target_count']
    df_proc = df_proc.drop(columns=target_cols, errors='ignore')
    fit_data = fit_data.drop(columns=target_cols, errors='ignore')
    
    # 3. Keep only numeric columns
    numeric_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
    df_proc = df_proc[numeric_cols]
    fit_data = fit_data[numeric_cols]
    
    # 4. Simple median imputation
    imputer = SimpleImputer(strategy='median')
    imputer.fit(fit_data)
    
    df_proc = pd.DataFrame(
        imputer.transform(df_proc),
        columns=numeric_cols
    )
    
    # 5. Optional scaling
    if scale:
        scaler = StandardScaler()
        scaler.fit(fit_data)
        
        df_proc = pd.DataFrame(
            scaler.transform(df_proc),
            columns=numeric_cols
        )
    
    # 6. Reset index
    df_proc = df_proc.reset_index(drop=True)
    if y_proc is not None:
        y_proc = y_proc.reset_index(drop=True)
    
    if y is not None:
        return df_proc, y_proc
    return df_proc