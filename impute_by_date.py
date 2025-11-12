def impute_numeric_by_time(
    df: pd.DataFrame,
    date_col: str = "Date",
    place_col: str = "Place_ID",
    method: str = "weekly",
    rolling_window: int = 7
) -> pd.DataFrame:
    """
    Imputet fehlende Werte nur in numerischen Spalten.
    Datum und IDs werden ignoriert.
    
    Methoden:
      - 'weekly':    ersetzt NaNs durch Wochenmittel je Ort
      - 'daily_prev': ersetzt NaNs durch Vortagswert je Ort,
                      sonst rollierendes Mittel aus Vergangenheit
    """
    df = df.copy()

    # --- 1) Datums-Spalte konvertieren & sortieren ---
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, place_col]).sort_values([place_col, date_col]).reset_index(drop=True)

    # --- 2) Nur numerische Spalten auswählen ---
    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in (date_col, place_col)]

    if not numeric_cols:
        print("⚠️ Keine numerischen Spalten zum Imputen gefunden.")
        return df

    # --- 3) Imputation nach Methode ---
    if method == "weekly":
        iso = df[date_col].dt.isocalendar()
        df["_iso_year"], df["_iso_week"] = iso.year, iso.week

        for col in numeric_cols:
            weekly_mean = (
                df.groupby([place_col, "_iso_year", "_iso_week"])[col]
                  .transform("mean")
            )
            df[col] = df[col].fillna(weekly_mean)

        df.drop(columns=["_iso_year", "_iso_week"], inplace=True)

    elif method == "daily_prev":
        for col in numeric_cols:
            # 1) Vortagswert pro Ort
            df[col] = (
                df.groupby(place_col, group_keys=False)[col]
                  .apply(lambda s: s.fillna(s.shift(1)))
            )
            # 2) Rollierendes Mittel aus Vergangenheit
            df[col] = (
                df.groupby(place_col, group_keys=False)[col]
                  .apply(lambda s: s.fillna(
                      s.shift(1).rolling(rolling_window, min_periods=1).mean()
                  ))
            )
            # 3) Fallback: Median pro Ort
            df[col] = df[col].fillna(df.groupby(place_col)[col].transform("median"))
    else:
        raise ValueError("method must be 'weekly' or 'daily_prev'")

    return df
