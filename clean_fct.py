def filter_cols(df):
    #to use add: from clean_fct import filter_cols

    keep_cols = ['L3_CLOUD_cloud_fraction', 
                 'L3_CLOUD_cloud_base_height', 
                 'L3_CLOUD_cloud_optical_depth', 
                 'L3_AER_AI_absorbing_aerosol_index']

    mask = (
        (df.columns.str.startswith('L3')) & 
        (~df.columns.str.contains('column_number_density')) |
        df.columns.str.contains('slant|stratospheric|amf'))
    
    return df.loc[:, ~mask | df.columns.isin(keep_cols)]