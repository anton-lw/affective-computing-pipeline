# d_temporal_analyzer.py
import pandas as pd
import numpy as np
import ruptures as rpt

def aggregate_to_daily_time_series(df):
    """Aggregates the annotated DataFrame into a daily time series."""
    print("\n--- Aggregating data into a daily time series ---")
    if df is None or 'vader_compound' not in df.columns:
        print("ERROR: Annotated data is missing required columns for aggregation.")
        return None
    
    df_ts = df.set_index('timestamp')
    
    agg_dict = {
        'vader_compound': ['mean', 'std'],
        'i_talk_freq': 'mean',
        'absolutist_freq': 'mean',
    }
    # Dynamically add all NRC columns to the aggregation dictionary
    nrc_cols = [col for col in df.columns if col.startswith('nrc_')]
    for col in nrc_cols:
        agg_dict[col] = 'mean'
        
    daily_data = df_ts.resample('D').agg(agg_dict)
    daily_data.columns = ['_'.join(col).strip() for col in daily_data.columns.values]
    return daily_data.dropna()

def detect_change_points(time_series_data, column_name='vader_compound_mean', window=30, pen_multiplier=2.0):
    """Detects change points in a specific column of the time series data."""
    print(f"\n--- Detecting change points in '{column_name}' ---")
    signal = time_series_data[column_name].rolling(window=window).mean().dropna()
    
    if len(signal) < 2:
        print("Not enough data to perform change point detection.")
        return signal, []
    
    algo = rpt.Pelt(model="rbf").fit(signal.values)
    penalty = np.log(len(signal)) * signal.var() * pen_multiplier
    result = algo.predict(pen=penalty)
    
    change_points = [signal.index[i-1] for i in result if i < len(signal)]
    print(f"Detected {len(change_points)} significant change point(s).")
    return signal, change_points