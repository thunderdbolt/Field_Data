import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple

# ==========================================
# 1. FORENSIC LOGIC ENGINE
# ==========================================
class ForensicFlowAnalyzer:
    def __init__(self, df: pd.DataFrame, source_col: str, sink_col: str):
        self.df = df.copy()
        self.source_col = source_col
        self.sink_col = sink_col
        self.anomalies = []

    def detect_anomalies(self, threshold_std: float = 3.0, window: int = 10) -> List[int]:
        self.df['diff'] = self.df[self.source_col] - self.df[self.sink_col]
        rolling_mean = self.df['diff'].rolling(window=window).mean()
        rolling_std = self.df['diff'].rolling(window=window).std()
        
        upper_bound = rolling_mean + (threshold_std * rolling_std)
        lower_bound = rolling_mean - (threshold_std * rolling_std)
        
        self.df['is_anomaly'] = (self.df['diff'] > upper_bound) | (self.df['diff'] < lower_bound)
        self.anomalies = self.df[self.df['is_anomaly']].index.tolist()
        return self.anomalies

    def get_auto_calibration_factor(self) -> float:
        clean_df = self.df[~self.df['is_anomaly']]
        total_source = clean_df[self.source_col].sum()
        total_sink = clean_df[self.sink_col].sum()
        return total_sink / total_source if total_source != 0 else 1.0

    def get_optimal_lag(self, max_lag: int = 20) -> int:
        clean_df = self.df[~self.df['is_anomaly']].fillna(0)
        source = clean_df[self.source_col]
        sink = clean_df[self.sink_col]
        
        correlations = [source.shift(lag).fillna(0).corr(sink) for lag in range(-max_lag, max_lag + 1)]
        return range(-max_lag, max_lag + 1)[np.argmax(correlations)]

    def get_corrected_data(self, gain: float = 1.0, lag: int = 0, smoothing: int = 1) -> Tuple[pd.Series, pd.Series]:
        source_corrected = self.df[self.source_col].shift(lag) * gain
        
        if smoothing > 1:
            source_corrected = source_corrected.rolling(window=smoothing).mean()
            sink_smoothed = self.df[self.sink_col].rolling(window=smoothing).mean()
        else:
            sink_smoothed = self.df[self.sink_col]
            
        return source_corrected, sink_smoothed

# --- CALIBRATION & STACKING FUNCTION ---
@st.cache_data
def calibrate_raw_detailed_data(raw_df: pd.DataFrame, gain: float, lag: int, pressure_offset: float, meeting_times: list):
    """Processes granular source/sink node data based on aggregated meeting times."""
    source_cols = ['Sources/Sinks', 'NAME', 'Time (Hr)', 'Water Q (STB)', 'Pressure (psig)']
    source_cols_present = [c for c in source_cols if c in raw_df.columns]
    
    sources_df = raw_df[source_cols_present].dropna(subset=['Sources/Sinks']).copy()
    is_source = sources_df['Sources/Sinks'].str.lower() == 'source'
    
    # Apply Corrections safely using .loc
    if 'Water Q (STB)' in sources_df.columns and 'Time (Hr)' in sources_df.columns:
        sources_df.loc[is_source, 'Water Q (STB)'] = pd.to_numeric(sources_df.loc[is_source, 'Water Q (STB)'], errors='coerce') * gain
        sources_df.loc[is_source, 'Time (Hr)'] = pd.to_numeric(sources_df.loc[is_source, 'Time (Hr)'], errors='coerce') + lag
        
    if 'Pressure (psig)' in sources_df.columns:
        sources_df.loc[is_source, 'Pressure (psig)'] = pd.to_numeric(sources_df.loc[is_source, 'Pressure (psig)'], errors='coerce') + pressure_offset
    
    filtered_sources = sources_df[sources_df['Time (Hr)'].isin(meeting_times)]
    
    # Process Sinks (assuming .1 suffix structure from specific export tool)
    if 'Sources/Sinks.1' in raw_df.columns:
        sink_cols_present = [f"{c}.1" for c in source_cols if f"{c}.1" in raw_df.columns]
        sinks_df = raw_df[sink_cols_present].dropna(subset=['Sources/Sinks.1']).copy()
        sinks_df.rename(columns={f"{c}.1": c for c in source_cols}, inplace=True)
        
        filtered_sinks = sinks_df[pd.to_numeric(sinks_df['Time (Hr)'], errors='coerce').isin(meeting_times)]
        final_df = pd.concat([filtered_sources, filtered_sinks])
    else:
        final_df = filtered_sources

    # Aggregate Duplicates
    if 'Water Q (STB)' in final_df.columns:
        final_df['Water Q (STB)'] = pd.to_numeric(final_df['Water Q (STB)'], errors='coerce').fillna(0)
    if 'Pressure (psig)' in final_df.columns:
        final_df['Pressure (psig)'] = pd.to_numeric(final_df['Pressure (psig)'], errors='coerce').fillna(0)

    group_cols = [c for c in ['Sources/Sinks', 'NAME', 'Time (Hr)'] if c in final_df.columns]
    agg_dict = {col: 'sum' if col == 'Water Q (STB)' else 'mean' for col in ['Water Q (STB)', 'Pressure (psig)'] if col in final_df.columns}

    if agg_dict:
        final_df = final_df.groupby(group_cols, as_index=False).agg(agg_dict)
        
    return final_df.sort_values(by=['Time (Hr)', 'Sources/Sinks', 'NAME'])


# ==========================================
# 2. DATA LOADING & GENERATION
# ==========================================
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data # CRITICAL FIX: Caching prevents data from regenerating on slider moves
def generate_demo_data():
    np.random.seed(42)
    steps = 350
    base = np.random.normal(140000, 5000, steps)
    sink = base + np.random.normal(0, 2000, steps)
    source = (base * 0.85) + np.random.normal(0, 2000, steps)
    source[328:] = source[328:] * 1.5 
    sink[328:] = sink[328:] * 0.6
    return pd.DataFrame({'Source': source, 'Sink': sink})


# ==========================================
# 3. STREAMLIT UI & STATE SETUP
# ==========================================
st.set_page_config(page_title="Forensic Flow Workspace", layout="wide")
st.title("Forensic Flow Analyzer")

# Initialize Session States
if 'lag_val' not in st.session_state: st.session_state['lag_val'] = 0
if 'gain_val' not in st.session_state: st.session_state['gain_val'] = 1.00

with st.sidebar:
    st.header("1. Data Input")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    use_demo = st.checkbox("Use Demo Data", value=(uploaded_file is None))

    df, raw_full_df = None, None
    source_col, sink_col = 'Source', 'Sink'

    if uploaded_file:
        df = load_data(uploaded_file)
        raw_full_df = df.copy()
        
        st.subheader("Map Columns")
        all_cols = df.columns.tolist()
        source_col = st.selectbox("Select Source Column", all_cols, index=0)
        sink_col = st.selectbox("Select Sink Column", all_cols, index=1 if len(all_cols) > 1 else 0)
        
    elif use_demo:
        df = generate_demo_data()
        raw_full_df = df.copy()

    # Early exit if no data
    if df is None:
        st.info("Please upload a CSV or select Demo Data to begin.")
        st.stop()

    try:
        # Prepare the dataframe
        df[source_col] = pd.to_numeric(df[source_col], errors='coerce')
        df[sink_col] = pd.to_numeric(df[sink_col], errors='coerce')
        df = df.dropna(subset
