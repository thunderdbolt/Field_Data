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
        # Using MEDIAN instead of SUM to ignore massive spikes/drops
        clean_df = self.df[~self.df['is_anomaly']]
        
        median_source = clean_df[self.source_col].median()
        median_sink = clean_df[self.sink_col].median()
        
        return median_sink / median_source if median_source != 0 else 1.0

    def get_optimal_lag(self, max_lag: int = 20) -> int:
        clean_df = self.df[~self.df['is_anomaly']].fillna(0)
        
        # PRE-SMOOTH the data to remove jitter before correlating
        # A rolling median of 5 helps flatten out single-point spikes
        source_smoothed = clean_df[self.source_col].rolling(window=5, min_periods=1).median()
        sink_smoothed = clean_df[self.sink_col].rolling(window=5, min_periods=1).median()
        
        correlations = []
        lags = range(-max_lag, max_lag + 1)
        
        for lag in lags:
            s_shifted = source_smoothed.shift(lag).fillna(0)
            # Use 'spearman' correlation which evaluates trend, ignoring magnitude outliers
            correlations.append(s_shifted.corr(sink_smoothed, method='spearman'))
        
        return lags[np.argmax(correlations)]

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
        df = df.dropna(subset=[source_col, sink_col])

        analyzer = ForensicFlowAnalyzer(df, source_col, sink_col)
        anomalies = analyzer.detect_anomalies()
        suggested_lag = analyzer.get_optimal_lag()
        suggested_gain = analyzer.get_auto_calibration_factor()
    except Exception as e:
        st.error(f"Error processing data mapping: {e}")
        st.stop()

# ==========================================
# 4. FORENSIC CONTROLS
# ==========================================
with st.sidebar:
    st.divider()
    st.header("2. Forensic Controls")
    
    st.subheader("Time Alignment")
    st.info(f"Detected Lag: **{suggested_lag} units**")
    if st.button("Apply Auto-Lag", use_container_width=True):
        st.session_state['lag_val'] = int(suggested_lag)
        
    lag = st.slider("Manual Lag Adjustment", -20, 20, key='lag_val')

    st.subheader("Meter Calibration")
    st.info(f"Suggested Gain: **{suggested_gain:.3f}**")
    if st.button("Apply Auto-Calibration", use_container_width=True):
        st.session_state['gain_val'] = float(suggested_gain)

    gain = st.number_input("Calibration Factor", 0.0, 5.0, step=0.01, format="%.3f", key='gain_val')

    pressure_offset = st.number_input("Pressure Offset (psig)", -500.0, 500.0, value=0.0, step=5.0, 
                                      help="Accounts for sensor zero-drift or elevation.")

    st.divider()
    smoothing = st.slider("Smoothing Window", 1, 20, 1)
    
    st.subheader("Meeting Points Criteria")
    tolerance_pct = st.slider("Tolerance %", 0.0, 10.0, 0.32, step=0.01) / 100.0

# ==========================================
# 5. MAIN VISUALIZATION
# ==========================================
source_corrected, sink_smoothed = analyzer.get_corrected_data(gain, lag, smoothing)
diff = source_corrected - sink_smoothed

# Top Metrics
c1, c2, c3 = st.columns(3)
with c1: st.metric("Total Sink Volume", f"{sink_smoothed.sum()/1e6:.2f} M")
with c2: st.metric("Total Source (Adj)", f"{source_corrected.sum()/1e6:.2f} M")
with c3: 
    balance_err = (source_corrected.sum() - sink_smoothed.sum()) / sink_smoothed.sum() * 100 if sink_smoothed.sum() != 0 else 0
    st.metric("Net Imbalance", f"{balance_err:.2f} %", delta_color="inverse")

if anomalies:
    st.warning(f"⚠️ BURST/ANOMALY DETECTED (Index {min(anomalies)} - {max(anomalies)})")

tab1, tab2, tab3 = st.tabs(["Forensic View", "Raw Data", "Meeting Points (Export)"])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        subplot_titles=("Flow Rate Comparison", "Residual Analysis"))

    # Main Plot
    fig.add_trace(go.Scatter(x=df.index, y=sink_smoothed, mode='lines', 
                             name=f'Sink ({sink_col})', line=dict(color='#ff7f0e')), row=1, col=1) # Plotly Orange
    fig.add_trace(go.Scatter(x=df.index, y=source_corrected, mode='lines', 
                             name=f'Source ({source_col})', line=dict(color='#1f77b4')), row=1, col=1) # Plotly Blue

    # Residuals
    fig.add_trace(go.Scatter(x=df.index, y=diff, mode='lines', name='Delta', 
                             line=dict(color='gray')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=diff, fill='tozeroy', mode='none', 
                             fillcolor='rgba(255, 0, 0, 0.2)', showlegend=False), row=2, col=1)
    
    # NEW: Add a zero-line to residuals to easily see balance
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)

    if anomalies:
        fig.add_vrect(x0=min(anomalies), x1=max(anomalies), 
                      fillcolor="red", opacity=0.15, layer="below")

    fig.update_layout(height=600, hovermode="x unified", margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.dataframe(df, use_container_width=True)

with tab3:
    st.markdown("### 1. Aggregated Meeting Points")
    st.write(f"Showing times where the total calibrated source is within **{tolerance_pct*100:.2f}%** of the total sink.")
    
    is_meeting = abs(diff) <= (abs(sink_smoothed) * tolerance_pct)
    meeting_times = df.index[is_meeting].tolist() 
    
    meeting_df = pd.DataFrame({
        'Time_Index': meeting_times,
        'Calibrated_Source': source_corrected[is_meeting].round(2),
        'Smoothed_Sink': sink_smoothed[is_meeting].round(2),
        'Difference': diff[is_meeting].round(2)
    })
    
    if meeting_df.empty:
        st.warning("No meeting points found with current calibration and tolerance settings.")
    else:
        st.dataframe(meeting_df, use_container_width=True)
        
        st.divider()
        st.markdown("### 2. Export Filtered & Calibrated Node History")
        
        required_cols = ['Sources/Sinks', 'NAME', 'Time (Hr)', 'Water Q (STB)']
        
        if all(col in raw_full_df.columns for col in required_cols):
            calibrated_raw_df = calibrate_raw_detailed_data(raw_full_df, gain, int(lag), pressure_offset, meeting_times)
            st.dataframe(calibrated_raw_df, use_container_width=True) 
            
            st.download_button(
                label="Download Calibrated Node History (CSV)",
                data=calibrated_raw_df.to_csv(index=False).encode('utf-8'),
                file_name='calibrated_meeting_nodes.csv',
                mime='text/csv',
                type='primary'
            )
        else:
            st.warning(f"Detailed export requires columns: `{', '.join(required_cols)}`. Ensure your uploaded CSV contains these headers.")

# ==========================================
# 6. DIAGNOSTIC EXPLANATION
# ==========================================
st.divider()
with st.expander("How to interpret this analysis"):
    st.markdown("""
    1.  **Calibration Error:** If the *Blue* and *Orange* lines have the same shape but different heights, adjust the **Calibration Factor** until they overlap. The `Suggested Gain` calculates this for you.
    2.  **Timing Mismatch:** If the peaks are misaligned, adjust the **Lag Slider**.
    3.  **Pressure Offset:** Use the new control to dial in your field gauges.
    4.  **Leaks/Bursts:** Look at the bottom chart (Residuals).
        * **Flat Line @ 0:** Perfect balance.
        * **Negative Dip (Red):** Potential Leak or Burst.
        * **Positive Spike:** Tank filling or sensor error.
    """)
