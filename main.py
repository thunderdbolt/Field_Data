import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. FORENSIC LOGIC ENGINE
# ==========================================
class ForensicFlowAnalyzer:
    def __init__(self, df, source_col, sink_col):
        self.df = df.copy()
        self.source_col = source_col
        self.sink_col = sink_col
        self.anomalies = []

    def detect_anomalies(self, threshold_std=3.0, window=10):
        self.df['diff'] = self.df[self.source_col] - self.df[self.sink_col]
        rolling_mean = self.df['diff'].rolling(window=window).mean()
        rolling_std = self.df['diff'].rolling(window=window).std()
        
        upper_bound = rolling_mean + (threshold_std * rolling_std)
        lower_bound = rolling_mean - (threshold_std * rolling_std)
        
        self.df['is_anomaly'] = (self.df['diff'] > upper_bound) | (self.df['diff'] < lower_bound)
        self.anomalies = self.df[self.df['is_anomaly']].index.tolist()
        return self.anomalies

    def get_auto_calibration_factor(self):
        clean_df = self.df[~self.df['is_anomaly']]
        total_source = clean_df[self.source_col].sum()
        total_sink = clean_df[self.sink_col].sum()
        return total_sink / total_source if total_source != 0 else 1.0

    def get_optimal_lag(self, max_lag=20):
        clean_df = self.df[~self.df['is_anomaly']].fillna(0)
        source = clean_df[self.source_col]
        sink = clean_df[self.sink_col]
        
        correlations = []
        lags = range(-max_lag, max_lag + 1)
        for lag in lags:
            s_shifted = source.shift(lag).fillna(0)
            correlations.append(s_shifted.corr(sink))
        
        return lags[np.argmax(correlations)]

    def get_corrected_data(self, gain=1.0, lag=0, smoothing=1):
        source_shifted = self.df[self.source_col].shift(lag)
        source_corrected = source_shifted * gain
        
        if smoothing > 1:
            source_corrected = source_corrected.rolling(window=smoothing).mean()
            sink_smoothed = self.df[self.sink_col].rolling(window=smoothing).mean()
        else:
            sink_smoothed = self.df[self.sink_col]
            
        return source_corrected, sink_smoothed

# --- CALIBRATION & STACKING FUNCTION ---
def calibrate_raw_detailed_data(raw_df, gain, lag, pressure_offset, meeting_times):
    # 1. Extract the Source columns (left side of your CSV)
    source_cols = ['Sources/Sinks', 'NAME', 'Time (Hr)', 'Water Q (STB)', 'Pressure (psig)']
    source_cols_present = [c for c in source_cols if c in raw_df.columns]
    
    sources_df = raw_df[source_cols_present].copy()
    sources_df = sources_df.dropna(subset=['Sources/Sinks']) 
    
    # Apply Gain, Lag, and Pressure Offset to Sources only
    is_source = sources_df['Sources/Sinks'].str.lower() == 'source'
    
    if 'Water Q (STB)' in sources_df.columns and 'Time (Hr)' in sources_df.columns:
        sources_df.loc[is_source, 'Water Q (STB)'] = pd.to_numeric(sources_df.loc[is_source, 'Water Q (STB)'], errors='coerce') * gain
        sources_df.loc[is_source, 'Time (Hr)'] = pd.to_numeric(sources_df.loc[is_source, 'Time (Hr)'], errors='coerce') + lag
        
    if 'Pressure (psig)' in sources_df.columns:
        sources_df.loc[is_source, 'Pressure (psig)'] = pd.to_numeric(sources_df.loc[is_source, 'Pressure (psig)'], errors='coerce') + pressure_offset
    
    # Filter for Meeting Times
    filtered_sources = sources_df[sources_df['Time (Hr)'].isin(meeting_times)]
    
    # 2. Extract the Sink columns (middle of your CSV)
    if 'Sources/Sinks.1' in raw_df.columns:
        sink_cols = ['Sources/Sinks.1', 'NAME.1', 'Time (Hr).1', 'Water Q (STB).1', 'Pressure (psig).1']
        sink_cols_present = [c for c in sink_cols if c in raw_df.columns]
        sinks_df = raw_df[sink_cols_present].copy()
        
        rename_dict = {f"{c}.1": c for c in source_cols}
        sinks_df.rename(columns=rename_dict, inplace=True)
        sinks_df = sinks_df.dropna(subset=['Sources/Sinks'])
        
        filtered_sinks = sinks_df[pd.to_numeric(sinks_df['Time (Hr)'], errors='coerce').isin(meeting_times)]
        
        # 3. Stack them vertically!
        final_df = pd.concat([filtered_sources, filtered_sinks])
    else:
        final_df = filtered_sources

    # --- NEW: AGGREGATE DUPLICATES UPSTREAM ---
    if 'Water Q (STB)' in final_df.columns:
        final_df['Water Q (STB)'] = pd.to_numeric(final_df['Water Q (STB)'], errors='coerce').fillna(0)
    if 'Pressure (psig)' in final_df.columns:
        final_df['Pressure (psig)'] = pd.to_numeric(final_df['Pressure (psig)'], errors='coerce').fillna(0)

    # Group by the identifiers and apply math to the values
    group_cols = [c for c in ['Sources/Sinks', 'NAME', 'Time (Hr)'] if c in final_df.columns]
    agg_dict = {}
    if 'Water Q (STB)' in final_df.columns:
        agg_dict['Water Q (STB)'] = 'sum'
    if 'Pressure (psig)' in final_df.columns:
        agg_dict['Pressure (psig)'] = 'mean'

    if agg_dict:
        final_df = final_df.groupby(group_cols, as_index=False).agg(agg_dict)
    # ------------------------------------------
        
    return final_df.sort_values(by=['Time (Hr)', 'Sources/Sinks', 'NAME'])

# ==========================================
# 2. STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="Forensic Flow Workspace", layout="wide")
st.title("Forensic Flow Analyzer")

# ==========================================
# 3. DATA LOADING & COLUMN MAPPING
# ==========================================
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def generate_demo_data():
    np.random.seed(42)
    steps = 350
    base = np.random.normal(140000, 5000, steps)
    sink = base + np.random.normal(0, 2000, steps)
    source = (base * 0.85) + np.random.normal(0, 2000, steps)
    source[328:] = source[328:] * 1.5 
    sink[328:] = sink[328:] * 0.6
    return pd.DataFrame({'Source': source, 'Sink': sink})

with st.sidebar:
    st.header("1. Data Input")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    use_demo = st.checkbox("Use Demo Data", value=(uploaded_file is None))

    df = None
    raw_full_df = None
    source_col = 'Source'
    sink_col = 'Sink'

    if uploaded_file:
        df = load_data(uploaded_file)
        st.success("File Uploaded!")
        
        # Save an untouched copy of the raw data before dropna ruins the granular data
        raw_full_df = df.copy()
        
        st.subheader("Map Columns")
        all_cols = df.columns.tolist()
        source_col = st.selectbox("Select Source Column", all_cols, index=0)
        sink_col = st.selectbox("Select Sink Column", all_cols, index=1 if len(all_cols) > 1 else 0)
        
    elif use_demo:
        df = generate_demo_data()
        raw_full_df = df.copy()

    if df is not None:
        try:
            # Prepare the totals dataframe for the forensic analyzer
            df[source_col] = pd.to_numeric(df[source_col], errors='coerce')
            df[sink_col] = pd.to_numeric(df[sink_col], errors='coerce')
            df = df.dropna(subset=[source_col, sink_col])

            analyzer = ForensicFlowAnalyzer(df, source_col, sink_col)
            anomalies = analyzer.detect_anomalies()
            suggested_lag = analyzer.get_optimal_lag()
            suggested_gain = analyzer.get_auto_calibration_factor()
        except Exception as e:
            st.error(f"Error processing data: {e}")
            st.stop()
    else:
        st.info("Please upload a CSV or select Demo Data.")
        st.stop()

# ==========================================
# 4. FORENSIC CONTROLS
# ==========================================
with st.sidebar:
    st.divider()
    st.header("2. Forensic Controls")
    
    st.subheader("Time Alignment")
    st.info(f"Detected Lag: **{suggested_lag} units**")
    if st.button("Apply Auto-Lag"):
        st.session_state['lag_val'] = suggested_lag
        
    lag = st.slider("Manual Lag Adjustment", -20, 20, 
                    value=st.session_state.get('lag_val', 0))

    st.subheader("Meter Calibration")
    st.info(f"Suggested Gain: **{suggested_gain:.3f}**")
    if st.button("Apply Auto-Calibration"):
        st.session_state['gain_val'] = suggested_gain

    gain = st.number_input("Calibration Factor", 0.0, 5.0, 
                           value=st.session_state.get('gain_val', 1.00), step=0.01, format="%.3f")

    # --- RE-ADDED PRESSURE OFFSET CONTROL ---
    pressure_offset = st.number_input("Pressure Offset (psig)", -500.0, 500.0, 
                                      value=0.0, step=5.0, 
                                      help="Adds or subtracts a flat value to source pressures to account for sensor zero-drift or elevation.")

    st.divider()
    smoothing = st.slider("Smoothing Window", 1, 20, 1)
    
    st.subheader("Meeting Points Criteria")
    tolerance_pct = st.slider("Tolerance %", 0.0, 10.0, 0.32, 
                              help="How close do the lines need to be to be considered 'meeting'?", 
                              step=0.01) / 100.0

# ==========================================
# 5. MAIN VISUALIZATION
# ==========================================
source_corrected, sink_smoothed = analyzer.get_corrected_data(gain, lag, smoothing)
diff = source_corrected - sink_smoothed

c1, c2, c3 = st.columns(3)
with c1: st.metric("Total Sink Volume", f"{sink_smoothed.sum()/1e6:.2f} M")
with c2: st.metric("Total Source (Adj)", f"{source_corrected.sum()/1e6:.2f} M")
with c3: 
    balance_err = (source_corrected.sum() - sink_smoothed.sum()) / sink_smoothed.sum() * 100
    st.metric("Net Imbalance", f"{balance_err:.2f} %", delta_color="inverse")

if anomalies:
    st.warning(f"BURST DETECTED (Idx {min(anomalies)}-{max(anomalies)})")

tab1, tab2, tab3 = st.tabs(["Forensic View", "Raw Data", "Meeting Points (Export)"])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        subplot_titles=("Flow Rate Comparison", "Residual Analysis"))

    fig.add_trace(go.Scatter(x=df.index, y=sink_smoothed, mode='lines', 
                             name=f'Sink ({sink_col})', line=dict(color='orange')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=source_corrected, mode='lines', 
                             name=f'Source ({source_col})', line=dict(color='blue')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=diff, mode='lines', name='Delta', 
                             line=dict(color='gray')), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=diff, fill='tozeroy', mode='none', 
                             fillcolor='rgba(255, 0, 0, 0.2)', showlegend=False), row=2, col=1)

    if anomalies:
        fig.add_vrect(x0=min(anomalies), x1=max(anomalies), 
                      fillcolor="red", opacity=0.15, layer="below")

    fig.update_layout(height=600, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.dataframe(df)

with tab3:
    st.markdown("### 1. Aggregated Meeting Points")
    st.write(f"Showing times where the total calibrated source is within **{tolerance_pct*100:.2f}%** of the total sink.")
    
    # Find the indices (times) where they meet
    is_meeting = abs(diff) <= (abs(sink_smoothed) * tolerance_pct)
    
    # If using Demo data or standard index, we extract the index.
    # Otherwise, extract the actual Time column if that represents the meeting time.
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
        st.info(f"Filtering the raw dataset to only show individual node data for the **{len(meeting_times)}** meeting times discovered above.")
        
        # Check if the raw data format exists in the pristine dataframe
        required_cols = ['Sources/Sinks', 'NAME', 'Time (Hr)', 'Water Q (STB)']
        
        if all(col in raw_full_df.columns for col in required_cols):
            
            # --- FIXED: Added pressure_offset to the function call! ---
            calibrated_raw_df = calibrate_raw_detailed_data(raw_full_df, gain, int(lag), pressure_offset, meeting_times)
            
            st.dataframe(calibrated_raw_df, use_container_width=True) 
            
            csv_raw = calibrated_raw_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Calibrated Node History (CSV)",
                data=csv_raw,
                file_name='calibrated_meeting_nodes.csv',
                mime='text/csv',
                type='primary'
            )
        else:
            st.warning(f"Detailed export requires columns: {required_cols}. Please ensure your uploaded CSV contains these exact headers.")

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
