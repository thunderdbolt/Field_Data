import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. FORENSIC LOGIC ENGINE
# ==========================================
class PipelineDataReconciler:
    def __init__(self, df: pd.DataFrame, source_q: str, sink_q: str, source_p: str = None, sink_p: str = None):
        self.df = df.copy()
        
        # 1. STRICT OVERLAP BOUNDING BOX
        valid_src = self.df[self.df[source_q].notnull()].index
        valid_snk = self.df[self.df[sink_q].notnull()].index
        
        if len(valid_src) > 0 and len(valid_snk) > 0:
            start_idx = max(valid_src.min(), valid_snk.min())
            end_idx = min(valid_src.max(), valid_snk.max())
            self.df = self.df.loc[start_idx:end_idx].reset_index(drop=True)
        
        self.source_q = source_q
        self.sink_q = sink_q
        self.source_p = source_p
        self.sink_p = sink_p
        self.df['Reconciled_Source_Q'] = self.df[self.source_q]
        self.df['Active_Gain'] = 1.0
        self.df['Trust_Score'] = 'N/A'

    def clean_and_stitch_sinks(self, spike_threshold: float = 3.0, window: int = 5):
        rolling_median = self.df[self.sink_q].rolling(window, center=True, min_periods=1).median()
        mad = (self.df[self.sink_q] - rolling_median).abs().rolling(window, center=True, min_periods=1).median()
        
        is_high_peak = (self.df[self.sink_q] - rolling_median) > (spike_threshold * mad)
        self.df.loc[is_high_peak, self.sink_q] = np.nan
        self.df[self.sink_q] = self.df[self.sink_q].interpolate(method='linear')
        return self

    def align_time_lag(self, max_lag: int = 20):
        smooth_source = self.df[self.source_q].rolling(3, min_periods=1).mean()
        smooth_sink = self.df[self.sink_q].rolling(3, min_periods=1).mean()

        correlations = [smooth_source.shift(lag).corr(smooth_sink) for lag in range(-max_lag, max_lag + 1)]
        best_lag = range(-max_lag, max_lag + 1)[np.nanargmax(correlations)]
        
        self.df['Reconciled_Source_Q'] = self.df['Reconciled_Source_Q'].shift(best_lag).bfill().ffill()
        return best_lag

    def segment_calibration(self, dt: int):
        segment_metrics = []
        col_idx_q = self.df.columns.get_loc('Reconciled_Source_Q')
        col_idx_g = self.df.columns.get_loc('Active_Gain')
        col_idx_t = self.df.columns.get_loc('Trust_Score')
        
        for start_idx in range(0, len(self.df), dt):
            end_idx = min(start_idx + dt, len(self.df))
            chunk = self.df.iloc[start_idx:end_idx]
            
            sum_sink = chunk[self.sink_q].sum()
            sum_source = chunk['Reconciled_Source_Q'].sum()
            local_gain = sum_sink / sum_source if sum_source != 0 else 1.0
            
            self.df.iloc[start_idx:end_idx, col_idx_q] *= local_gain
            self.df.iloc[start_idx:end_idx, col_idx_g] = local_gain
            
            avg_dp, trust_score = np.nan, "N/A"
            if self.source_p and self.sink_p:
                chunk_dp = chunk[self.source_p] - chunk[self.sink_p]
                chunk_q = self.df.iloc[start_idx:end_idx, col_idx_q] 
                avg_dp = chunk_dp.mean()
                
                if chunk_q.std() > 0 and chunk_dp.std() > 0:
                    corr = chunk_q.corr(chunk_dp, method='spearman')
                    if corr >= 0.4: trust_score = "🟢 High (Physics Align)"
                    elif corr >= 0: trust_score = "🟡 Low (Noisy)"
                    else: trust_score = "🔴 FAIL (Inverse Physics)"
                else: trust_score = "⚪ Static"
            
            self.df.iloc[start_idx:end_idx, col_idx_t] = trust_score
                
            segment_metrics.append({
                'Start Time': chunk.index.min(),
                'End Time': chunk.index.max(),
                'Local Gain': round(local_gain, 4),
                'Avg Delta P (psi)': round(avg_dp, 2) if pd.notna(avg_dp) else None,
                'Trust Score': trust_score
            })
            
        return pd.DataFrame(segment_metrics)

# ==========================================
# 2. DATA PRE-PROCESSORS & FORMATTERS
# ==========================================
@st.cache_data
def preprocess_field_data(raw_df):
    try:
        src_cols = ['Time (Hr)', 'Water Q (STB)', 'Pressure (psig)']
        src_df = raw_df[src_cols].dropna()
        src_agg = src_df.groupby('Time (Hr)').agg({'Water Q (STB)':'sum', 'Pressure (psig)':'mean'}).reset_index()
        src_agg.columns = ['Time', 'Source_Q', 'Source_P']
        
        snk_cols = ['Time (Hr).1', 'Water Q (STB).1', 'Pressure (psig).1']
        snk_df = raw_df[snk_cols].dropna()
        snk_agg = snk_df.groupby('Time (Hr).1').agg({'Water Q (STB).1':'sum', 'Pressure (psig).1':'mean'}).reset_index()
        snk_agg.columns = ['Time', 'Sink_Q', 'Sink_P']
        
        aligned_df = pd.merge(src_agg, snk_agg, on='Time', how='inner').set_index('Time')
        return aligned_df
    except Exception: return None

@st.cache_data
def generate_granular_node_export(raw_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    try:
        src_cols = ['Sources/Sinks', 'NAME', 'Time (Hr)', 'Water Q (STB)', 'Pressure (psig)']
        sources = raw_df[src_cols].dropna(subset=['NAME']).copy()
        sources.columns = ['Type', 'Node_Name', 'Time', 'Original_Q', 'Pressure']
        
        snk_cols = ['Sources/Sinks.1', 'NAME.1', 'Time (Hr).1', 'Water Q (STB).1', 'Pressure (psig).1']
        sinks = raw_df[snk_cols].dropna(subset=['NAME.1']).copy()
        sinks.columns = ['Type', 'Node_Name', 'Time', 'Original_Q', 'Pressure']
        
        sources['Reconciled_Q'], sources['Applied_Gain'] = sources['Original_Q'], 1.0
        sinks['Reconciled_Q'], sinks['Applied_Gain'] = sinks['Original_Q'], 1.0
        
        for _, row in metrics_df.iterrows():
            mask = (sources['Time'] >= row['Start Time']) & (sources['Time'] <= row['End Time'])
            sources.loc[mask, 'Applied_Gain'] = row['Local Gain']
            sources.loc[mask, 'Reconciled_Q'] = sources.loc[mask, 'Original_Q'] * row['Local Gain']
            
        return pd.concat([sources, sinks], ignore_index=True).sort_values(by=['Time', 'Type', 'Node_Name']).reset_index(drop=True)
    except Exception: return pd.DataFrame()

@st.cache_data
def prep_pipesim_batch_cases(reconciled_df: pd.DataFrame, num_cases: int = 50, exclude_fails: bool = True):
    df = reconciled_df.copy()
    if exclude_fails and 'Trust_Score' in df.columns:
        df = df[~df['Trust_Score'].str.contains("FAIL", na=False)]
    
    df['Flow_Volatility'] = df['Reconciled_Source_Q'].rolling(window=5).std()
    stable_df = df[df['Flow_Volatility'] <= df['Flow_Volatility'].quantile(0.75)].copy()
    stable_df = stable_df.sort_values(by='Reconciled_Source_Q')
    
    actual_num = min(len(stable_df), num_cases)
    indices = np.linspace(0, len(stable_df) - 1, actual_num, dtype=int)
    batch_cases = stable_df.iloc[indices].copy()
    
    # Map back to Time indices for the granular extraction
    return batch_cases.index.tolist(), batch_cases

# ==========================================
# 3. STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="Forensic Flow Workspace", layout="wide")
st.title("Forensic Flow Analyzer & Data Reconciler")

with st.sidebar:
    st.header("1. Data Input")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    source_q_col, sink_q_col = 'Source_Q', 'Sink_Q'
    source_p_col, sink_p_col = 'Source_P', 'Sink_P'
    has_pressure = True
    raw_df = None
    
    if not uploaded_file:
        np.random.seed(42)
        steps = 335
        base = np.random.normal(140000, 5000, steps)
        df = pd.DataFrame({
            'Source_Q': (base * 0.85) + np.random.normal(0, 2000, steps), 
            'Sink_Q': base + np.random.normal(0, 2000, steps), 
            'Source_P': np.random.normal(150, 10, steps), 
            'Sink_P': np.random.normal(100, 5, steps)
        })
        st.info("Using Demo Data. Upload your Wide CSV to unlock granular exports.")
    else:
        raw_df = pd.read_csv(uploaded_file)
        df = preprocess_field_data(raw_df)
        
        if df is not None:
            st.success("Wide CSV successfully parsed and aggregated!")
        else:
            st.error("Could not parse columns. Ensure your CSV has the standard output format.")
            st.stop()

    st.divider()
    st.header("⚙️ Calibration Engine")
    dt_size = st.number_input("Calibration Window (dt)", min_value=1, value=10, step=1)
    run_engine = st.button("🚀 Run Reconciliation Engine", type="primary", use_container_width=True)

# ==========================================
# 4. EXECUTION
# ==========================================
file_id = uploaded_file.name
if run_engine or 'processed_df' not in st.session_state or st.session_state.get('current_file') != file_id:
    reconciler = PipelineDataReconciler(df, 'Source_Q', 'Sink_Q', 'Source_P', 'Sink_P')
    reconciler.clean_and_stitch_sinks()
    lag = reconciler.align_time_lag()
    mdf = reconciler.segment_calibration(dt=dt_size)
    st.session_state.update({'processed_df': reconciler.df, 'metrics_df': mdf, 'lag': lag, 'current_file': file_id})

pdf, mdf = st.session_state['processed_df'], st.session_state['metrics_df']

# Tabs
t1, t2, t3, t4 = st.tabs(["Visual Diagnostics", "Segmented Regime Map", "Granular Node Export", "PIPESIM Batch Export"])

with t1:
    fig_rows = 3 if has_pressure else 2
    row_heights = [0.5, 0.25, 0.25] if has_pressure else [0.7, 0.3]
    titles = ("Flow Rate & Active Gain Multiplier", "Residual Error", "Delta P Validation Check") if has_pressure else ("Flow Rate & Active Gain Multiplier", "Residual Error")
    
    fig = make_subplots(rows=fig_rows, cols=1, shared_xaxes=True, row_heights=row_heights, 
                        subplot_titles=titles, specs=[[{"secondary_y": True}], [{"secondary_y": False}]] + ([[{'secondary_y': False}]] if has_pressure else []))

    fig.add_trace(go.Scatter(x=pdf.index, y=pdf[sink_q_col], mode='lines', name='Cleaned Sink Q', line=dict(color='#ff7f0e')), row=1, col=1)
    fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Reconciled_Source_Q'], mode='lines', name='Reconciled Source Q', line=dict(color='#1f77b4', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Active_Gain'], mode='lines', name='Segment Gain', line=dict(color='rgba(44, 160, 44, 0.6)', width=2, shape='hv')), row=1, col=1, secondary_y=True)

    diff = pdf['Reconciled_Source_Q'] - pdf[sink_q_col]
    fig.add_trace(go.Scatter(x=pdf.index, y=diff, mode='lines', name='Delta Q', line=dict(color='gray')), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)

    if has_pressure:
        delta_p = pdf[source_p_col] - pdf[sink_p_col]
        fig.add_trace(go.Scatter(x=pdf.index, y=delta_p, mode='lines', name='Measured Delta P', fill='tozeroy', fillcolor='rgba(148, 103, 189, 0.2)', line=dict(color='#9467bd')), row=3, col=1)

    fig.update_layout(height=800 if has_pressure else 600, hovermode="x unified", margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], subplot_titles=("Flow Rate & Gain", "Residuals", "Delta P Check"), specs=[[{"secondary_y": True}], [{}], [{}]])
    # fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Sink_Q'], name='Sink Q'), row=1, col=1)
    # fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Reconciled_Source_Q'], name='Reconciled Source Q', line=dict(dash='dot')), row=1, col=1)
    # fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Active_Gain'], name='Gain', line=dict(shape='hv')), row=1, col=1, secondary_y=True)
    # fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Reconciled_Source_Q']-pdf['Sink_Q'], name='Delta Q'), row=2, col=1)
    # fig.add_trace(go.Scatter(x=pdf.index, y=pdf['Source_P']-pdf['Sink_P'], name='Delta P'), row=3, col=1)
    # st.plotly_chart(fig, use_container_width=True)

with t2:
    st.dataframe(mdf, use_container_width=True)

with t3:
    granular_full = generate_granular_node_export(raw_df, mdf)
    st.dataframe(granular_full, use_container_width=True)
    st.download_button("Download Full Granular Node CSV", granular_full.to_csv(index=False), "full_granular.csv")

with t4:
    st.markdown("### PIPESIM Batch Cases (Nodes Only)")
    col_a, col_b = st.columns([1, 2])
    num_cases = col_a.number_input("Number of Cases", 10, 500, 50)
    exclude_f = col_b.checkbox("Exclude 'FAIL' Scores", value=True)
    
    if st.button("Generate Batch Node Export", type="primary"):
        target_times, batch_summary = prep_pipesim_batch_cases(pdf, num_cases, exclude_f)
        
        # Filter the granular data to ONLY include those 50 timestamps
        granular_full = generate_granular_node_export(raw_df, mdf)
        batch_nodes = granular_full[granular_full['Time'].isin(target_times)].copy()
        
        # Add a Case_ID mapping
        time_to_case = {t: f"Case_{str(i+1).zfill(3)}" for i, t in enumerate(target_times)}
        batch_nodes['Case_ID'] = batch_nodes['Time'].map(time_to_case)
        
        st.write(f"Generated {len(target_times)} Cases with {len(batch_nodes)} Node entries.")
        st.dataframe(batch_nodes[['Case_ID', 'Time', 'Type', 'Node_Name', 'Reconciled_Q', 'Pressure']], use_container_width=True)
        
        st.download_button(
            label="Download Batch Node CSV for PIPESIM",
            data=batch_nodes.to_csv(index=False).encode('utf-8'),
            file_name='pipesim_batch_node_data.csv',
            mime='text/csv'
        )
