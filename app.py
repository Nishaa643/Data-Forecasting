import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from backend.analyzer import analyze_dataset_and_recommend, run_transformer_forecast

# 1. Page Configuration
st.set_page_config(
    page_title="Data Forecasting - AI Intelligence Workbench",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Appearance Selector Header (Top Right)
top_col1, top_col2 = st.columns([4, 1])

with top_col2:
    theme_mode = st.segmented_control(
        "Theme",
        options=["Dark Mode", "Light Mode"],
        default="Dark Mode",
        key="app_theme_control"
    )

# 3. Dynamic Editorial Style Engine
if theme_mode == "Light Mode":
    bg_main = "#FAFAFC"
    card_bg = "#FFFFFF"
    text_primary = "#111827"
    text_muted = "#475569"
    border_color = "#CBD5E1"
    plotly_template = "plotly_white"
    chart_text = "#111827"
    line_hist = "#475569"
    line_forecast = "#BE123C"
    line_base = "#6D28D9"
    hero_bg = "linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 58%, #FFF1D6 100%)"
    hero_border = "rgba(180, 83, 9, 0.35)"
    hero_title = "#172033"
    hero_subtitle = "#334155"
    hero_badge_bg = "rgba(180, 83, 9, 0.08)"
    hero_badge_border = "rgba(180, 83, 9, 0.35)"
    hero_shadow = "0 12px 28px rgba(15, 23, 42, 0.10)"
    uploader_button_bg = "#F1F5F9"
    uploader_button_hover_bg = "#E2E8F0"
    uploader_button_text = "#0F172A"
else: # Dark Mode
    bg_main = "#07080C"
    card_bg = "#121520"
    text_primary = "#F8FAFC"
    text_muted = "#CBD5E1"
    border_color = "#39445A"
    plotly_template = "plotly_dark"
    chart_text = "#F8FAFC"
    line_hist = "#A8B4C7"
    line_forecast = "#38BDF8"
    line_base = "#FF6685"
    hero_bg = "linear-gradient(135deg, #181514 0%, #292321 50%, #3D2314 100%)"
    hero_border = "rgba(252, 211, 77, 0.4)"
    hero_title = "#FCD34D"
    hero_subtitle = "#FDE68A"
    hero_badge_bg = "rgba(255, 255, 255, 0.08)"
    hero_badge_border = "rgba(252, 211, 77, 0.35)"
    hero_shadow = "0 15px 35px rgba(0, 0, 0, 0.3)"
    uploader_button_bg = "#273247"
    uploader_button_hover_bg = "#334155"
    uploader_button_text = "#F8FAFC"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@700;900&family=Cormorant+Garamond:ital,wght@0,600;0,700;1,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Global Overrides */
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_main} !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }}
    
    [data-testid="stHeader"] {{ background: transparent !important; }}

    /* Improve contrast for native controls in both themes without changing their layout. */
    [data-testid="stAppViewContainer"] input,
    [data-testid="stAppViewContainer"] textarea,
    [data-testid="stAppViewContainer"] [data-baseweb="select"] > div,
    [data-testid="stAppViewContainer"] [data-baseweb="input"] > div,
    [data-testid="stAppViewContainer"] [data-testid="stFileUploaderDropzone"] {{
        color: {text_primary} !important;
        background-color: {card_bg} !important;
        border-color: {border_color} !important;
    }}
    [data-testid="stAppViewContainer"] input::placeholder,
    [data-testid="stAppViewContainer"] textarea::placeholder {{
        color: {text_muted} !important;
        opacity: 1 !important;
    }}
    [data-testid="stAppViewContainer"] [data-baseweb="select"] svg,
    [data-testid="stAppViewContainer"] [data-baseweb="input"] svg {{
        fill: {text_primary} !important;
        color: {text_primary} !important;
    }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        border-color: {border_color} !important;
    }}

    /* Universal Text Color Lock (Prevents fading on hover/cursor movement) */
    [data-testid="stAppViewContainer"] p, 
    [data-testid="stAppViewContainer"] span:not(.brand-badge):not(.card-value), 
    [data-testid="stAppViewContainer"] label, 
    [data-testid="stTabs"] button,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {{
        color: {text_primary} !important;
        fill: {text_primary} !important;
        opacity: 1 !important;
    }}

    /* Lock Expander and Container States Across Hover/Focus */
    [data-testid="stExpander"], 
    [data-testid="stExpander"] details, 
    [data-testid="stExpander"] summary {{
        background-color: {card_bg} !important;
        border-color: {border_color} !important;
    }}

    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:focus {{
        background-color: {card_bg} !important;
        color: {text_primary} !important;
    }}

    [data-testid="stExpander"] summary:hover span,
    [data-testid="stExpander"] summary:hover p {{
        color: {text_primary} !important;
    }}

    /* Modern App Metric Cards */
    .app-card {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 18px;
        padding: 22px 26px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.04);
    }}
    .card-label {{
        color: {text_muted};
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }}
    .card-value {{
        color: {text_primary};
        font-size: 2.1rem;
        font-weight: 800;
        font-family: 'Cinzel Decorative', serif;
        margin-top: 4px;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {card_bg} !important;
        border-right: 1px solid {border_color} !important;
    }}

    /* Action Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, #D97706 0%, #B45309 100%) !important;
        color: #FFFFFF !important;
        border-radius: 14px !important;
        border: none !important;
        font-weight: 700 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1rem !important;
        padding: 0.75rem 1.8rem !important;
        box-shadow: 0 8px 25px rgba(217, 119, 6, 0.35) !important;
        transition: all 0.25s ease !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px rgba(217, 119, 6, 0.5) !important;
    }}

    /* File Uploader Container */
    [data-testid="stFileUploader"] {{
        background-color: {bg_main} !important;
        border: 2px dashed {border_color} !important;
        border-radius: 16px !important;
        padding: 16px !important;
    }}

    /* Keep the native Upload button readable in default, hover, and focus states. */
    [data-testid="stFileUploader"] section button,
    [data-testid="stFileUploader"] section button p,
    [data-testid="stFileUploader"] section button span,
    [data-testid="stFileUploader"] section button svg {{
        color: {uploader_button_text} !important;
        fill: {uploader_button_text} !important;
        opacity: 1 !important;
    }}
    [data-testid="stFileUploader"] section button {{
        background-color: {uploader_button_bg} !important;
        border: 1px solid {border_color} !important;
    }}
    [data-testid="stFileUploader"] section button:hover,
    [data-testid="stFileUploader"] section button:focus,
    [data-testid="stFileUploader"] section button:focus-visible,
    [data-testid="stFileUploader"] section button:active {{
        background-color: {uploader_button_hover_bg} !important;
        color: {uploader_button_text} !important;
        border-color: {border_color} !important;
        box-shadow: 0 0 0 2px {border_color} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# 4. Compact App Hero Header
st.markdown(f"""
    <div style="background: {hero_bg}; border: 1px solid {hero_border}; border-radius: 18px; padding: 18px 28px; margin-bottom: 24px; box-shadow: {hero_shadow}; display: flex; align-items: center; gap: 20px; width: 100%; position: relative; overflow: hidden;">
        <svg width="42" height="42" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink: 0;">
            <path d="M3 20H21" stroke="{hero_title}" stroke-width="2.8" stroke-linecap="round"/>
            <path d="M3 14L8 9L12 13L18 7L21 10" stroke="#38BDF8" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="18" cy="7" r="3" fill="{hero_title}"/>
            <circle cx="8" cy="9" r="3" fill="#38BDF8"/>
        </svg>
        <div style="flex-grow: 1; overflow: hidden;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h1 style="font-family: 'Cinzel Decorative', serif !important; font-size: 2.1rem !important; font-weight: 900 !important; color: {hero_title} !important; margin: 0 !important; line-height: 1.1 !important; white-space: nowrap !important; letter-spacing: 1.5px; text-shadow: 0 2px 10px rgba(0,0,0,0.16);">Data Forecasting</h1>
                    <div style="font-family: 'Cormorant Garamond', serif !important; font-size: 1rem !important; font-style: italic !important; font-weight: 700 !important; color: {hero_subtitle} !important; letter-spacing: 1px; margin-top: 2px;">Universal Time-Series & Deep Learning Prediction Platform</div>
                </div>
                <span style="background: {hero_badge_bg}; color: {hero_title}; font-size: 0.68rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px; padding: 4px 10px; border-radius: 100px; border: 1px solid {hero_badge_border}; backdrop-filter: blur(8px);">STUDIO PRO</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# 5. Sidebar Navigation & Session State Dataset Persistence
st.sidebar.markdown(f"<h3 style='color:{text_primary}; font-family: Cinzel Decorative; font-size: 1.1rem;'>1. Data Input</h3>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"], key="dataset_uploader")

if uploaded_file is not None:
    temp_df = pd.read_csv(uploaded_file)
    
    for col in temp_df.columns:
        if temp_df[col].dtype == 'object':
            try:
                cleaned_col = temp_df[col].astype(str).str.replace(r'[\$,]', '', regex=True)
                converted = pd.to_numeric(cleaned_col, errors='coerce')
                if converted.notnull().sum() > 0.5 * len(temp_df):
                    temp_df[col] = converted
            except Exception:
                pass
                
    st.session_state['df_original'] = temp_df.copy()

if 'df_original' in st.session_state:
    original_df = st.session_state['df_original']
    df = original_df.copy()

    st.sidebar.markdown(
        f"<h3 style='color:{text_primary}; font-family: Cinzel Decorative; font-size: 1.1rem;'>2. Row Filtering</h3>",
        unsafe_allow_html=True,
    )
    filter_column = st.sidebar.selectbox(
        "Filter column",
        options=["None"] + original_df.columns.tolist(),
        index=0,
        key="row_filter_column",
        help="Choose a column, such as Country, to filter its rows.",
    )

    if filter_column != "None":
        filter_values = original_df[filter_column].dropna().unique().tolist()
        selected_values = st.sidebar.multiselect(
            f"Values in {filter_column}",
            options=filter_values,
            default=[],
            key="row_filter_values",
            help="Select one or more values. Leave empty to show all rows.",
        )
        if selected_values:
            df = original_df[original_df[filter_column].isin(selected_values)].copy()

    st.sidebar.caption(f"Showing {len(df):,} of {len(original_df):,} rows")

    with st.expander("🔍 Dataset Explorer", expanded=False):
        st.dataframe(df.head(5) if len(df) < 5 else df.sample(5), use_container_width=True)

    col_names = df.columns.tolist()
    
    st.sidebar.markdown(f"<h3 style='color:{text_primary}; font-family: Cinzel Decorative; font-size: 1.1rem;'>3. Hyperparameters</h3>", unsafe_allow_html=True)
    time_col = st.sidebar.selectbox("Time/Date Axis (Optional)", [None] + col_names)
    
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    if not numeric_cols:
        st.error("No numerical columns found in uploaded dataset.")
        target_col = None
    else:
        target_col = st.sidebar.selectbox("Target Variable", numeric_cols)
    
    model_type = st.sidebar.selectbox(
        "Transformer Architecture",
        ["Informer", "PatchTST", "FEDformer", "ETSformer", "Crossformer"]
    )
    
    lookback_len = st.sidebar.slider("Lookback Window", 12, 168, 24)
    pred_len = st.sidebar.slider("Prediction Horizon", 6, 72, 12)

    if target_col and st.sidebar.button("Run Intelligence Engine", type="primary"):
        with st.spinner("Processing deep learning architecture pipeline..."):
            
            history_unscaled, forecast_unscaled, naive_forecast = run_transformer_forecast(
                df, target_col, model_type, lookback_len, pred_len
            )

            try:
                analysis = analyze_dataset_and_recommend(df, target_col)
            except Exception:
                analysis = {}

            if isinstance(analysis, tuple):
                analysis = analysis[0]

            if isinstance(analysis, dict) and "dataset_summary" in analysis:
                total_samples = analysis["dataset_summary"].get("total_samples", len(df))
                variability_score = analysis["dataset_summary"].get("variability_score", 0.0)
                recommendations = analysis.get("recommendations", [])
            else:
                series = pd.to_numeric(df[target_col], errors='coerce').dropna()
                total_samples = len(series)
                mean_val = series.mean()
                std_val = series.std()
                variability_score = round(float(std_val / mean_val), 3) if mean_val != 0 else 0.0
                recommendations = []

        # 6. Custom Metric Cards (Responsive font sizing clamp applied here)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
                <div class="app-card">
                    <div class="card-label">Active Architecture</div>
                    <div class="card-value" style="color: #D97706; font-size: clamp(1rem, 2vw, 1.7rem); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{model_type}</div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="app-card">
                    <div class="card-label">Evaluated Records</div>
                    <div class="card-value">{total_samples:,}</div>
                </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
                <div class="app-card">
                    <div class="card-label">Variability Coefficient</div>
                    <div class="card-value">{variability_score}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 7. Styled Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Forecast Horizon", "📉 Benchmark vs Naive", "💡 Neural Recommendations"])

        with tab1:
            fig = go.Figure()
            x_hist = list(range(0, len(history_unscaled)))
            fig.add_trace(go.Scatter(x=x_hist, y=history_unscaled, mode='lines', name='Historical Input', line=dict(color=line_hist, width=2.5)))
            
            x_pred = list(range(len(history_unscaled), len(history_unscaled) + len(forecast_unscaled)))
            fig.add_trace(go.Scatter(x=x_pred, y=forecast_unscaled, mode='lines+markers', name=f'{model_type} Model', line=dict(color=line_forecast, width=3.5)))

            fig.update_layout(
                xaxis_title="Time Steps",
                yaxis_title=target_col,
                template=plotly_template,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=chart_text, size=14),
                legend=dict(font=dict(color=chart_text, size=13)),
                xaxis=dict(
                    tickfont=dict(color=chart_text, size=13),
                    title_font=dict(color=chart_text, size=14),
                    gridcolor=border_color,
                    zerolinecolor=border_color,
                    linecolor=border_color,
                ),
                yaxis=dict(
                    tickfont=dict(color=chart_text, size=13),
                    title_font=dict(color=chart_text, size=14),
                    gridcolor=border_color,
                    zerolinecolor=border_color,
                    linecolor=border_color,
                ),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            comp_fig = go.Figure()
            comp_fig.add_trace(go.Scatter(x=x_pred, y=forecast_unscaled, mode='lines+markers', name=f'Active Model ({model_type})', line=dict(color=line_forecast, width=3.5)))
            comp_fig.add_trace(go.Scatter(x=x_pred, y=naive_forecast, mode='lines', name='Persistence Baseline', line=dict(dash='dash', color=line_base, width=2.5)))
            
            comp_fig.update_layout(
                xaxis_title="Future Time Steps",
                yaxis_title=target_col,
                template=plotly_template,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=chart_text, size=14),
                legend=dict(font=dict(color=chart_text, size=13)),
                xaxis=dict(
                    tickfont=dict(color=chart_text, size=13),
                    title_font=dict(color=chart_text, size=14),
                    gridcolor=border_color,
                    zerolinecolor=border_color,
                    linecolor=border_color,
                ),
                yaxis=dict(
                    tickfont=dict(color=chart_text, size=13),
                    title_font=dict(color=chart_text, size=14),
                    gridcolor=border_color,
                    zerolinecolor=border_color,
                    linecolor=border_color,
                ),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(comp_fig, use_container_width=True)

        with tab3:
            for rec in recommendations:
                with st.expander(f"✨ **{rec.get('model', 'Transformer Model')}**"):
                    st.write(rec.get("reason", "No description provided."))

else:
    st.info("👈 Upload any CSV time-series dataset from the sidebar to initialize the Data Forecasting workspace.")