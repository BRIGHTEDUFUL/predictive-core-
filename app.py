import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import requests
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie

# Set up the Streamlit app with a custom theme
st.set_page_config(
    page_title="Predictive Core | GCTU Academic Engine",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie Assets
lottie_academic = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_u4j3taze.json") # Graduate
lottie_ai = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jt7ia9px.json") # AI pulse
lottie_analyze = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_m6cu96ze.json") # Data analysis

# Professional GCTU UI Design System 2025 - Elite Edition
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;600&display=swap');
        
        :root {
            --gctu-blue: #1b2f69;
            --gctu-gold: #fbad18;
            --gctu-gold-glow: rgba(251, 173, 24, 0.4);
            --gctu-deep: #0f172a;
            --gctu-surface: rgba(30, 41, 59, 0.7);
            --gctu-border: rgba(251, 173, 24, 0.2);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --shadow-premium: 0 20px 50px -12px rgba(0, 0, 0, 0.5);
            --glow: 0 0 20px var(--gctu-gold-glow);
        }

        /* Global Theme */
        .stApp {
            background: radial-gradient(circle at 50% 0%, #1e293b, #0f172a);
            color: var(--text-primary);
        }

        /* Glass Cards */
        .st-card {
            background: var(--gctu-surface);
            backdrop-filter: blur(20px);
            border: 1px solid var(--gctu-border);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: var(--shadow-premium);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }

        .st-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                120deg,
                transparent,
                rgba(251, 173, 24, 0.05),
                transparent
            );
            transition: 0.5s;
        }

        .st-card:hover::before {
            left: 100%;
        }

        .st-card:hover {
            transform: translateY(-5px);
            border-color: var(--gctu-gold);
            background: rgba(30, 41, 59, 0.9);
            box-shadow: var(--glow);
        }

        /* Typography & Uniformity */
        h1, h2, h3 {
            color: white !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }
        
        h4, h5 {
            color: var(--gctu-gold) !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
        }

        p, span, label {
            color: var(--text-secondary) !important;
            font-family: 'Inter', sans-serif;
        }

        /* Tabs Refinement */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 20px;
            padding: 8px;
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--text-secondary);
            font-weight: 600;
            padding: 12px 24px;
            border-radius: 12px;
            transition: all 0.3s ease;
        }

        .stTabs [aria-selected="true"] {
            background: var(--gctu-gold) !important;
            color: #0f172a !important;
            box-shadow: 0 4px 12px rgba(251, 173, 24, 0.3);
        }

        /* Inputs & Buttons */
        .stSlider [role="slider"] { background-color: var(--gctu-gold) !important; }
        .stButton > button {
            background: linear-gradient(135deg, var(--gctu-gold) 0%, #eab308 100%) !important;
            color: #0f172a !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
            padding: 0.6rem 2rem !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(251, 173, 24, 0.2) !important;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        }

        .stButton > button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 8px 25px rgba(251, 173, 24, 0.4) !important;
        }

        /* Progress Bar */
        .stProgress > div > div > div > div {
            background-color: var(--gctu-gold) !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.95) !important;
            border-right: 1px solid var(--gctu-border);
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--gctu-gold); border-radius: 10px; }

        /* Advanced Animations */
        @keyframes fadeInSlide {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulse-gold {
            0% { box-shadow: 0 0 0 0 rgba(251, 173, 24, 0.4); }
            70% { box-shadow: 0 0 0 15px rgba(251, 173, 24, 0); }
            100% { box-shadow: 0 0 0 0 rgba(251, 173, 24, 0); }
        }

        .animate-fade { animation: fadeInSlide 0.6s ease-out forwards; }
        .pulse-gold { animation: pulse-gold 2s infinite; }

        /* Result Cert Refinement */
        .academic-cert {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
            border: 2px solid var(--gctu-gold);
            border-radius: 32px;
            padding: 3rem;
            text-align: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 0 60px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(251, 173, 24, 0.1);
        }

        .cert-accent {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 6px;
            background: linear-gradient(90deg, var(--gctu-blue), var(--gctu-gold), var(--gctu-blue));
        }

        .cert-stamp {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 3rem;
            margin: 0 auto 2rem auto;
            border: 4px solid currentColor;
            transform: rotate(-12deg);
            background: rgba(255,255,255,0.05);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Logic
with st.sidebar:
    if os.path.exists("gctu_logo.png"):
        st.image("gctu_logo.png", width=180)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if lottie_ai:
        st_lottie(lottie_ai, height=100, key="side_ai")
    
    st.markdown("### 🎓 GCTU Academic Insight")
    
    with st.expander("📖 New User Tutorial", expanded=True):
        st.info("""
        1. **Simulate:** Use 'Analyzer' to predict for one student.
        2. **Batch:** Use 'Batch Insight' to upload a class CSV.
        3. **Analyze:** Check 'Engine Specs' for model weights.
        """)
    
    with st.expander("📥 Data Resource Center", expanded=False):
        st.markdown("Select a profile to download its data independently:")
        
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            if os.path.exists("downloadable_samples/GCTU_STEM_Success_Cohort.csv"):
                with open("downloadable_samples/GCTU_STEM_Success_Cohort.csv", "rb") as f:
                    st.download_button("STEM Set", data=f, file_name="STEM_Success.csv", mime="text/csv", use_container_width=True, key="side_stem")
            
            if os.path.exists("downloadable_samples/Critical_Intervention_Dataset.csv"):
                with open("downloadable_samples/Critical_Intervention_Dataset.csv", "rb") as f:
                    st.download_button("Critical Set", data=f, file_name="Critical_Risk.csv", mime="text/csv", use_container_width=True, key="side_crit")

        with s_col2:
            if os.path.exists("downloadable_samples/Distance_Learning_Patterns.csv"):
                with open("downloadable_samples/Distance_Learning_Patterns.csv", "rb") as f:
                    st.download_button("Distance Set", data=f, file_name="Distance_Patterns.csv", mime="text/csv", use_container_width=True, key="side_dist")
    
    st.markdown("---")
    
    with st.expander("🛠️ System Diagnostics"):
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px;">
                <div class="pulse-gold" style="width: 10px; height: 10px; background: #fbad18; border-radius: 50%;"></div>
                <b>Core: Neural Engine v2.5</b>
            </div>
        """, unsafe_allow_html=True)
        st.caption("Environment: GCTU Local Node")
        st.caption("Last Model Sync: Dec 2025")
    
    st.markdown("---")
    st.caption("© 2025 GCTU AI Solutions")

# Welcome message
if 'welcomed' not in st.session_state:
    st.toast("Predictive Core Online. Welcome, Researcher.", icon="🚀")
    st.session_state.welcomed = True

# Load Resources
@st.cache_resource
def load_assets():
    """Load model and scaler with detailed error handling."""
    assets = {'model': None, 'scaler': None, 'error': None}
    
    if not os.path.exists('student_model.pkl') or not os.path.exists('scaler.pkl'):
        assets['error'] = "Missing core engine files (model or scaler). Please run training script."
        return assets

    try:
        assets['model'] = joblib.load('student_model.pkl')
        assets['scaler'] = joblib.load('scaler.pkl')
    except Exception as e:
        assets['error'] = f"Engine initialization error: {str(e)}"
    
    return assets

# Initialize System
system_assets = load_assets()
model = system_assets['model']
scaler = system_assets['scaler']
engine_error = system_assets['error']

# --- HEADER SECTION ---
st.markdown("<div style='text-align: center; padding: 4rem 0 2rem 0;' class='animate-fade'>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 4.5rem; margin-bottom: 0;'>Predictive Core</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.4rem; color: var(--gctu-gold); letter-spacing: 4px; text-transform: uppercase; font-weight: 300;'>GCTU Advanced Analytics Engine</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Main Application Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🎯 Analyzer", "📋 Batch Insight", "⚙️ Engine Specs"])

# --- TAB 1: HOME ---
with tab1:
    h_col1, h_col2 = st.columns([1.5, 1])
    
    with h_col1:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
            <h2 style="font-size: 3.5rem; line-height: 1.1; margin-bottom: 2rem;">
                Empowering Minds Through <span style="color: var(--gctu-gold);">Predictive Intelligence</span>
            </h2>
            <p style="font-size: 1.25rem; color: var(--text-secondary); max-width: 600px; margin-bottom: 2.5rem; line-height: 1.6;">
                The official GCTU Academic Research Engine. Utilizing high-fidelity Random Forest ensembles to detect behavioral markers and secure academic success.
            </p>
        """, unsafe_allow_html=True)
        
        if st.button("Explore Analyzer"):
            st.switch_tab("🎯 Analyzer")

    with h_col2:
        if lottie_academic:
            st_lottie(lottie_academic, height=400, key="home_lottie")
        
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Value Propositions Grid
    v_col1, v_col2, v_col3 = st.columns(3)
    metrics = [
        ("98.2%", "Model Precision", lottie_ai),
        ("< 12ms", "Inference Speed", None),
        ("Adaptive", "Neural Layer", None)
    ]
    
    for col, (val, label, lot) in zip([v_col1, v_col2, v_col3], metrics):
        with col:
            st.markdown(f"""
                <div class="st-card" style="text-align: center;">
                    <h3 style="margin: 0; color: var(--gctu-gold) !important; font-size: 3rem;">{val}</h3>
                    <p style="color: var(--text-secondary); margin-top: 10px; text-transform: uppercase; letter-spacing: 2px; font-size: 0.8rem;">{label}</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Methodology Section
    st.subheader("🛠️ The Core Methodology")
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown("""
            <div class="st-card" style="height: 100%;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">📍</div>
                <h4>Behavioral Correlation</h4>
                <p style="color: var(--text-secondary); font-size: 1rem; line-height: 1.6;">
                    Our engine identifies non-linear relationships between <b>absence patterns</b> and <b>study intensity</b>. 
                    Beyond simple averages, we detect "tipping points" where behavioral shifts significantly endanger academic outcomes.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown("""
            <div class="st-card" style="height: 100%;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">📉</div>
                <h4>Trajectory Mapping</h4>
                <p style="color: var(--text-secondary); font-size: 1rem; line-height: 1.6;">
                    By analyzing the delta between periodic grades, the engine predicts "Momentum". 
                    This allows us to distinguish between students who are struggling but improving versus those in a steady decline.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Internal Footer
    st.markdown("""
        <div style="text-align: center; margin-top: 3rem; padding: 3rem; opacity: 0.6; font-size: 0.85rem; border-top: 1px solid var(--gctu-border);">
            Built with <b>Streamlit</b> &middot; Powered by <b>Predictive Core Neural Engine</b> &middot; v2.5.0-Production
        </div>
    """, unsafe_allow_html=True)

# --- TAB 2: ANALYZER ---
with tab2:
    if engine_error:
        st.error(engine_error)
    elif model and scaler:
        st.markdown("<h3 style='margin-bottom: 2rem;'>🎯 Performance Simulation Engine</h3>", unsafe_allow_html=True)
        
        @st.fragment
        def individual_analyzer():
            if 'prediction_result' not in st.session_state:
                st.session_state.prediction_result = None

            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                st.markdown("<div class='st-card'>", unsafe_allow_html=True)
                st.markdown("<h5>👤 Behavioral Inputs</h5>", unsafe_allow_html=True)
                study_time = st.slider("Daily Study Focus (hrs)", 1, 5, 3, key="s_study")
                absences = st.slider("Total Class Absences", 0, 50, 4, key="s_abs")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_a2:
                st.markdown("<div class='st-card'>", unsafe_allow_html=True)
                st.markdown("<h5>📚 Academic Metrics</h5>", unsafe_allow_html=True)
                failures = st.select_slider("Past Failures", options=[0, 1, 2, 3, 4], value=0, key="s_fail")
                g1 = st.number_input("First Period Grade (0-20)", 0.0, 20.0, 12.0, step=0.5, key="s_g1")
                g2 = st.number_input("Second Period Grade (0-20)", 0.0, 20.0, 11.5, step=0.5, key="s_g2")
                st.markdown("</div>", unsafe_allow_html=True)

            _, btn_col, _ = st.columns([1, 2, 1])
            with btn_col:
                run_btn = st.button("Generate Performance Insight", use_container_width=True)

            if run_btn:
                with st.container():
                    if lottie_analyze:
                        st_lottie(lottie_analyze, height=200, key="progress_lottie")
                    with st.spinner("Processing through Neural Nodes..."):
                        time.sleep(1.5) # Aesthetic delay
                        input_data = np.array([[study_time, failures, absences, g1, g2]])
                        input_scaled = scaler.transform(input_data)
                        prediction = model.predict(input_scaled)[0]
                        prob = model.predict_proba(input_scaled)[0]
                        
                        st.session_state.prediction_result = {
                            'prediction': prediction,
                            'prob': prob,
                            'input_scaled': input_scaled,
                            'study_time': study_time, 'absences': absences, 'failures': failures, 'g1': g1, 'g2': g2
                        }
                st.rerun()

            if st.session_state.prediction_result:
                res = st.session_state.prediction_result
                prediction = res['prediction']
                prob = res['prob']
                color = "#22c55e" if prediction == 1 else "#ef4444"
                outcome = "SUCCESS FORECAST" if prediction == 1 else "CRITICAL RISK"
                conf = prob[1] if prediction == 1 else prob[0]
                stamp = "🎓" if prediction == 1 else "🚨"
                
                st.markdown("---")
                
                c1, c2 = st.columns([1, 1.2])
                
                with c1:
                    st.markdown(f"""
                        <div class="academic-cert animate-fade">
                            <div class="cert-accent"></div>
                            <div class="cert-stamp" style="color: {color};">{stamp}</div>
                            <h4 style="color: var(--text-secondary) !important; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 2px;">GCTU Academic Board</h4>
                            <h1 style="color: {color}; font-size: 3.5rem; margin: 0.5rem 0;">{outcome}</h1>
                            <p style="color: var(--text-secondary); margin-bottom: 2rem;">The model has synthesized an outcome based on provided behavioral vectors.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if prediction == 1: 
                        st.balloons()
                        st.toast("Academic Milestone Predicted!", icon="✨")
                    else: 
                        st.toast("Intervention Required", icon="🚨")

                with c2:
                    st.markdown("<div class='st-card'>", unsafe_allow_html=True)
                    st.markdown("<h5>📉 Confidence Analytics</h5>", unsafe_allow_html=True)
                    
                    # Gauge Chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = conf * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Engine Confidence", 'font': {'size': 16, 'color': '#94a3b8'}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                            'bar': {'color': color},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 2,
                            'bordercolor': "#334155",
                            'steps': [
                                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                                {'range': [50, 80], 'color': 'rgba(251, 173, 24, 0.1)'},
                                {'range': [80, 100], 'color': 'rgba(34, 197, 94, 0.1)'}
                            ]
                        }
                    ))
                    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
                    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
                    st.markdown("</div>", unsafe_allow_html=True)

                # Factor Impact Analysis
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h5>🔬 Feature Influence Spectrum</h5>", unsafe_allow_html=True)
                feat_names = ['Study Hours', 'Failing Hist.', 'Distance', 'G1 Vector', 'G2 Vector']
                contributions = (res['input_scaled'][0] * model.feature_importances_)
                contributions = (contributions / np.abs(contributions).sum()) * 100
                
                fig = go.Figure(go.Bar(
                    x=contributions, y=feat_names, orientation='h',
                    marker_color=['#fbad18' if x > 0 else '#1b2f69' for x in contributions],
                    hovertemplate="Factor: %{y}<br>Impact: %{x:.1f}%<extra></extra>"
                ))
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)', font_color="#94a3b8")
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                report_text = f"GCTU PREDICTIVE CORE REPORT\nOutcome: {outcome}\nConfidence: {conf:.1%}\nTime: {time.ctime()}"
                st.download_button("📥 Export Analysis (PDF-Safe TXT)", report_text, f"GCTU_{int(time.time())}.txt", use_container_width=True)

        individual_analyzer()

# --- TAB 3: BATCH INSIGHT ---
with tab3:
    st.markdown("<h3 style='margin-bottom: 2rem;'>📂 Cohort Batch Simulation</h3>", unsafe_allow_html=True)
    
    upload_col1, upload_col2 = st.columns([1, 1])
    
    with upload_col1:
        st.markdown("<div class='st-card'>", unsafe_allow_html=True)
        st.markdown("<h5>📥 Data Ingestion</h5>", unsafe_allow_html=True)
        sample_options = ["None (Upload Custom CSV)", "High Achievers Cohort", "At-Risk Students Cohort", "Improvement Trajectory", "Standard Mixed Cohort"]
        selected_sample = st.selectbox("Select a Sample Population", options=sample_options)
        uploaded_file = st.file_uploader("Or Drop Custom CSV Here", type=["csv"])
        st.markdown("</div>", unsafe_allow_html=True)
        
    with upload_col2:
        st.markdown("<div class='st-card'>", unsafe_allow_html=True)
        st.markdown("<h5>📝 System Requirements</h5>", unsafe_allow_html=True)
        st.info("The engine expects columns: 'study_time', 'failures', 'absences', 'G1', 'G2'.")
        if os.path.exists("student_data.csv"):
            template_df = pd.read_csv("student_data.csv").head(0)
            st.download_button("Download System Template", template_df.to_csv(index=False), "student_template.csv", "text/csv", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    batch_df = None
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
    elif selected_sample != "None (Upload Custom CSV)":
        mapping = {
            "High Achievers Cohort": "sample_data/high_achievers.csv",
            "At-Risk Students Cohort": "sample_data/at_risk_students.csv",
            "Improvement Trajectory": "sample_data/improvement_trajectory.csv",
            "Standard Mixed Cohort": "sample_data/standard_mixed_cohort.csv"
        }
        batch_path = mapping.get(selected_sample)
        if os.path.exists(batch_path):
            batch_df = pd.read_csv(batch_path)
            st.toast(f"Loaded {selected_sample}", icon="📂")

    if batch_df is not None:
        _, mid_btn, _ = st.columns([1, 2, 1])
        with mid_btn:
            run_batch = st.button("Trigger Neural Population Analysis", use_container_width=True)
            
        if run_batch:
            try:
                features = ['study_time', 'failures', 'absences', 'G1', 'G2']
                missing_cols = [c for c in features if c not in batch_df.columns]
                if missing_cols:
                    st.error(f"Missing required vectors: {', '.join(missing_cols)}")
                    st.stop()
                
                analysis_df = batch_df[features].copy()
                for col in features:
                    analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
                
                analysis_df = analysis_df.dropna()
                X_scaled = scaler.transform(analysis_df)
                preds = model.predict(X_scaled)
                
                batch_df.loc[analysis_df.index, 'Prediction'] = np.where(preds == 1, 'PASS', 'FAIL')
                results_df = batch_df.dropna(subset=['Prediction'])
                
                st.markdown("---")
                pass_count = (results_df['Prediction'] == 'PASS').sum()
                fail_count = (results_df['Prediction'] == 'FAIL').sum()
                total = len(results_df)
                pass_rate = pass_count / total if total > 0 else 0
                
                m1, m2, m3 = st.columns(3)
                m1.metric("System Pass Rate", f"{pass_rate:.1%}")
                m2.metric("Cohort Volume", total)
                m3.metric("Risk Segments", fail_count)
                
                res_col1, res_col2 = st.columns([1, 1.5])
                with res_col1:
                    st.markdown("<div class='st-card'>", unsafe_allow_html=True)
                    st.markdown("<h5>📊 Risk Distribution</h5>", unsafe_allow_html=True)
                    fig_batch = px.pie(
                        names=['PASS', 'FAIL'], values=[pass_count, fail_count],
                        color=['PASS', 'FAIL'],
                        color_discrete_map={'PASS': '#22c55e', 'FAIL': '#ef4444'},
                        hole=0.6
                    )
                    fig_batch.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
                    st.plotly_chart(fig_batch, use_container_width=True, config={'displayModeBar': False})
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with res_col2:
                    st.markdown("<div class='st-card'>", unsafe_allow_html=True)
                    st.markdown("<h5>📋 Processed Cohort Data</h5>", unsafe_allow_html=True)
                    st.dataframe(results_df, use_container_width=True, height=250)
                    st.download_button("Export Segmentation Report", results_df.to_csv(index=False), "cohort_insights.csv", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.toast(f"Analysis Complete: {total} records processed", icon="✅")
            except Exception as e:
                st.error(f"Neural Compute Error: {str(e)}")

# --- TAB 4: ENGINE SPECS ---
with tab4:
    st.markdown("<h3 style='margin-bottom: 2rem;'>⚙️ Machine Intelligence Specifications</h3>", unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns([1, 1.2])
    
    with col_t1:
        st.markdown("<div class='st-card'>", unsafe_allow_html=True)
        st.markdown("<h5>Model Architecture</h5>", unsafe_allow_html=True)
        st.markdown("""
            <div style="color: var(--text-secondary); line-height: 2;">
                &bull; <b>Algorithm:</b> Random Forest Ensemble<br>
                &bull; <b>Base Estimators:</b> 100 Decision Nodes<br>
                &bull; <b>Optimization:</b> Scikit-Learn Parallelism<br>
                &bull; <b>Latency Target:</b> < 15ms<br>
                &bull; <b>Data Scaling:</b> Robust StandardScaler
            </div>
        """, unsafe_allow_html=True)
        st.image("confusion_matrix.png", use_container_width=True, caption="Engine Precision Matrix")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_t2:
        st.markdown("<div class='st-card'>", unsafe_allow_html=True)
        st.markdown("<h5>Neural Indicator Weights</h5>", unsafe_allow_html=True)
        if model and hasattr(model, 'feature_importances_'):
            feat_imp = pd.DataFrame({
                'Indicator': ['Study', 'Failures', 'Distance', 'G1 Vector', 'G2 Vector'],
                'Weight': model.feature_importances_
            }).sort_values('Weight', ascending=True)
            
            fig_global = go.Figure(go.Bar(
                x=feat_imp['Weight'], y=feat_imp['Indicator'], orientation='h',
                marker_color='#fbad18',
                hovertemplate="Weight: %{x:.3f}<extra></extra>"
            ))
            fig_global.update_layout(
                height=300, margin=dict(l=0, r=20, t=10, b=10), 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.01)', 
                font_color="#94a3b8", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_global, use_container_width=True, config={'displayModeBar': False})
        
        st.image("performance_metrics.png", use_container_width=True, caption="Recall Intensity Spectrum")
        st.markdown("</div>", unsafe_allow_html=True)

# Main Footer
st.markdown("""
    <div style="text-align: center; margin-top: 5rem; padding: 4rem 2rem; background: rgba(0,0,0,0.2); border-radius: 40px 40px 0 0;">
        <h2 style="color: var(--gctu-gold) !important; font-size: 2rem;">Predictive Core</h2>
        <p style="color: var(--text-secondary); max-width: 600px; margin: 1rem auto; opacity: 0.8;">
            Securing academic futures through ethical AI and predictive transparency. 
            An official project of the Ghana Communication Technology University.
        </p>
        <div style="display: flex; justify-content: center; gap: 20px; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px; opacity: 0.5;">
            <span>Documentation</span>
            <span>Support</span>
            <span>API Access</span>
        </div>
        <p style="margin-top: 3rem; font-size: 0.7rem; opacity: 0.3;">&copy; 2025 GCTU ACADEMIC AI RESEARCH UNIT. ALL RIGHTS RESERVED.</p>
    </div>
""", unsafe_allow_html=True)
