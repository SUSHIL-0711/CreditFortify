import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="CreditFortify",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #2563eb;
        --secondary-color: #1e40af;
        --success-color: #16a34a;
        --warning-color: #ea580c;
        --danger-color: #dc2626;
        --background-primary: #0f172a;
        --background-secondary: #1e293b;
        --background-tertiary: #334155;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --border-color: #475569;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--background-primary) 0%, var(--background-secondary) 100%);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Header */
    .app-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 400;
    }
    
    /* Cards */
    .card {
        background: var(--background-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    
    .card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.15);
    }
    
    .card h3 {
        color: var(--text-primary);
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Risk Level Cards */
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid;
        font-weight: 600;
    }
    
    .risk-low { 
        background: rgba(22, 163, 74, 0.1);
        border-color: var(--success-color);
        color: var(--success-color);
    }
    
    .risk-medium { 
        background: rgba(234, 88, 12, 0.1);
        border-color: var(--warning-color);
        color: var(--warning-color);
    }
    
    .risk-high { 
        background: rgba(220, 38, 38, 0.1);
        border-color: var(--danger-color);
        color: var(--danger-color);
    }
    
    /* Metrics */
    .metric {
        background: var(--background-tertiary);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid var(--border-color);
    }
    
    .metric h4 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        color: var(--primary-color);
    }
    
    .metric p {
        color: var(--text-secondary);
        margin: 0;
        font-size: 0.9rem;
    }
    
    /* Forms */
    .stSelectbox > div > div {
        background: var(--background-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        color: var(--text-primary);
    }
    
    .stNumberInput > div > div > input {
        background: var(--background-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        color: var(--text-primary);
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--primary-color);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: var(--secondary-color);
        transform: translateY(-1px);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-success {
        background: rgba(22, 163, 74, 0.2);
        color: var(--success-color);
    }
    
    .status-error {
        background: rgba(220, 38, 38, 0.2);
        color: var(--danger-color);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive */
    @media (max-width: 768px) {
        .app-title {
            font-size: 2rem;
        }
        .card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Model Loading Functions
@st.cache_resource
def load_saved_models():
    """Load saved models and preprocessor"""
    try:
        if not os.path.exists('models'):
            return None, None, None, False
            
        model = joblib.load('models/best_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        
        return model, preprocessor, feature_names, True
    except Exception as e:
        return None, None, None, False

def create_risk_gauge(risk_score, risk_color):
    """Create a clean gauge chart for risk score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score", 'font': {'size': 20, 'color': '#f8fafc'}},
        gauge = {
            'axis': {
                'range': [None, 100],
                'tickwidth': 1,
                'tickcolor': "#cbd5e1",
                'tickfont': {'color': '#cbd5e1', 'size': 12}
            },
            'bar': {'color': risk_color, 'thickness': 0.3},
            'steps': [
                {'range': [0, 30], 'color': "rgba(22, 163, 74, 0.2)"},
                {'range': [30, 60], 'color': "rgba(234, 88, 12, 0.2)"},
                {'range': [60, 100], 'color': "rgba(220, 38, 38, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "#f8fafc", 'width': 3},
                'thickness': 0.75,
                'value': risk_score
            },
            'borderwidth': 2,
            'bordercolor': "#475569"
        },
        number = {'font': {'size': 36, 'color': '#f8fafc'}}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def predict_default_risk(input_data):
    """Make prediction for applicant - simplified version"""
    try:
        # Mock prediction based on simple rules for demo
        credit_score = input_data.get('credit_score', 650)
        loan_to_income = input_data.get('loan_percent_income', 0.3)
        has_defaults = 1 if input_data.get('previous_loan_defaults_on_file') == 'Yes' else 0
        
        # Simple risk calculation
        risk_score = min(100, max(0, int(
            (850 - credit_score) * 0.15 +
            loan_to_income * 150 +
            has_defaults * 25 +
            np.random.normal(0, 5)  # Add some variance
        )))
        
        default_prob = risk_score / 100
        prediction = 'Default' if risk_score > 50 else 'No Default'
        
        # Determine risk level
        if risk_score < 30:
            risk_level = "Low Risk"
            risk_color = "#16a34a"
        elif risk_score < 60:
            risk_level = "Medium Risk" 
            risk_color = "#ea580c"
        else:
            risk_level = "High Risk"
            risk_color = "#dc2626"
        
        return {
            'success': True,
            'prediction': prediction,
            'default_probability': float(default_prob),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'confidence': float(max(default_prob, 1 - default_prob))
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load models
model, preprocessor, feature_names, model_loaded = load_saved_models()

# Header
st.markdown("""
<div class="app-header">
    <h1 class="app-title">CreditFortify</h1>
    <p class="app-subtitle">Credit Risk Assessment Platform</p>
</div>
""", unsafe_allow_html=True)

# Model Status
col_status1, col_status2 = st.columns([3, 1])
with col_status2:
    if model_loaded:
        st.markdown('<span class="status-badge status-success">Models Loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-error">Demo Mode</span>', unsafe_allow_html=True)

# Main Assessment Form
st.markdown('<div class="card"><h3>Credit Risk Assessment</h3></div>', unsafe_allow_html=True)

with st.form("assessment_form", clear_on_submit=False):
    # Personal Information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal Info")
        person_age = st.number_input("Age", min_value=18, max_value=100, value=35)
        person_income = st.number_input("Annual Income ($)", min_value=0, max_value=500000, value=50000, step=1000)
        person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=50, value=5)
    
    with col2:
        st.subheader("Loan Details")
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=100000, value=15000, step=500)
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=12.5, step=0.1)
        loan_intent = st.selectbox("Loan Purpose", 
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    
    with col3:
        st.subheader("Credit Info")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        cb_person_cred_hist_length = st.number_input("Credit History (years)", min_value=0, max_value=50, value=8)
        previous_loan_defaults_on_file = st.selectbox("Previous Defaults", ["No", "Yes"])
    
    # Additional fields in a single row
    col4, col5, col6 = st.columns(3)
    with col4:
        person_gender = st.selectbox("Gender", ["male", "female"])
    with col5:
        person_education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "Doctorate"])
    with col6:
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    
    # Calculate metrics
    loan_percent_income = loan_amnt / person_income if person_income > 0 else 0
    
    # Quick metrics display
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f'<div class="metric"><h4>{loan_percent_income:.1%}</h4><p>Loan-to-Income</p></div>', unsafe_allow_html=True)
    with col_m2:
        debt_level = "Low" if loan_percent_income < 0.2 else "Medium" if loan_percent_income < 0.4 else "High"
        st.markdown(f'<div class="metric"><h4>{debt_level}</h4><p>Debt Level</p></div>', unsafe_allow_html=True)
    with col_m3:
        coverage = person_income / loan_amnt if loan_amnt > 0 else 0
        st.markdown(f'<div class="metric"><h4>{coverage:.1f}x</h4><p>Income Coverage</p></div>', unsafe_allow_html=True)
    
    # Submit button
    submitted = st.form_submit_button("Assess Risk")

# Process Results
if submitted:
    input_data = {
        'person_age': person_age,
        'person_gender': person_gender,
        'person_education': person_education,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': person_home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
    }
    
    if person_income <= 0 or loan_amnt <= 0:
        st.error("Income and loan amount must be greater than 0")
    else:
        with st.spinner("Analyzing risk..."):
            result = predict_default_risk(input_data)
        
        if result['success']:
            # Store in history
            result['timestamp'] = datetime.now()
            result['input_data'] = input_data
            st.session_state.prediction_history.append(result)
            
            st.markdown('<div class="card"><h3>Assessment Results</h3></div>', unsafe_allow_html=True)
            
            # Results display
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                risk_class = "risk-low" if result['risk_score'] < 30 else "risk-medium" if result['risk_score'] < 60 else "risk-high"
                st.markdown(f'''
                <div class="{risk_class}">
                    <h3>{result['prediction']}</h3>
                    <p>{result['risk_level']}</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col_r2:
                st.markdown(f'<div class="metric"><h4>{result["risk_score"]}</h4><p>Risk Score</p></div>', unsafe_allow_html=True)
                
            with col_r3:
                st.markdown(f'<div class="metric"><h4>{result["default_probability"]:.1%}</h4><p>Default Probability</p></div>', unsafe_allow_html=True)
            
            # Risk gauge
            col_gauge1, col_gauge2, col_gauge3 = st.columns([1, 2, 1])
            with col_gauge2:
                gauge_fig = create_risk_gauge(result['risk_score'], result['risk_color'])
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Recommendation
            if result['risk_score'] < 30:
                st.success("Recommendation: Approve Application")
            elif result['risk_score'] < 60:
                st.warning("Recommendation: Manual Review Required")
            else:
                st.error("Recommendation: Decline Application")
        else:
            st.error(f"Prediction error: {result['error']}")

# Assessment History (if any)
if st.session_state.prediction_history:
    st.markdown('<div class="card"><h3>Recent Assessments</h3></div>', unsafe_allow_html=True)
    
    # Summary metrics
    total = len(st.session_state.prediction_history)
    high_risk = sum(1 for p in st.session_state.prediction_history if p.get('risk_score', 0) > 60)
    avg_risk = np.mean([p.get('risk_score', 0) for p in st.session_state.prediction_history])
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown(f'<div class="metric"><h4>{total}</h4><p>Total</p></div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown(f'<div class="metric"><h4>{high_risk}</h4><p>High Risk</p></div>', unsafe_allow_html=True)
    with col_s3:
        st.markdown(f'<div class="metric"><h4>{avg_risk:.1f}</h4><p>Avg Score</p></div>', unsafe_allow_html=True)
    with col_s4:
        if st.button("Clear History", type="secondary"):
            st.session_state.prediction_history = []
            st.rerun()
    
    # Recent history table
    if len(st.session_state.prediction_history) > 0:
        history_data = []
        for i, pred in enumerate(reversed(st.session_state.prediction_history[-5:])):  # Last 5 only
            history_data.append({
                'Time': pred.get('timestamp', datetime.now()).strftime('%H:%M:%S'),
                'Prediction': pred.get('prediction', 'Unknown'),
                'Risk Score': f"{pred.get('risk_score', 0)}/100",
                'Risk Level': pred.get('risk_level', 'Unknown')
            })
        
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: var(--text-muted); margin-top: 2rem;">CreditFortify- Professional Credit Risk Assessment</p>', 
    unsafe_allow_html=True
)
