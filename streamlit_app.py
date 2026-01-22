"""
=============================================================================
🎯 CUSTOMER SEGMENTATION - UNIFIED DASHBOARD
=============================================================================
Complete ML-Powered Customer Analytics & Recommendation Platform
Built with Streamlit, K-Means, and Python
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

current_dir = Path(__file__).parent
MODELS_PATH = current_dir / "models"
DATA_PATH = current_dir / "data" / "processed"
RAW_DATA_PATH = current_dir / "data" / "raw"

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/NourLouta',
        'Report a bug': 'https://github.com/NourLouta',
        'About': '# Customer Segmentation System\nBuilt with ❤️ using Streamlit & K-Means'
    }
)

# ============================================================================
# CUSTOM CSS STYLING (TEAL THEME)
# ============================================================================

def load_custom_css():
    """Load premium custom CSS with modern teal design"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* ========== GLOBAL STYLES ========== */
    * {
        font-family: 'Inter', 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background with animated teal gradient */
    .main {
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 50%, #03045e 100%);
        background-attachment: fixed;
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Content container with glassmorphism */
    .block-container {
        padding: 2rem 3rem;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 25px;
        box-shadow: 0 25px 70px rgba(0,0,0,0.25);
        backdrop-filter: blur(20px);
        margin: 2rem auto;
        max-width: 1500px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* ========== HEADER STYLING ========== */
    .main-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%);
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 15px 40px rgba(0, 180, 216, 0.5);
        animation: slideDown 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.3rem;
        margin-top: 0.8rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* ========== METRIC CARDS (TEAL GRADIENT) ========== */
    .metric-card {
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 180, 216, 0.4);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: fadeInUp 0.6s ease-out;
        margin: 1rem 0;
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-15px) scale(1.05);
        box-shadow: 0 20px 50px rgba(0, 180, 216, 0.6);
    }
    
    .metric-card h4 {
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card p {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .metric-card small {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* ========== TABS STYLING (TEAL THEME) ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: linear-gradient(90deg, #00b4d8 0%, #0077b6 100%);
        padding: 15px;
        border-radius: 20px;
        box-shadow: 0 5px 20px rgba(0, 180, 216, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.15);
        color: white;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 600;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255,255,255,0.25);
        transform: translateY(-3px);
        border-color: rgba(255,255,255,0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #0077b6 !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        transform: translateY(-2px);
    }
    
    /* ========== BUTTONS (TEAL THEME) ========== */
    .stButton > button {
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%);
        color: white;
        border: none;
        padding: 14px 35px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(0, 180, 216, 0.5);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(0, 180, 216, 0.7);
        background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-2px);
    }
    
    /* ========== INPUT FIELDS ========== */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input {
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 12px 18px;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextInput > div > div > input:focus {
        border-color: #00b4d8;
        box-shadow: 0 0 0 4px rgba(0, 180, 216, 0.15);
        outline: none;
    }
    
    /* ========== INFO BOXES ========== */
    .info-box {
        background: linear-gradient(135deg, rgba(0, 180, 216, 0.1) 0%, rgba(0, 180, 216, 0.05) 100%);
        border-left: 5px solid #00b4d8;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(0, 200, 83, 0.1) 0%, rgba(0, 200, 83, 0.05) 100%);
        border-left: 5px solid #00c853;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%);
        border-left: 5px solid #ffc107;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.1);
    }
    
    /* ========== PREDICTION RESULT ========== */
    .prediction-result {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%);
        border-radius: 25px;
        color: white;
        margin: 2.5rem 0;
        box-shadow: 0 15px 40px rgba(0, 180, 216, 0.5);
        animation: pulse 2s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-result::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 15s linear infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* ========== CLUSTER CARDS (DIFFERENT COLORS) ========== */
    .cluster-card-0 {
        background: linear-gradient(135deg, #00b4d8 0%, #0096c7 100%);
    }
    
    .cluster-card-1 {
        background: linear-gradient(135deg, #e63946 0%, #d62828 100%);
    }
    
    .cluster-card-2 {
        background: linear-gradient(135deg, #06d6a0 0%, #00c896 100%);
    }
    
    .cluster-card-3 {
        background: linear-gradient(135deg, #f77f00 0%, #f48c06 100%);
    }
    
    .cluster-card-4 {
        background: linear-gradient(135deg, #9b5de5 0%, #7209b7 100%);
    }
    
    /* ========== ANIMATIONS ========== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-60px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* ========== RESPONSIVE DESIGN ========== */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2.2rem; }
        .block-container { padding: 1rem; margin: 1rem; }
        .metric-card p { font-size: 2rem; }
        .prediction-result { font-size: 2.5rem; padding: 2rem 1rem; }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_customer_data():
    """Load customer segmentation data"""
    try:
        df = pd.read_csv(DATA_PATH / "customer_segments.csv")
        return df
    except Exception as e:
        st.error(f"❌ Error loading customer data: {str(e)}")
        return None

@st.cache_data
def load_transaction_data():
    """Load transaction data"""
    try:
        transactions = pd.read_csv(RAW_DATA_PATH / "Cleaned_Data_Merchant_Level_2.csv")
        transactions.rename(columns={
            'Mer_Id': 'Merchant_Id',
            'Trx_Vlu': 'Transaction_Amount',
            'Category In English': 'Category'
        }, inplace=True)
        return transactions
    except Exception as e:
        st.error(f"❌ Error loading transaction data: {str(e)}")
        return None

@st.cache_data
def load_recommendations():
    """Load recommendation data"""
    try:
        merchant_recs = pd.read_csv(DATA_PATH / "merchant_recommendations_by_cluster.csv")
        category_recs = pd.read_csv(DATA_PATH / "category_recommendations_by_cluster.csv")
        return merchant_recs, category_recs
    except Exception as e:
        st.warning(f"⚠️ Recommendation data not available: {str(e)}")
        return None, None

@st.cache_resource
def load_model():
    """Load clustering model"""
    try:
        model = joblib.load(MODELS_PATH / "kmeans_model.pkl")
        scaler = joblib.load(MODELS_PATH / "scaler.pkl")
        return model, scaler
    except Exception as e:
        st.warning(f"⚠️ Model files not found: {str(e)}")
        return None, None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    load_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Customer Segmentation Dashboard</h1>
        <p>AI-Powered Customer Intelligence & Recommendation Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_customer_data()
    transactions = load_transaction_data()
    merchant_recs, category_recs = load_recommendations()
    model, scaler = load_model()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        st.markdown("---")
        
        # Data status
        if df is not None:
            st.success(f"✅ Customers Loaded ({len(df):,})")
        else:
            st.error("❌ Data Not Loaded")
        
        if transactions is not None:
            st.success(f"✅ Transactions Loaded ({len(transactions):,})")
        
        st.markdown("---")
        
        # Quick stats
        if df is not None:
            st.markdown("### 📊 Quick Stats")
            st.metric("Total Customers", f"{len(df):,}")
            st.metric("Total Segments", len(df['KMeans_Cluster'].unique()))
            st.metric("Total Revenue", f"₹{df['Total_Spend'].sum():,.0f}")
            st.metric("Avg Customer Value", f"₹{df['Total_Spend'].mean():,.0f}")
        
        st.markdown("---")
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Home",
        "📊 Segment Overview",
        "🔍 Customer Lookup",
        "🎯 Recommendations",
        "ℹ️ About"
    ])
    
    # TAB 1: HOME
    with tab1:
        show_home_page(df, transactions)
    
    # TAB 2: SEGMENT OVERVIEW
    with tab2:
        show_segment_overview(df, transactions)
    
    # TAB 3: CUSTOMER LOOKUP
    with tab3:
        show_customer_lookup(df, transactions)
    
    # TAB 4: RECOMMENDATIONS
    with tab4:
        show_recommendations_page(df, merchant_recs, category_recs)
    
    # TAB 5: ABOUT
    with tab5:
        show_about_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
        <p>Built with ❤️ using Streamlit, K-Means, and Python</p>
        <p>© 2026 Customer Segmentation System | Version 1.0</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 1: HOME
# ============================================================================

def show_home_page(df, transactions):
    """Display home page"""
    
    st.markdown("## 🏠 Welcome to Customer Segmentation Dashboard")
    
    # Hero section
    st.markdown("""
    <div class="info-box">
        <h3 style="margin-top: 0;">🎯 About This System</h3>
        <p style="font-size: 1.05rem; line-height: 1.7;">
            This intelligent system uses <strong>K-Means Clustering</strong> and <strong>RFM Analysis</strong> 
            to segment customers into <strong>5 actionable groups</strong>. Get personalized recommendations 
            and insights to boost customer engagement and revenue!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.error("❌ Data not loaded. Please check data files.")
        return
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>👥 Customers</h4>
            <p>{:,}</p>
            <small>Total Users</small>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>💰 Revenue</h4>
            <p>₹{:,.0f}</p>
            <small>Total Spend</small>
        </div>
        """.format(df['Total_Spend'].sum()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Avg Value</h4>
            <p>₹{:,.0f}</p>
            <small>Per Customer</small>
        </div>
        """.format(df['Total_Spend'].mean()), unsafe_allow_html=True)
    
    with col4:
        active_customers = len(df[df['Recency'] <= 90])
        st.markdown("""
        <div class="metric-card">
            <h4>✅ Active</h4>
            <p>{:,}</p>
            <small>{:.1f}% of Total</small>
        </div>
        """.format(active_customers, (active_customers/len(df))*100), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Cluster overview cards
    st.markdown("## 🎨 Customer Segments Overview")
    
    cluster_colors = {
        0: "#00b4d8",  # Teal
        1: "#e63946",  # Red
        2: "#06d6a0",  # Green
        3: "#f77f00",  # Orange
        4: "#9b5de5"   # Purple
    }
    
    cols = st.columns(5)
    
    for idx, (cluster_id, cluster_name) in enumerate(df.groupby('KMeans_Cluster')['Cluster_Name'].first().items()):
        cluster_data = df[df['KMeans_Cluster'] == cluster_id]
        
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-card cluster-card-{cluster_id}" style="background: {cluster_colors[cluster_id]};">
                <h3 style="color: white; margin: 0;">{cluster_name}</h3>
                <h2 style="color: white; margin: 10px 0;">{len(cluster_data):,}</h2>
                <p style="margin: 0;">customers</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Avg Spend", f"₹{cluster_data['Total_Spend'].mean():,.0f}")
            st.metric("Avg Frequency", f"{cluster_data['Frequency'].mean():.1f}")
    
    st.markdown("---")
    
    # Visualizations
    if transactions is not None:
        st.markdown("## 📊 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster distribution
            cluster_dist = df['Cluster_Name'].value_counts().reset_index()
            cluster_dist.columns = ['Segment', 'Count']
            
            fig = px.pie(
                cluster_dist,
                values='Count',
                names='Segment',
                title='Customer Distribution by Segment',
                hole=0.4,
                color_discrete_sequence=['#00b4d8', '#e63946', '#06d6a0', '#f77f00', '#9b5de5']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Revenue by segment
            segment_revenue = df.groupby('Cluster_Name')['Total_Spend'].sum().reset_index()
            segment_revenue = segment_revenue.sort_values('Total_Spend', ascending=False)
            
            fig = px.bar(
                segment_revenue,
                x='Cluster_Name',
                y='Total_Spend',
                title='Total Revenue by Segment',
                color='Total_Spend',
                color_continuous_scale='Teal'
            )
            fig.update_layout(template='plotly_white', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # How to use
    st.markdown("### 📖 How to Use This System")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%); 
                    border-radius: 15px; color: white; margin: 0.5rem 0;">
            <h3 style="margin: 0; font-size: 2.5rem;">1️⃣</h3>
            <h4 style="margin: 0.5rem 0;">Explore</h4>
            <p style="margin: 0; font-size: 0.9rem;">View segment analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%); 
                    border-radius: 15px; color: white; margin: 0.5rem 0;">
            <h3 style="margin: 0; font-size: 2.5rem;">2️⃣</h3>
            <h4 style="margin: 0.5rem 0;">Search</h4>
            <p style="margin: 0; font-size: 0.9rem;">Lookup customer profiles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%); 
                    border-radius: 15px; color: white; margin: 0.5rem 0;">
            <h3 style="margin: 0; font-size: 2.5rem;">3️⃣</h3>
            <h4 style="margin: 0.5rem 0;">Recommend</h4>
            <p style="margin: 0; font-size: 0.9rem;">Get personalized offers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%); 
                    border-radius: 15px; color: white; margin: 0.5rem 0;">
            <h3 style="margin: 0; font-size: 2.5rem;">4️⃣</h3>
            <h4 style="margin: 0.5rem 0;">Act</h4>
            <p style="margin: 0; font-size: 0.9rem;">Implement strategies</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: SEGMENT OVERVIEW
# ============================================================================

def show_segment_overview(df, transactions):
    """Display segment overview"""
    
    st.markdown("## 📊 Segment Overview & Analytics")
    
    if df is None:
        st.error("❌ Data not loaded")
        return
    
    # Segment comparison
    st.markdown("### 📈 Segment Comparison")
    
    segment_stats = df.groupby('Cluster_Name').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Total_Spend': 'mean',
        'RFM_Score': 'mean',
        'User_Id': 'count'
    }).reset_index()
    
    segment_stats.columns = ['Segment', 'Avg Recency', 'Avg Frequency', 'Avg Spend', 'Avg RFM Score', 'Customer Count']
    
    st.dataframe(
        segment_stats.style.format({
            'Avg Recency': '{:.0f}',
            'Avg Frequency': '{:.1f}',
            'Avg Spend': '₹{:,.2f}',
            'Avg RFM Score': '{:.2f}',
            'Customer Count': '{:,}'
        }).background_gradient(subset=['Avg Spend'], cmap='teal'),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # RFM Analysis
    st.markdown("### 📊 RFM Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            df,
            x='Recency',
            y='Frequency',
            color='Cluster_Name',
            size='Total_Spend',
            hover_data=['User_Id', 'RFM_Score'],
            title='Recency vs Frequency',
            color_discrete_sequence=['#00b4d8', '#e63946', '#06d6a0', '#f77f00', '#9b5de5']
        )
        fig.update_layout(template='plotly_white', height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df,
            x='Frequency',
            y='Total_Spend',
            color='Cluster_Name',
            size='RFM_Score',
            hover_data=['User_Id', 'Recency'],
            title='Frequency vs Monetary Value',
            color_discrete_sequence=['#00b4d8', '#e63946', '#06d6a0', '#f77f00', '#9b5de5']
        )
        fig.update_layout(template='plotly_white', height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Category analysis
    if transactions is not None:
        st.markdown("### 🏷️ Category Performance by Segment")
        
        transactions_with_clusters = transactions.merge(
            df[['User_Id', 'Cluster_Name']],
            on='User_Id'
        )
        
        category_by_segment = transactions_with_clusters.groupby(['Cluster_Name', 'Category'])['Transaction_Amount'].sum().reset_index()
        
        fig = px.sunburst(
            category_by_segment,
            path=['Cluster_Name', 'Category'],
            values='Transaction_Amount',
            title='Category Distribution by Segment',
            color_discrete_sequence=px.colors.sequential.Teal
        )
        fig.update_layout(template='plotly_white', height=600)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: CUSTOMER LOOKUP
# ============================================================================

def show_customer_lookup(df, transactions):
    """Display customer lookup page"""
    
    st.markdown("## 🔍 Customer Profile Lookup")
    
    if df is None:
        st.error("❌ Data not loaded")
        return
    
    # Customer search
    customer_id = st.number_input(
        "Enter Customer ID",
        min_value=int(df['User_Id'].min()),
        max_value=int(df['User_Id'].max()),
        value=int(df['User_Id'].iloc[0])
    )
    
    if st.button("🔍 Search Customer", use_container_width=True):
        if customer_id in df['User_Id'].values:
            customer = df[df['User_Id'] == customer_id].iloc[0]
            
            st.markdown("---")
            st.markdown(f"## Customer Profile: **{customer_id}**")
            
            # Customer metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Segment", customer['Cluster_Name'])
            
            with col2:
                st.metric("Recency", f"{int(customer['Recency'])} days")
            
            with col3:
                st.metric("Frequency", f"{int(customer['Frequency'])} txns")
            
            with col4:
                st.metric("Total Spend", f"₹{customer['Total_Spend']:,.2f}")
            
            st.markdown("---")
            
            # Transaction history
            if transactions is not None:
                st.markdown("### 📜 Transaction History")
                
                customer_txns = transactions[transactions['User_Id'] == customer_id]
                
                st.dataframe(
                    customer_txns[['Transaction_Amount', 'Category', 'Merchant_Id']].head(10),
                    use_container_width=True
                )
                
                # Spending analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    category_spend = customer_txns.groupby('Category')['Transaction_Amount'].sum().reset_index()
                    fig = px.bar(category_spend, x='Category', y='Transaction_Amount', 
                                title='Spending by Category',
                                color='Transaction_Amount',
                                color_continuous_scale='Teal')
                    fig.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    merchant_spend = customer_txns.groupby('Merchant_Id')['Transaction_Amount'].sum().nlargest(5).reset_index()
                    fig = px.bar(merchant_spend, x='Merchant_Id', y='Transaction_Amount', 
                                title='Top 5 Merchants',
                                color='Transaction_Amount',
                                color_continuous_scale='Teal')
                    fig.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error(f"❌ Customer ID {customer_id} not found!")

# ============================================================================
# PAGE 4: RECOMMENDATIONS
# ============================================================================

def show_recommendations_page(df, merchant_recs, category_recs):
    """Display recommendations page"""
    
    st.markdown("## 🎯 Personalized Recommendations")
    
    if df is None:
        st.error("❌ Data not loaded")
        return
    
    # Customer selection
    customer_id = st.selectbox(
        "Select Customer ID",
        options=df['User_Id'].values,
        index=0
    )
    
    if st.button("🎁 Generate Recommendations", use_container_width=True):
        customer = df[df['User_Id'] == customer_id].iloc[0]
        cluster_id = customer['KMeans_Cluster']
        
        st.markdown("---")
        st.markdown(f"## Recommendations for Customer **{customer_id}**")
        st.markdown(f"### Segment: {customer['Cluster_Name']}")
        
        # Personalized offer
        offers = {
            "🌟 VIP Champions": "Exclusive 20% off + Free Premium Delivery",
            "💎 High-Value Loyalists": "15% off + Double Loyalty Points",
            "🆕 Recent Shoppers": "Welcome Bonus: 10% off your next 3 purchases",
            "😴 At-Risk/Dormant": "We miss you! 25% off to come back",
            "🛒 Regular Customers": "Special offer: 10% off + Free Shipping"
        }
        
        st.markdown(f"""
        <div class="success-box">
            <h3 style="margin-top: 0;">🎁 Personalized Offer</h3>
            <p style="font-size: 1.2rem; margin: 0;">{offers.get(customer['Cluster_Name'], 'Special discount available!')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recommendations
        col1, col2 = st.columns(2)
        
        if merchant_recs is not None:
            with col1:
                st.markdown("### 🏪 Recommended Merchants")
                
                top_merchants = merchant_recs[merchant_recs['Cluster'] == cluster_id].nlargest(5, 'Merchant_Score')
                
                for _, merchant in top_merchants.iterrows():
                    st.markdown(f"""
                    - **Merchant {int(merchant['Merchant_Id'])}**
                      - Score: {merchant['Merchant_Score']:.2f}
                      - Avg Transaction: ₹{merchant['Avg_Transaction']:,.2f}
                    """)
        
        if category_recs is not None:
            with col2:
                st.markdown("### 🏷️ Recommended Categories")
                
                top_categories = category_recs[category_recs['Cluster'] == cluster_id].nlargest(5, 'Total_Spend')
                
                for _, category in top_categories.iterrows():
                    st.markdown(f"""
                    - **{category['Category']}**
                      - Popularity: {category['Transaction_Count']} transactions
                      - Total Spend: ₹{category['Total_Spend']:,.2f}
                    """)

# ============================================================================
# PAGE 5: ABOUT
# ============================================================================

def show_about_page():
    """Display about page"""
    
    st.markdown("## ℹ️ About This System")
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Customer Segmentation System</h3>
        <p>An advanced machine learning system for customer segmentation and personalized recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Project Overview")
    
    st.markdown("""
    This system uses **K-Means Clustering** and **RFM Analysis** to segment customers into 5 actionable groups:
    
    - **🌟 VIP Champions**: Highest value customers
    - **💎 High-Value Loyalists**: Frequent, high-spending customers
    - **🆕 Recent Shoppers**: New or recently active customers
    - **😴 At-Risk/Dormant**: Customers who need re-engagement
    - **🛒 Regular Customers**: Consistent, moderate spenders
    
    The system provides personalized recommendations and insights to boost engagement and revenue.
    """)
    
    st.markdown("---")
    
    # Developer info
    st.markdown("### 👨‍💻 Developer Information")
    
    st.markdown("""
    <div class="success-box">
        <p><strong>👤 Developed by:</strong> Nour Eldeen Mohammed</p>
        <p><strong>🚀 Project:</strong> Customer Segmentation System</p>
        <p><strong>📌 Version:</strong> 1.0</p>
        <p><strong>📅 Last Updated:</strong> January 2026</p>
        <p><strong>🎓 Institution:</strong> Epsilon AI - AI Internship Program</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📧 Connect With Me")
    
    st.markdown("""
    <style>
    .social-btn {
        display: inline-block;
        padding: 12px 24px;
        margin: 10px 5px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .social-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .email-btn {
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%);
        color: white;
    }
    
    .linkedin-btn {
        background: linear-gradient(135deg, #0077b5 0%, #00a0dc 100%);
        color: white;
    }
    
    .github-btn {
        background: linear-gradient(135deg, #333 0%, #555 100%);
        color: white;
    }
    </style>
    
    <div style="text-align: center; margin: 30px 0;">
        <a href="mailto:nourlouta@gmail.com" class="social-btn email-btn" target="_blank">
            📧 Email Me
        </a>
        <a href="https://www.linkedin.com/in/nour-eldeen-mohammed-mba-0b439721a/" 
        class="social-btn linkedin-btn" target="_blank">
            💼 LinkedIn Profile
        </a>
        <a href="https://github.com/NourLouta" class="social-btn github-btn" target="_blank">
            🐙 GitHub Portfolio
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Copyright
    st.markdown("""
    <div style="text-align: center; padding: 15px; 
                background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%); 
                border-radius: 10px; color: white; margin-top: 30px;">
        <p style="margin: 0; font-size: 14px;">
            © 2026 <strong>Nour Eldeen Mohammed</strong> | All Rights Reserved
        </p>
        <p style="margin: 5px 0 0 0; font-size: 13px; opacity: 0.9;">
            Built with ❤️ using Python, Streamlit & Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
