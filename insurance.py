import base64
import calendar
import hashlib
import json
import pickle
import sqlite3
import time
import warnings
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Configure the page with modern settings
st.set_page_config(
    page_title="Gen-Health FWA Detection System",
    page_icon="C:/Users/tmapillar/Final FWA 2025/Generation health logo.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling with medication-specific styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }

    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        margin: 1rem 0;
    }

    .flag-fraud { background-color: #ffebee; border-left: 4px solid #f44336; }
    .flag-waste { background-color: #fff3e0; border-left: 4px solid #ff9800; }
    .flag-abuse { background-color: #e8f5e8; border-left: 4px solid #4caf50; }

    .summary-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }

    .history-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: transform 0.2s;
    }

    .history-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }

    .history-card.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    .trend-up { color: #4CAF50; }
    .trend-down { color: #F44336; }
    .trend-neutral { color: #FF9800; }

    .risk-high { background-color: #ffebee; color: #d32f2f; }
    .risk-medium { background-color: #fff3e0; color: #f57c00; }
    .risk-low { background-color: #e8f5e9; color: #388e3c; }

    .insight-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }

    /* Medication-specific styling */
    .medication-chronic {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }

    .medication-acute {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }

    .medication-otc {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }

    .medication-highlight {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 4px solid #9c27b0;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }

    .benefit-analysis {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }

    .medication-card {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }

    .chronic-bg { background-color: #e3f2fd; }
    .acute-bg { background-color: #fff3e0; }
    .otc-bg { background-color: #e8f5e9; }

    .medication-header {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #333;
    }

    .benefit-distribution {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


class ActurialAnalytics:
    """Acturial analytical methods for FWA detection"""

    # ... (unchanged, keep as provided)
    @staticmethod
    def calculate_benfords_law(data_column):
        # ... (as before)
        pass

    @staticmethod
    def detect_statistical_outliers(data, column='TOTAL PAID', method='iqr'):
        # ... (as before)
        pass

    @staticmethod
    def calculate_claim_pattern_metrics(claims_data):
        # ... (as before)
        pass

    @staticmethod
    def perform_temporal_analysis(claims_data):
        # ... (as before)
        pass

    @staticmethod
    def calculate_provider_risk_scores(claims_data, flagged_data):
        # ... (as before)
        pass

    @staticmethod
    def analyze_benefit_patterns(claims_data):
        # ... (as before)
        pass

    @staticmethod
    def analyze_medication_patterns(claims_data):
        # ... (as before)
        pass

class HistoricalAnalysisManager:
    """Minimal implementation to avoid AttributeError."""
    def __init__(self):
        self.analyses = []

    def save_analysis(self, detector):
        """Save analysis metadata (simplified)."""
        self.analyses.append({
            'timestamp': datetime.now(),
            'name': getattr(detector, 'analysis_name', 'Unnamed'),
            'flagged_count': detector.flagged_claims['FWA_Flag'].sum() if detector.flagged_claims is not None else 0
        })


class FWADetector:
    def __init__(self):
        self.claims_data = None
        self.tariff_data = None
        self.flagged_claims = None
        self.history_manager = HistoricalAnalysisManager()
        self.analytics = ActurialAnalytics()

    def load_data(self, claims_file, tariff_file):
        """Load claims and tariff data from uploaded Excel files."""
        try:
            # Read tariff file (sheet 'Sample Tariff')
            self.tariff_data = pd.read_excel(tariff_file, sheet_name='Sample Tariff')
            # Read claims file (assuming first sheet)
            self.claims_data = pd.read_excel(claims_file)
            # Ensure required columns exist (optional, but good practice)
            required_cols = ['MEMBER GENDER', 'SERVICE DATE', 'TOTAL PAID', 'PROVIDER NAME']
            for col in required_cols:
                if col not in self.claims_data.columns:
                    return False, f"Missing required column: {col}"
            return True, "Data loaded successfully"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"

    def detect_fwa(self, analysis_name="Unnamed Analysis"):
        """Run FWA detection logic and populate flagged_claims."""
        if self.claims_data is None or self.tariff_data is None:
            return False, "No data loaded. Please load claims and tariff files first."

        try:
            # Create a copy of claims data to add flag columns
            self.flagged_claims = self.claims_data.copy()
            self.flagged_claims['FWA_Flag'] = False
            self.flagged_claims['Flag_Reason'] = ''
            self.flagged_claims['Flag_Type'] = ''

            # Run various checks (each method adds flags)
            self._run_basic_checks()
            self._perform_Acturial_analytics()
            self._check_gender_appropriateness()
            self._check_male_inappropriate_codes()
            self._check_female_inappropriate_codes()
            self._check_age_appropriateness()
            self._check_unit_overdosing()
            self._check_duplicate_claims()

            # Store analysis metadata
            self.analysis_name = analysis_name
            self.analysis_timestamp = datetime.now()

            # Optionally save to history
            self.history_manager.save_analysis(self)

            return True, f"Analysis '{analysis_name}' completed. Found {self.flagged_claims['FWA_Flag'].sum()} flagged claims."
        except Exception as e:
            return False, f"Error during analysis: {str(e)}"

    # ------------------------------------------------------------------
    # Internal detection methods (simplified examples)
    # ------------------------------------------------------------------
    def _run_basic_checks(self):
        """Placeholder for basic checks."""
        # Example: flag claims with zero amount
        mask = (self.flagged_claims['TOTAL PAID'] == 0)
        self._add_flag(mask, "Zero paid amount", "W")

    def _perform_Acturial_analytics(self):
        """Call Acturial analytics methods (simplified)."""
        # Example: outlier detection using IQR
        if 'TOTAL PAID' in self.flagged_claims.columns:
            Q1 = self.flagged_claims['TOTAL PAID'].quantile(0.25)
            Q3 = self.flagged_claims['TOTAL PAID'].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            outlier_mask = (self.flagged_claims['TOTAL PAID'] < lower) | (self.flagged_claims['TOTAL PAID'] > upper)
            self._add_flag(outlier_mask, "Statistical outlier (IQR)", "A")

    def _check_gender_appropriateness(self):
        """Flag claims where service is inappropriate for gender (example)."""
        # This is a simplified demo – in reality you'd use a mapping from tariff data
        # For illustration, flag any claim where MEMBER GENDER is 'F' and some condition
        if 'MEMBER GENDER' not in self.flagged_claims.columns:
            return
        # Assume some codes are male‑only (e.g., prostate exams)
        male_only_codes = ['99214']  # placeholder
        mask = self.flagged_claims['MEMBER GENDER'].str.upper() == 'F'
        mask &= self.flagged_claims['PROCEDURE CODE'].isin(male_only_codes)
        self._add_flag(mask, "Gender inappropriate (male service for female)", "F")

    def _check_male_inappropriate_codes(self):
        """Flag male‑inappropriate services (e.g., pregnancy tests)."""
        if 'MEMBER GENDER' not in self.flagged_claims.columns:
            return
        female_only_codes = ['PREG']  # placeholder
        mask = self.flagged_claims['MEMBER GENDER'].str.upper() == 'M'
        mask &= self.flagged_claims['PROCEDURE CODE'].isin(female_only_codes)
        self._add_flag(mask, "Gender inappropriate (female service for male)", "F")

    def _check_female_inappropriate_codes(self):
        """Alias for female‑inappropriate checks (could merge with above)."""
        pass  # already handled in _check_male_inappropriate_codes if needed

    def _check_age_appropriateness(self):
        """Flag services that are age‑inappropriate (simplified)."""
        if 'MEMBER AGE' not in self.flagged_claims.columns:
            return
        # Example: pediatric services for elderly
        pediatric_codes = ['PED']
        mask = self.flagged_claims['MEMBER AGE'] > 18
        mask &= self.flagged_claims['PROCEDURE CODE'].isin(pediatric_codes)
        self._add_flag(mask, "Age inappropriate (pediatric service for adult)", "A")

    def _check_unit_overdosing(self):
        """Flag unusually high units claimed."""
        if 'UNITS' not in self.flagged_claims.columns:
            return
        # Flag claims with units > 30
        mask = self.flagged_claims['UNITS'] > 30
        self._add_flag(mask, "Units claimed exceed typical limit", "W")

    def _check_duplicate_claims(self):
        """Flag duplicate claims based on key fields."""
        if 'CLAIM ID' in self.flagged_claims.columns:
            # Simple duplicate check on claim ID
            dup_mask = self.flagged_claims.duplicated(subset=['CLAIM ID'], keep=False)
            self._add_flag(dup_mask, "Duplicate claim ID", "W")
        else:
            # Fallback: duplicate on service date, provider, amount
            dup_cols = ['PROVIDER NAME', 'SERVICE DATE', 'TOTAL PAID']
            if all(col in self.flagged_claims.columns for col in dup_cols):
                dup_mask = self.flagged_claims.duplicated(subset=dup_cols, keep=False)
                self._add_flag(dup_mask, "Duplicate claim (date/provider/amount)", "W")

    def _add_flag(self, mask, reason, flag_type):
        """Helper to set flag columns for rows matching mask."""
        self.flagged_claims.loc[mask, 'FWA_Flag'] = True
        # Append reason (avoid overwriting multiple flags)
        self.flagged_claims.loc[mask, 'Flag_Reason'] = (
            self.flagged_claims.loc[mask, 'Flag_Reason'] + '; ' + reason
        ).str.lstrip('; ')
        self.flagged_claims.loc[mask, 'Flag_Type'] = (
            self.flagged_claims.loc[mask, 'Flag_Type'] + flag_type
        )

    def generate_executive_summary(self):
        """Return a simple executive summary string."""
        if self.flagged_claims is None:
            return "No analysis performed."
        total = len(self.flagged_claims)
        flagged = self.flagged_claims['FWA_Flag'].sum()
        pct = (flagged / total * 100) if total else 0
        amount = self.flagged_claims.loc[self.flagged_claims['FWA_Flag'], 'TOTAL PAID'].sum()
        return (f"Analysis of {total} claims identified {flagged} flagged claims "
                f"({pct:.1f}%) with a total paid amount of ${amount:,.2f}.")

    def debug_gender_flags(self):
        """Debug method – unchanged from your original."""
        if self.flagged_claims is None:
            return "No data available"
        flagged = self.flagged_claims[self.flagged_claims['FWA_Flag'] == True]
        gender_flags = flagged[
            flagged['Flag_Reason'].str.contains('gender|Gender|male|Male|female|Female|cannot', case=False, na=False)
        ]['Flag_Reason'].unique()
        result = "=== Gender Flag Debug ===\n"
        result += f"Total flagged claims: {len(flagged)}\n"
        result += f"Gender-related flags found: {len(gender_flags)}\n\n"
        for i, flag in enumerate(gender_flags[:10], 1):
            result += f"{i}. {flag}\n"
        if len(gender_flags) > 10:
            result += f"... and {len(gender_flags) - 10} more\n"
        return result

class HistoricalAnalysisManager:
    # ... (unchanged)
    pass


class FWADetector:
    def __init__(self):
        self.claims_data = None
        self.tariff_data = None
        self.flagged_claims = None
        self.history_manager = HistoricalAnalysisManager()
        self.analytics = ActurialAnalytics()

    # ... (all existing methods, including detect_fwa, _run_basic_checks, etc.)

    # ADD THE DEBUG METHOD HERE (Fix 3)
    def debug_gender_flags(self):
        """Debug method to see what gender flags are being detected"""
        if self.flagged_claims is None:
            return "No data available"

        flagged = self.flagged_claims[self.flagged_claims['FWA_Flag'] == True]

        # Find all unique flag reasons related to gender
        gender_flags = flagged[
            flagged['Flag_Reason'].str.contains('gender|Gender|male|Male|female|Female|cannot', case=False, na=False)
        ]['Flag_Reason'].unique()

        result = "=== Gender Flag Debug ===\n"
        result += f"Total flagged claims: {len(flagged)}\n"
        result += f"Gender-related flags found: {len(gender_flags)}\n\n"

        for i, flag in enumerate(gender_flags[:10], 1):
            result += f"{i}. {flag}\n"

        if len(gender_flags) > 10:
            result += f"... and {len(gender_flags) - 10} more\n"

        return result

    # ... (rest of the class methods remain unchanged)
    def load_data(self, claims_file, tariff_file):
        # ... (as before)
        pass

    def detect_fwa(self, analysis_name="Unnamed Analysis"):
        # ... (as before)
        pass

    def _run_basic_checks(self):
        # ... (as before)
        pass

    def _perform_Acturial_analytics(self):
        # ... (as before)
        pass

    def _check_gender_appropriateness(self):
        # ... (as before)
        pass

    def _check_male_inappropriate_codes(self):
        # ... (as before)
        pass

    def _check_female_inappropriate_codes(self):
        # ... (as before)
        pass

    def _check_age_appropriateness(self):
        # ... (as before)
        pass

    def _check_unit_overdosing(self):
        # ... (as before)
        pass

    def _check_duplicate_claims(self):
        # ... (as before)
        pass

    def _add_flag(self, index, reason, flag_type):
        # ... (as before)
        pass

    def generate_executive_summary(self):
        # ... (as before)
        pass

    def _generate_key_findings(self, flagged_claims):
        # ... (as before)
        pass


def create_Acturial_analytics_dashboard(flagged_data, claims_data=None):
    # ... (as before)
    pass


def create_key_insights_panel(flagged_data):
    # ... (as before)
    pass


def get_download_link(df, filename):
    # ... (as before)
    pass


def main():
    # Header with modern design
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div class="main-header">FWA Detection System</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Acturial Analytics for Fraud, Waste & Abuse Detection in Healthcare Claims</div>',
            unsafe_allow_html=True)

    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("###  Acturial Analytics")

    # Initialize session state
    if 'fwa_detector' not in st.session_state:
        st.session_state.fwa_detector = FWADetector()
    if 'tariff_uploaded' not in st.session_state:
        st.session_state.tariff_uploaded = False
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False
    if 'last_analysis_id' not in st.session_state:
        st.session_state.last_analysis_id = None
    if 'analysis_name' not in st.session_state:
        st.session_state.analysis_name = ""
    if 'historical_view' not in st.session_state:
        st.session_state.historical_view = False

    # Modern sidebar for file uploads and history
    with st.sidebar:
        st.markdown("### 📁 Data Management")

        historical_mode = st.toggle("📊 View Historical Analytics", value=st.session_state.historical_view)
        if historical_mode != st.session_state.historical_view:
            st.session_state.historical_view = historical_mode
            st.rerun()

        if not st.session_state.historical_view:
            st.markdown("#### 1. Upload Tariff File")
            st.markdown('<div class="upload-box">Drag and drop your tariff file here</div>', unsafe_allow_html=True)
            tariff_file = st.file_uploader("", type=['xlsx'], key='tariff', label_visibility="collapsed")

            if tariff_file:
                try:
                    st.session_state.tariff_data = pd.read_excel(tariff_file, sheet_name='Sample Tariff')
                    st.session_state.tariff_uploaded = True
                    st.success("✅ Tariff file uploaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading tariff file: {str(e)}")

            st.markdown("#### 2. Upload Claims File")
            st.markdown('<div class="upload-box">Drag and drop your claims file here</div>', unsafe_allow_html=True)
            claims_file = st.file_uploader("", type=['xlsx'], key='claims', label_visibility="collapsed")

            analysis_name = st.text_input("Analysis Name", value="FWA Analysis " + datetime.now().strftime("%Y-%m-%d"))

            if st.session_state.tariff_uploaded and claims_file:
                if st.button("🚀 Run FWA Analysis", type="primary", use_container_width=True):
                    with st.spinner("Analyzing claims for FWA patterns..."):
                        success, message = st.session_state.fwa_detector.load_data(claims_file, tariff_file)
                        if success:
                            success, message = st.session_state.fwa_detector.detect_fwa(analysis_name)
                            if success:
                                st.session_state.analysis_run = True
                                st.session_state.historical_view = False
                                st.success("✅ Analysis completed successfully!")
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                        else:
                            st.error(f"❌ {message}")

            # Debug section (only visible after analysis)
            if st.session_state.analysis_run and st.session_state.fwa_detector.flagged_claims is not None:
                with st.expander("🔧 Debug Tools"):
                    if st.button("Debug Gender Flags"):
                        debug_info = st.session_state.fwa_detector.debug_gender_flags()
                        st.text(debug_info)

    # Main content area
    if st.session_state.historical_view:
        st.markdown("### 📊 Historical Analytics")
        st.info("Historical analytics dashboard will be available after running analyses")

    elif st.session_state.tariff_uploaded:
        if st.session_state.analysis_run and hasattr(st.session_state.fwa_detector,
                                                     'flagged_claims') and st.session_state.fwa_detector.flagged_claims is not None:
            flagged_data = st.session_state.fwa_detector.flagged_claims

            # Create modern tabs with enhanced analytics
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Executive Dashboard",
                "📈 Acturial Analytics",
                "🔍 Detailed View",
                "📥 Export",
                "📋 History",
                "🎯 Insights"
            ])

            with tab1:
                st.markdown("### 📈 Key Performance Indicators")

                flagged_claims = flagged_data[flagged_data['FWA_Flag'] == True]
                total_claims = len(flagged_data)
                total_flagged = len(flagged_claims)

                col1, col2, col3, col4, col5, col6 = st.columns(6)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Claims</div>
                        <div class="metric-value">{total_claims:,}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Flagged Claims</div>
                        <div class="metric-value">{total_flagged:,}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    flagged_percentage = (total_flagged / total_claims * 100) if total_claims > 0 else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Flagged %</div>
                        <div class="metric-value">{flagged_percentage:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    total_flagged_amount = flagged_claims['TOTAL PAID'].sum()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Flagged Amount</div>
                        <div class="metric-value">${total_flagged_amount:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col5:
                    potential_recovery = flagged_claims['TOTAL PAID'].sum()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Potential Recovery</div>
                        <div class="metric-value">${potential_recovery:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col6:
                    avg_amount = flagged_data['TOTAL PAID'].mean() if 'TOTAL PAID' in flagged_data.columns else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Avg Paid Amount</div>
                        <div class="metric-value">${avg_amount:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                create_key_insights_panel(flagged_data)

                st.markdown("### 📋 Executive Summary")
                st.markdown(st.session_state.fwa_detector.generate_executive_summary())

            with tab2:
                create_Acturial_analytics_dashboard(flagged_data, st.session_state.fwa_detector.claims_data)

            with tab3:
                st.markdown("### 🔍 Detailed Claims Analysis")

                with st.expander("🔧 Filter Options", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        show_only_flagged = st.checkbox("Show only flagged claims", value=True)
                    with col2:
                        flag_type_filter = st.selectbox("Filter by flag type:",
                                                        ["All", "Fraud (F)", "Waste (W)", "Abuse (A)"])
                    with col3:
                        provider_filter = st.selectbox("Filter by provider:",
                                                       ["All"] + list(flagged_data['PROVIDER NAME'].unique()))
                    with col4:
                        issue_filter = st.selectbox("Filter by issue:",
                                                    ["All", "Gender Issues", "Duplicate Claims", "Unit Overdosing"])

                display_data = flagged_data
                if show_only_flagged:
                    display_data = display_data[display_data['FWA_Flag'] == True]

                if flag_type_filter != "All":
                    flag_char = flag_type_filter[0]
                    display_data = display_data[display_data['Flag_Type'].str.contains(flag_char, na=False)]

                if provider_filter != "All":
                    display_data = display_data[display_data['PROVIDER NAME'] == provider_filter]

                if issue_filter == "Gender Issues":
                    display_data = display_data[display_data['Flag_Reason'].str.contains('cannot claim', na=False)]
                elif issue_filter == "Duplicate Claims":
                    display_data = display_data[display_data['Flag_Reason'].str.contains('duplicate claim', na=False)]
                elif issue_filter == "Unit Overdosing":
                    display_data = display_data[display_data['Flag_Reason'].str.contains('Units claimed', na=False)]

                if len(display_data) > 0:
                    st.dataframe(
                        display_data,
                        use_container_width=True,
                        column_config={
                            "FWA_Flag": st.column_config.CheckboxColumn(
                                "Flagged",
                                help="Indicates if the claim was flagged for FWA"
                            )
                        }
                    )
                else:
                    st.warning("No claims match the current filter criteria")

            with tab4:
                st.markdown("### 📥 Export Results")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("""
                    <div class="card">
                        <h3>📊 Complete Analysis Report</h3>
                        <p>Download the complete analysis including:</p>
                        <ul>
                            <li>All flagged claims with detailed reasons</li>
                            <li>Summary statistics and metrics</li>
                            <li>Executive summary data</li>
                            <li>Acturial analytics results</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("Generate Download File", type="primary", key="export1"):
                        st.markdown(get_download_link(flagged_data, "FWA_Analysis_Results.xlsx"),
                                    unsafe_allow_html=True)

            with tab5:
                st.markdown("### 📋 Analysis History")
                st.info("Analysis history will be populated as you run more analyses")

                if 'last_analysis_id' in st.session_state and st.session_state.last_analysis_id:
                    st.markdown(f"**Current Analysis ID**: {st.session_state.last_analysis_id}")
                    st.markdown(f"**Analysis Name**: {st.session_state.analysis_name}")

            with tab6:
                st.markdown("### 🎯 Analytical Insights")

                insights_col1, insights_col2 = st.columns(2)

                with insights_col1:
                    st.markdown("##### 🎯 Top Risk Areas")

                    if 'provider_risk' in st.session_state:
                        provider_risk = st.session_state['provider_risk']
                        high_risk_count = len(provider_risk[provider_risk['risk_level'] == 'High'])
                        medium_risk_count = len(provider_risk[provider_risk['risk_level'] == 'Medium'])

                        st.metric("High-Risk Providers", high_risk_count)
                        st.metric("Medium-Risk Providers", medium_risk_count)

                    if 'TOTAL PAID' in flagged_data.columns:
                        flagged_amounts = flagged_data[flagged_data['FWA_Flag']]['TOTAL PAID']
                        if len(flagged_amounts) > 0:
                            top_5_percent = flagged_amounts.quantile(0.95)
                            high_value_count = len(flagged_amounts[flagged_amounts > top_5_percent])
                            st.metric("High-Value Flagged Claims", high_value_count)

                with insights_col2:
                    st.markdown("##### 📊 Pattern Detection")

                    insights = []

                    duplicate_count = len(flagged_data[flagged_data['Flag_Reason'].str.contains('duplicate', na=False)])
                    if duplicate_count > 0:
                        insights.append(f"🔍 **{duplicate_count} duplicate claims** detected")

                    if 'SERVICE DATE' in flagged_data.columns:
                        try:
                            flagged_dates = pd.to_datetime(flagged_data[flagged_data['FWA_Flag']]['SERVICE DATE'],
                                                           errors='coerce')
                            if len(flagged_dates) > 0:
                                day_counts = flagged_dates.dt.dayofweek.value_counts()
                                most_common_day = day_counts.idxmax() if len(day_counts) > 0 else None
                                if most_common_day is not None:
                                    day_name = calendar.day_name[most_common_day]
                                    insights.append(f"📅 **Most flagged day**: {day_name}")
                        except:
                            pass

                    if 'medication_analysis' in st.session_state:
                        medication_analysis = st.session_state['medication_analysis']
                        if 'total_medication_claims' in medication_analysis and medication_analysis[
                            'total_medication_claims'] > 0:
                            insights.append(
                                f"💊 **Medication claims**: {medication_analysis['total_medication_claims']:,} medication-related claims")
                        if 'polypharmacy_members' in medication_analysis and medication_analysis[
                            'polypharmacy_members'] > 0:
                            insights.append(
                                f"⚠️ **Polypharmacy**: {medication_analysis['polypharmacy_members']} patients on multiple medications")

                    if 'benefit_analysis' in st.session_state:
                        benefit_analysis = st.session_state['benefit_analysis']
                        if 'total_benefit_categories' in benefit_analysis:
                            insights.append(
                                f"🏥 **Benefit categories**: {benefit_analysis['total_benefit_categories']} different benefit types")

                    for insight in insights:
                        st.markdown(f"""
                        <div class="insight-card">
                            {insight}
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("##### 💡 Recommendations")

                recommendations = [
                    "**Immediate Action**: Review all high-risk provider claims flagged as Fraud (F)",
                    "**Investigation Priority**: Focus on claims exceeding $10,000 for detailed review",
                    "**Process Improvement**: Address duplicate claims through automated validation",
                    "**Monitoring**: Establish ongoing monitoring for providers with risk scores > 2.0",
                    "**Training**: Provide gender-appropriateness training to frequently flagged providers"
                ]

                # Add conditional recommendations
                if 'medication_analysis' in st.session_state:
                    recommendations.append(
                        "**Medication Review**: Audit high-volume medication prescribers for appropriateness")

                if st.session_state.get('medication_analysis', {}).get('polypharmacy_members', 0) > 0:
                    recommendations.append(
                        "**Polypharmacy Management**: Implement monitoring for patients on multiple medications")

                if 'benefit_analysis' in st.session_state:
                    recommendations.append(
                        "**Benefit Analysis**: Monitor unusual benefit patterns and provider specialization")

                for rec in recommendations:
                    st.markdown(f"• {rec}")

        else:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem;">
                <h2>🚀 Ready to Analyze Claims</h2>
                <p style="font-size: 1.1rem; color: #6c757d; max-width: 600px; margin: 0 auto 2rem;">
                    Upload your claims data and run the analysis to uncover potential fraud, waste, and abuse patterns.
                </p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📋 Acturial Analytics Capabilities", expanded=True):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown("""
                    **📊 Statistical Analysis**
                    - Distribution analysis
                    - Outlier detection
                    - Skewness & kurtosis
                    - Coefficient of variation
                    """)

                with col2:
                    st.markdown("""
                    **📈 Temporal Analysis**
                    - Daily/weekly/monthly trends
                    - Peak detection
                    - Seasonal patterns
                    - Anomaly detection
                    """)

                with col3:
                    st.markdown("""
                    **🎯 Provider Analytics**
                    - Provider risk scoring
                    - Claim volume analysis
                    - Financial impact assessment
                    - Pattern recognition
                    """)

                with col4:
                    st.markdown("""
                    **💊 Enhanced Medication Analytics**
                    - MEDICATION - CHRONIC analysis
                    - MEDICATION - ACUTE analysis  
                    - MEDICATION - OTC analysis
                    - Polypharmacy detection
                    - Duplicate medication claims
                    - Pharmaceutical fraud patterns
                    """)

                st.markdown("""
                **🏥 Benefit Pattern Analysis**
                - BASE BENEFIT DESCRIPTION analysis
                - Benefit category distribution
                - High-cost benefit identification
                - Provider benefit specialization
                - Temporal benefit trends
                """)

    else:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <h2> Welcome to FWA Detection System</h2>
            <p style="font-size: 1.1rem; color: #6c757d; max-width: 700px; margin: 0 auto 2rem;">
                Acturial analytics platform for detecting Fraud, Waste, and Abuse in healthcare claims. 
                Get started by uploading your tariff file in the sidebar.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()