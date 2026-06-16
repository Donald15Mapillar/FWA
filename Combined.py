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
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Configure the page with modern settings
st.set_page_config(
    page_title="Gen-Health Analytics Platform",
    page_icon="📊",
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
        font-size: 2.5rem;
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

    /* NEW: Gender pie chart styling */
    .gender-pie-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    .age-warning {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        font-weight: 500;
    }

    .gender-analysis-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }

    /* Premium card styling */
    .premium-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .ibnr-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FWA DETECTION SYSTEM CLASSES
# ============================================================================

class ActurialAnalytics:
    """Acturial analytical methods for FWA detection"""

    @staticmethod
    def calculate_benfords_law(data_column):
        """Calculate Benford's Law distribution for first digit analysis"""
        try:
            # Extract first digit from numeric values
            first_digits = []
            for val in data_column.dropna():
                try:
                    str_val = str(abs(float(val))).replace('.', '').replace('-', '').lstrip('0')
                    if str_val and str_val[0].isdigit():
                        first_digits.append(int(str_val[0]))
                except:
                    continue

            if not first_digits:
                return None

            # Calculate observed frequencies
            observed = pd.Series(first_digits).value_counts().sort_index()
            observed_freq = observed / observed.sum() * 100

            # Benford's Law expected frequencies (1-9 only)
            benford_expected = {d: np.log10(1 + 1 / d) * 100 for d in range(1, 10)}
            expected = pd.Series(benford_expected)

            # Calculate chi-square statistic
            chi_square = 0
            for d in expected.index:
                obs = observed_freq.get(d, 0)
                exp = expected[d]
                if exp > 0:
                    chi_square += ((obs - exp) ** 2) / exp

            return {
                'observed': observed_freq,
                'expected': expected,
                'chi_square': chi_square,
                'p_value': 1 - stats.chi2.cdf(chi_square, df=8),
                'anomaly_score': chi_square / 20  # Normalized score
            }
        except Exception as e:
            st.warning(f"Benford's Law calculation failed: {str(e)}")
            return None

    @staticmethod
    def detect_statistical_outliers(data, column='TOTAL PAID', method='iqr'):
        """Detect statistical outliers using various methods"""
        try:
            values = pd.to_numeric(data[column], errors='coerce').dropna()

            if method == 'iqr':
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = values[(values < lower_bound) | (values > upper_bound)]

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(values.fillna(values.mean())))
                outliers = values[z_scores > 3]

            elif method == 'isolation_forest':
                # Reshape for Isolation Forest
                X = values.values.reshape(-1, 1)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                preds = iso_forest.fit_predict(X_scaled)
                outliers = values[preds == -1]

            outlier_percentage = (len(outliers) / len(values)) * 100

            return {
                'outliers': outliers,
                'count': len(outliers),
                'percentage': outlier_percentage,
                'method': method,
                'mean': values.mean(),
                'std': values.std(),
                'median': values.median()
            }
        except Exception as e:
            st.warning(f"Outlier detection failed for {column}: {str(e)}")
            return None

    @staticmethod
    def calculate_claim_pattern_metrics(claims_data):
        """Calculate various claim pattern metrics"""
        try:
            metrics = {}

            # Claim frequency analysis
            if 'MEMBER NO' in claims_data.columns:
                member_claims = claims_data['MEMBER NO'].value_counts()
                metrics['claims_per_member'] = {
                    'mean': member_claims.mean(),
                    'median': member_claims.median(),
                    'max': member_claims.max(),
                    'std': member_claims.std(),
                    'top_5': member_claims.head(5).to_dict()
                }

            # Provider claim volume analysis
            if 'PROVIDER NAME' in claims_data.columns:
                provider_claims = claims_data['PROVIDER NAME'].value_counts()
                metrics['claims_per_provider'] = {
                    'mean': provider_claims.mean(),
                    'median': provider_claims.median(),
                    'max': provider_claims.max(),
                    'std': provider_claims.std(),
                    'top_5': provider_claims.head(5).to_dict()
                }

            # Amount distribution analysis
            if 'TOTAL PAID' in claims_data.columns:
                amounts = pd.to_numeric(claims_data['TOTAL PAID'], errors='coerce').dropna()
                metrics['amount_distribution'] = {
                    'mean': amounts.mean(),
                    'median': amounts.median(),
                    'max': amounts.max(),
                    'min': amounts.min(),
                    'std': amounts.std(),
                    'skewness': amounts.skew(),
                    'kurtosis': amounts.kurtosis()
                }

            # Time-based patterns
            if 'SERVICE DATE' in claims_data.columns:
                try:
                    claims_data['SERVICE_DATE_DT'] = pd.to_datetime(claims_data['SERVICE DATE'], errors='coerce')
                    day_of_week = claims_data['SERVICE_DATE_DT'].dt.dayofweek.dropna()
                    metrics['day_of_week_distribution'] = day_of_week.value_counts().sort_index().to_dict()

                    # Monthly trends
                    month = claims_data['SERVICE_DATE_DT'].dt.month.dropna()
                    metrics['monthly_distribution'] = month.value_counts().sort_index().to_dict()
                except:
                    pass

            return metrics
        except Exception as e:
            st.warning(f"Claim pattern metrics calculation failed: {str(e)}")
            return {}

    @staticmethod
    def perform_temporal_analysis(claims_data):
        """Perform temporal analysis on claims data"""
        try:
            if 'SERVICE DATE' not in claims_data.columns:
                return None

            claims_data['SERVICE_DATE_DT'] = pd.to_datetime(claims_data['SERVICE DATE'], errors='coerce')
            claims_data = claims_data.dropna(subset=['SERVICE_DATE_DT'])

            # Daily trend
            daily_trend = claims_data.groupby(claims_data['SERVICE_DATE_DT'].dt.date).size()

            # Weekly pattern
            claims_data['WEEKDAY'] = claims_data['SERVICE_DATE_DT'].dt.day_name()
            weekly_pattern = claims_data['WEEKDAY'].value_counts()

            # Monthly pattern
            claims_data['MONTH'] = claims_data['SERVICE_DATE_DT'].dt.month_name()
            monthly_pattern = claims_data['MONTH'].value_counts()

            # Calculate day-to-day changes
            daily_changes = daily_trend.diff().dropna()

            # Detect spikes
            mean = daily_trend.mean()
            std = daily_trend.std()
            spikes = daily_trend[daily_trend > mean + 2 * std]

            return {
                'daily_trend': daily_trend,
                'weekly_pattern': weekly_pattern,
                'monthly_pattern': monthly_pattern,
                'daily_changes': daily_changes,
                'spikes': spikes,
                'volatility': daily_changes.std() / daily_trend.mean() if daily_trend.mean() > 0 else 0
            }
        except Exception as e:
            st.warning(f"Temporal analysis failed: {str(e)}")
            return None

    @staticmethod
    def calculate_provider_risk_scores(claims_data, flagged_data):
        """Calculate risk scores for providers"""
        try:
            if 'PROVIDER NAME' not in claims_data.columns:
                return None

            # Calculate basic statistics per provider
            provider_stats = claims_data.groupby('PROVIDER NAME').agg({
                'TOTAL PAID': ['count', 'sum', 'mean', 'std'],
                'MEMBER NO': 'nunique'
            }).round(2)

            provider_stats.columns = ['claim_count', 'total_amount', 'avg_amount', 'amount_std', 'unique_members']

            # Calculate flagged claims per provider
            flagged_stats = flagged_data[flagged_data['FWA_Flag']].groupby('PROVIDER NAME').agg({
                'TOTAL PAID': 'sum',
                'Flag_Type': lambda x: x.str.contains('F').sum()
            })
            flagged_stats.columns = ['flagged_amount', 'fraud_count']

            # Merge and calculate risk metrics
            provider_risk = provider_stats.merge(flagged_stats, how='left', left_index=True, right_index=True)
            provider_risk = provider_risk.fillna(0)

            # Calculate risk scores
            provider_risk['flag_rate'] = (provider_risk['fraud_count'] / provider_risk['claim_count'] * 100).fillna(0)
            provider_risk['amount_flag_rate'] = (
                    provider_risk['flagged_amount'] / provider_risk['total_amount'] * 100).fillna(0)

            # Normalize scores
            for col in ['flag_rate', 'amount_flag_rate', 'avg_amount', 'amount_std']:
                if provider_risk[col].std() > 0:
                    provider_risk[f'{col}_score'] = (provider_risk[col] - provider_risk[col].mean()) / provider_risk[
                        col].std()
                else:
                    provider_risk[f'{col}_score'] = 0

            # Composite risk score
            weights = {'flag_rate_score': 0.4, 'amount_flag_rate_score': 0.3, 'avg_amount_score': 0.2,
                       'amount_std_score': 0.1}
            provider_risk['risk_score'] = sum(provider_risk[f'{k}_score'] * v for k, v in weights.items())

            # Categorize risk levels
            provider_risk['risk_level'] = pd.cut(
                provider_risk['risk_score'],
                bins=[-float('inf'), -1, 1, float('inf')],
                labels=['Low', 'Medium', 'High']
            )

            return provider_risk.sort_values('risk_score', ascending=False)
        except Exception as e:
            st.warning(f"Provider risk score calculation failed: {str(e)}")
            return None

    @staticmethod
    def analyze_benefit_patterns(claims_data):
        """Analyze benefit patterns using BASE BENEFIT DESCRIPTION column"""
        try:
            benefit_metrics = {}

            # Check for BASE BENEFIT DESCRIPTION column
            if 'BASE BENEFIT DESCRIPTION' not in claims_data.columns:
                return {"warning": "No 'BASE BENEFIT DESCRIPTION' column found in the dataset"}

            # Clean and categorize benefit descriptions
            benefit_data = claims_data.copy()
            benefit_data['BASE BENEFIT DESCRIPTION'] = benefit_data['BASE BENEFIT DESCRIPTION'].astype(str).str.strip()

            # Overall benefit distribution
            benefit_distribution = benefit_data['BASE BENEFIT DESCRIPTION'].value_counts()
            benefit_metrics['total_benefit_categories'] = len(benefit_distribution)
            benefit_metrics['benefit_distribution'] = benefit_distribution.head(20).to_dict()

            # High-level categorization
            benefit_metrics['benefit_categories'] = {
                'MEDICATION': benefit_distribution[
                    benefit_distribution.index.str.contains('MEDICATION', case=False, na=False)].sum(),
                'CONSULTATION': benefit_distribution[
                    benefit_distribution.index.str.contains('CONSULT|VISIT', case=False, na=False)].sum(),
                'PROCEDURE': benefit_distribution[
                    benefit_distribution.index.str.contains('PROCEDURE|SURGERY', case=False, na=False)].sum(),
                'DIAGNOSTIC': benefit_distribution[
                    benefit_distribution.index.str.contains('DIAGNOSTIC|TEST|LAB', case=False, na=False)].sum(),
                'HOSPITAL': benefit_distribution[
                    benefit_distribution.index.str.contains('HOSPITAL|INPATIENT', case=False, na=False)].sum(),
                'OTHER': benefit_distribution[
                    ~benefit_distribution.index.str.contains('MEDICATION|CONSULT|PROCEDURE|DIAGNOSTIC|HOSPITAL',
                                                             case=False, na=False)].sum()
            }

            # Financial analysis by benefit category
            if 'TOTAL PAID' in benefit_data.columns:
                benefit_amounts = benefit_data.groupby('BASE BENEFIT DESCRIPTION')['TOTAL PAID'].sum().sort_values(
                    ascending=False)
                benefit_metrics['top_benefits_by_cost'] = benefit_amounts.head(10).to_dict()
                benefit_metrics['total_cost_by_benefit'] = benefit_amounts.to_dict()

                # Calculate average cost per benefit
                avg_cost_by_benefit = benefit_data.groupby('BASE BENEFIT DESCRIPTION')['TOTAL PAID'].mean().sort_values(
                    ascending=False)
                benefit_metrics['avg_cost_by_benefit'] = avg_cost_by_benefit.head(10).to_dict()

            # Provider analysis by benefit category
            if 'PROVIDER NAME' in benefit_data.columns:
                # Top providers by benefit type
                provider_benefit_counts = benefit_data.groupby(
                    ['PROVIDER NAME', 'BASE BENEFIT DESCRIPTION']).size().unstack(fill_value=0)
                benefit_metrics['provider_benefit_mix'] = provider_benefit_counts.sum(axis=1).nlargest(10).to_dict()

                # Specialized providers (high concentration in one benefit)
                provider_specialization = provider_benefit_counts.max(axis=1) / provider_benefit_counts.sum(axis=1)
                benefit_metrics['top_specialized_providers'] = provider_specialization.nlargest(10).to_dict()

            # Temporal analysis by benefit
            if 'SERVICE DATE' in benefit_data.columns:
                try:
                    benefit_data['SERVICE_DATE_DT'] = pd.to_datetime(benefit_data['SERVICE DATE'], errors='coerce')
                    benefit_data_clean = benefit_data.dropna(subset=['SERVICE_DATE_DT'])

                    # Monthly trends for top benefits
                    top_benefits = benefit_distribution.head(5).index.tolist()
                    monthly_trends = {}
                    for benefit in top_benefits:
                        benefit_subset = benefit_data_clean[benefit_data_clean['BASE BENEFIT DESCRIPTION'] == benefit]
                        monthly_counts = benefit_subset.groupby(
                            benefit_subset['SERVICE_DATE_DT'].dt.to_period('M')).size()
                        monthly_trends[benefit] = monthly_counts.to_dict()

                    benefit_metrics['monthly_trends_by_benefit'] = monthly_trends
                except:
                    pass

            # Member benefit utilization patterns
            if 'MEMBER NO' in benefit_data.columns:
                member_benefit_counts = benefit_data.groupby('MEMBER NO')['BASE BENEFIT DESCRIPTION'].nunique()
                benefit_metrics['members_multiple_benefits'] = (member_benefit_counts > 1).sum()
                benefit_metrics['avg_benefits_per_member'] = member_benefit_counts.mean()
                benefit_metrics['top_members_by_benefit_variety'] = member_benefit_counts.nlargest(10).to_dict()

            # Anomaly detection in benefits
            if 'TOTAL PAID' in benefit_data.columns:
                # Identify unusually high-cost benefits
                benefit_stats = benefit_data.groupby('BASE BENEFIT DESCRIPTION')['TOTAL PAID'].agg(
                    ['mean', 'std', 'count'])
                benefit_stats['z_score'] = (benefit_stats['mean'] - benefit_stats['mean'].mean()) / benefit_stats['std']
                high_cost_anomalies = benefit_stats[benefit_stats['z_score'] > 2].index.tolist()
                benefit_metrics['potential_high_cost_anomalies'] = high_cost_anomalies

            return benefit_metrics

        except Exception as e:
            st.warning(f"Benefit pattern analysis failed: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def analyze_medication_patterns(claims_data):
        """Enhanced medication pattern analysis with detailed categorization"""
        try:
            medication_metrics = {}

            # Check for BASE BENEFIT DESCRIPTION column first (primary method)
            if 'BASE BENEFIT DESCRIPTION' not in claims_data.columns:
                # Fallback to BENEFIT NAME
                if 'BENEFIT NAME' in claims_data.columns:
                    claims_data['BASE BENEFIT DESCRIPTION'] = claims_data['BENEFIT NAME']
                else:
                    return {"warning": "No benefit description columns found for medication analysis"}

            # Identify medication claims using BASE BENEFIT DESCRIPTION
            medication_mask = claims_data['BASE BENEFIT DESCRIPTION'].astype(str).str.contains(
                'medication|drug|pharmacy|rx|prescription', case=False, na=False)
            medication_data = claims_data[medication_mask].copy()

            if len(medication_data) == 0:
                return {"warning": "No medication claims identified using benefit description"}

            medication_metrics['total_medication_claims'] = len(medication_data)
            medication_metrics['medication_percentage'] = (len(medication_data) / len(claims_data) * 100) if len(
                claims_data) > 0 else 0

            # Categorize medication types
            medication_categories = {
                'CHRONIC': claims_data['BASE BENEFIT DESCRIPTION'].astype(str).str.contains(
                    'chronic|maintenance|long-term', case=False, na=False),
                'ACUTE': claims_data['BASE BENEFIT DESCRIPTION'].astype(str).str.contains('acute|short-term|emergency',
                                                                                          case=False, na=False),
                'OTC': claims_data['BASE BENEFIT DESCRIPTION'].astype(str).str.contains(
                    'otc|over-the-counter|non-prescription', case=False, na=False)
            }

            # Analyze each medication category
            for category, mask in medication_categories.items():
                category_data = claims_data[mask].copy()
                if len(category_data) > 0:
                    medication_metrics[f'{category.lower()}_medication_claims'] = len(category_data)
                    medication_metrics[f'{category.lower()}_medication_percentage'] = (
                            len(category_data) / len(medication_data) * 100)

                    # Financial analysis for each category
                    if 'TOTAL PAID' in category_data.columns:
                        category_costs = pd.to_numeric(category_data['TOTAL PAID'], errors='coerce').dropna()
                        if len(category_costs) > 0:
                            medication_metrics[f'{category.lower()}_cost_stats'] = {
                                'total_cost': category_costs.sum(),
                                'avg_cost': category_costs.mean(),
                                'median_cost': category_costs.median(),
                                'max_cost': category_costs.max(),
                                'cost_per_claim': category_costs.sum() / len(category_data)
                            }

            # Top medication benefits by frequency
            top_medication_benefits = medication_data['BASE BENEFIT DESCRIPTION'].value_counts().head(10)
            medication_metrics['top_medication_benefits'] = top_medication_benefits.to_dict()

            # Provider analysis for medications
            if 'PROVIDER NAME' in medication_data.columns:
                # Top medication prescribers
                provider_medication_counts = medication_data['PROVIDER NAME'].value_counts()
                medication_metrics['top_medication_providers'] = provider_medication_counts.head(10).to_dict()

                # Provider medication intensity
                provider_total_claims = claims_data.groupby('PROVIDER NAME').size()
                provider_medication_ratio = (provider_medication_counts / provider_total_claims * 100).fillna(0)
                medication_metrics['provider_medication_intensity'] = provider_medication_ratio.nlargest(10).to_dict()

                # Provider specialization by medication type
                for category in ['CHRONIC', 'ACUTE', 'OTC']:
                    if f'{category.lower()}_medication_claims' in medication_metrics:
                        category_mask = medication_categories[category]
                        category_providers = claims_data[category_mask]['PROVIDER NAME'].value_counts().head(5)
                        medication_metrics[f'top_{category.lower()}_providers'] = category_providers.to_dict()

            # Member medication patterns
            if 'MEMBER NO' in medication_data.columns:
                # Members with multiple medication types
                member_medication_types = medication_data.groupby('MEMBER NO')['BASE BENEFIT DESCRIPTION'].nunique()
                medication_metrics['members_multiple_med_types'] = (member_medication_types > 1).sum()
                medication_metrics['avg_med_types_per_member'] = member_medication_types.mean()

                # High-frequency medication users
                member_medication_counts = medication_data['MEMBER NO'].value_counts()
                medication_metrics['high_frequency_med_users'] = member_medication_counts[
                    member_medication_counts > 10].count()
                medication_metrics['top_medication_users'] = member_medication_counts.head(10).to_dict()

                # Cross-medication type usage
                member_cross_usage = {}
                for member in member_medication_counts.head(20).index:
                    member_data = medication_data[medication_data['MEMBER NO'] == member]
                    chronic_count = len(member_data[
                                            member_data['BASE BENEFIT DESCRIPTION'].str.contains('chronic', case=False,
                                                                                                 na=False)])
                    acute_count = len(member_data[
                                          member_data['BASE BENEFIT DESCRIPTION'].str.contains('acute', case=False,
                                                                                               na=False)])
                    otc_count = len(
                        member_data[member_data['BASE BENEFIT DESCRIPTION'].str.contains('otc', case=False, na=False)])

                    if chronic_count > 0 or acute_count > 0 or otc_count > 0:
                        member_cross_usage[member] = {
                            'chronic': chronic_count,
                            'acute': acute_count,
                            'otc': otc_count,
                            'total': chronic_count + acute_count + otc_count
                        }

                medication_metrics['member_cross_medication_usage'] = member_cross_usage

            # Temporal analysis of medications
            if 'SERVICE DATE' in medication_data.columns:
                try:
                    medication_data['SERVICE_DATE_DT'] = pd.to_datetime(medication_data['SERVICE DATE'],
                                                                        errors='coerce')
                    medication_data_clean = medication_data.dropna(subset=['SERVICE_DATE_DT'])

                    # Monthly trends
                    monthly_counts = medication_data_clean.groupby(
                        medication_data_clean['SERVICE_DATE_DT'].dt.to_period('M')).size()
                    medication_metrics['medication_monthly_trend'] = monthly_counts.to_dict()

                    # Weekly patterns
                    medication_data_clean['WEEKDAY'] = medication_data_clean['SERVICE_DATE_DT'].dt.day_name()
                    weekday_counts = medication_data_clean['WEEKDAY'].value_counts()
                    medication_metrics['medication_weekday_pattern'] = weekday_counts.to_dict()

                    # Seasonal patterns by medication type
                    seasonal_patterns = {}
                    for category in ['CHRONIC', 'ACUTE', 'OTC']:
                        if f'{category.lower()}_medication_claims' in medication_metrics:
                            category_mask = medication_categories[category]
                            category_data_temp = claims_data[category_mask].copy()
                            category_data_temp['SERVICE_DATE_DT'] = pd.to_datetime(category_data_temp['SERVICE DATE'],
                                                                                   errors='coerce')
                            category_data_temp = category_data_temp.dropna(subset=['SERVICE_DATE_DT'])

                            monthly_category_counts = category_data_temp.groupby(
                                category_data_temp['SERVICE_DATE_DT'].dt.to_period('M')).size()
                            seasonal_patterns[category] = monthly_category_counts.to_dict()

                    medication_metrics['seasonal_patterns_by_type'] = seasonal_patterns

                except Exception as e:
                    st.warning(f"Temporal medication analysis failed: {str(e)}")

            # Cost analysis
            if 'TOTAL PAID' in medication_data.columns:
                medication_costs = pd.to_numeric(medication_data['TOTAL PAID'], errors='coerce').dropna()
                if len(medication_costs) > 0:
                    medication_metrics['medication_cost_stats'] = {
                        'total_cost': medication_costs.sum(),
                        'avg_cost': medication_costs.mean(),
                        'median_cost': medication_costs.median(),
                        'max_cost': medication_costs.max(),
                        'min_cost': medication_costs.min(),
                        'std_cost': medication_costs.std(),
                        'cost_per_member': medication_costs.sum() / medication_data[
                            'MEMBER NO'].nunique() if 'MEMBER NO' in medication_data.columns else 0,
                        'cost_per_provider': medication_costs.sum() / medication_data[
                            'PROVIDER NAME'].nunique() if 'PROVIDER NAME' in medication_data.columns else 0
                    }

                    # High-cost medication threshold (top 5%)
                    high_cost_threshold = medication_costs.quantile(0.95)
                    high_cost_medications = medication_data[
                        pd.to_numeric(medication_data['TOTAL PAID'], errors='coerce') > high_cost_threshold]
                    medication_metrics['high_cost_medication_claims'] = len(high_cost_medications)
                    medication_metrics['high_cost_medication_threshold'] = high_cost_threshold
                    medication_metrics['high_cost_medication_percentage'] = (
                            len(high_cost_medications) / len(medication_data) * 100) if len(
                        medication_data) > 0 else 0

                    # Identify top high-cost medications
                    if len(high_cost_medications) > 0:
                        high_cost_benefits = high_cost_medications.groupby('BASE BENEFIT DESCRIPTION')[
                            'TOTAL PAID'].sum().nlargest(10)
                        medication_metrics['top_high_cost_medications'] = high_cost_benefits.to_dict()

            # Duplicate medication claim detection
            duplicate_indicators = []
            if 'CLAIM NO' in medication_data.columns:
                # Check for duplicate claim numbers within medication data
                duplicate_claims = medication_data.duplicated(subset=['CLAIM NO'], keep=False)
                duplicate_counts = duplicate_claims.sum()
                medication_metrics['duplicate_medication_claims'] = duplicate_counts
                medication_metrics['duplicate_medication_percentage'] = (
                        duplicate_counts / len(medication_data) * 100) if len(medication_data) > 0 else 0

                if duplicate_counts > 0:
                    duplicate_data = medication_data[duplicate_claims]
                    duplicate_groups = duplicate_data.groupby('CLAIM NO').size()
                    medication_metrics['duplicate_groups'] = len(duplicate_groups)
                    medication_metrics['largest_duplicate_group'] = duplicate_groups.max() if len(
                        duplicate_groups) > 0 else 0

            # Polypharmacy analysis
            if 'MEMBER NO' in medication_data.columns and 'BASE BENEFIT DESCRIPTION' in medication_data.columns:
                # Analyze medication combinations per member
                member_medication_combos = medication_data.groupby('MEMBER NO')['BASE BENEFIT DESCRIPTION'].unique()
                polypharmacy_members = member_medication_combos[
                    member_medication_combos.apply(len) > 3]  # Members with 4+ different medications
                medication_metrics['polypharmacy_members'] = len(polypharmacy_members)
                medication_metrics['polypharmacy_rate'] = (
                        len(polypharmacy_members) / len(member_medication_combos) * 100) if len(
                    member_medication_combos) > 0 else 0

                # Common medication combinations
                if len(polypharmacy_members) > 0:
                    common_combinations = {}
                    for member, meds in polypharmacy_members.head(10).items():
                        common_combinations[member] = list(meds)[:5]  # Show first 5 medications
                    medication_metrics['common_polypharmacy_combinations'] = common_combinations

            # NEW: Age analysis for chronic medications
            if 'CURRENT AGE' in medication_data.columns and 'BASE BENEFIT DESCRIPTION' in medication_data.columns:
                # Identify chronic medications
                chronic_mask = medication_data['BASE BENEFIT DESCRIPTION'].astype(str).str.contains(
                    'chronic|maintenance|long-term', case=False, na=False)
                chronic_medications = medication_data[chronic_mask]

                if len(chronic_medications) > 0:
                    # Analyze ages for chronic medication claims
                    chronic_ages = pd.to_numeric(chronic_medications['CURRENT AGE'], errors='coerce').dropna()

                    if len(chronic_ages) > 0:
                        # Count patients under 17 claiming chronic medications
                        under_17_count = len(chronic_ages[chronic_ages < 17])
                        medication_metrics['chronic_under_17_count'] = under_17_count

                        if under_17_count > 0:
                            medication_metrics['chronic_under_17_percentage'] = (
                                    under_17_count / len(chronic_medications) * 100
                            )

                            # Add to FWA indicators
                            if 'fwa_indicators' not in medication_metrics:
                                medication_metrics['fwa_indicators'] = {}
                            medication_metrics['fwa_indicators']['chronic_meds_under_17'] = under_17_count

            # Potential FWA indicators
            fwa_indicators = {}

            # 1. Unusual medication patterns (e.g., same medication too frequently)
            if 'MEMBER NO' in medication_data.columns and 'SERVICE DATE' in medication_data.columns:
                try:
                    # Check for members getting same medication too frequently
                    member_med_timelines = medication_data.groupby(['MEMBER NO', 'BASE BENEFIT DESCRIPTION'])[
                        'SERVICE DATE'].count()
                    frequent_meds = member_med_timelines[
                        member_med_timelines > 5]  # More than 5 claims for same medication
                    if len(frequent_meds) > 0:
                        fwa_indicators['frequent_same_medication'] = len(frequent_meds)
                except:
                    pass

            # 2. High-cost OTC medications (potential red flag)
            if 'OTC' in [cat for cat in medication_categories.keys() if
                         f'{cat.lower()}_medication_claims' in medication_metrics]:
                otc_data = claims_data[medication_categories['OTC']]
                if 'TOTAL PAID' in otc_data.columns:
                    otc_costs = pd.to_numeric(otc_data['TOTAL PAID'], errors='coerce').dropna()
                    if len(otc_costs) > 0 and otc_costs.max() > 100:  # OTC claims over $100
                        fwa_indicators['high_cost_otc_claims'] = len(otc_costs[otc_costs > 100])

            # 3. Providers prescribing multiple medication types excessively
            if 'PROVIDER NAME' in medication_data.columns:
                provider_med_variety = medication_data.groupby('PROVIDER NAME')['BASE BENEFIT DESCRIPTION'].nunique()
                excessive_variety_providers = provider_med_variety[
                    provider_med_variety > 10]  # More than 10 different medication types
                if len(excessive_variety_providers) > 0:
                    fwa_indicators['providers_excessive_med_variety'] = len(excessive_variety_providers)

            medication_metrics['fwa_indicators'] = fwa_indicators

            return medication_metrics

        except Exception as e:
            st.warning(f"Enhanced medication pattern analysis failed: {str(e)}")
            return {"error": str(e)}


class HistoricalAnalysisManager:
    """Manages storage and retrieval of historical FWA analyses"""

    def __init__(self):
        self.data_dir = Path("./historical_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "fwa_history.db"
        self.analytics = ActurialAnalytics()
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for storing analysis history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create main analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                analysis_name TEXT,
                timestamp TEXT,
                total_claims INTEGER,
                flagged_claims INTEGER,
                flagged_percentage REAL,
                total_paid_amount REAL,
                flagged_paid_amount REAL,
                potential_recovery REAL,
                fraud_count INTEGER,
                waste_count INTEGER,
                abuse_count INTEGER,
                duplicate_count INTEGER,
                male_inappropriate_count INTEGER,
                female_inappropriate_count INTEGER,
                file_hash TEXT,
                metadata TEXT,
                analytics_metrics TEXT
            )
        ''')

        # Create provider insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS provider_insights (
                analysis_id TEXT,
                provider_name TEXT,
                flagged_count INTEGER,
                total_amount_paid REAL,
                fraud_count INTEGER,
                waste_count INTEGER,
                abuse_count INTEGER,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id),
                PRIMARY KEY (analysis_id, provider_name)
            )
        ''')

        # Create procedure insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS procedure_insights (
                analysis_id TEXT,
                clm_code TEXT,
                description TEXT,
                flagged_count INTEGER,
                fraud_count INTEGER,
                waste_count INTEGER,
                abuse_count INTEGER,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id),
                PRIMARY KEY (analysis_id, clm_code)
            )
        ''')

        # Create analytics metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics_metrics (
                analysis_id TEXT,
                metric_type TEXT,
                metric_name TEXT,
                metric_value REAL,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id)
            )
        ''')

        conn.commit()
        conn.close()

    def save_analysis(self, flagged_data, analysis_name, file_hash, metadata=None):
        """Save analysis results to database with Acturial analytics"""
        try:
            # Generate analysis ID
            timestamp = datetime.now().isoformat()
            analysis_id = self.create_analysis_id(file_hash, timestamp)

            # Calculate basic metrics
            flagged_claims = flagged_data[flagged_data['FWA_Flag'] == True]

            analysis_data = {
                'id': analysis_id,
                'analysis_name': analysis_name,
                'timestamp': timestamp,
                'total_claims': len(flagged_data),
                'flagged_claims': len(flagged_claims),
                'flagged_percentage': (len(flagged_claims) / len(flagged_data) * 100) if len(flagged_data) > 0 else 0,
                'total_paid_amount': flagged_data['TOTAL PAID'].sum(),
                'flagged_paid_amount': flagged_claims['TOTAL PAID'].sum(),
                'potential_recovery': flagged_claims['TOTAL PAID'].sum(),
                'fraud_count': len(flagged_claims[flagged_claims['Flag_Type'].str.contains('F', na=False)]),
                'waste_count': len(flagged_claims[flagged_claims['Flag_Type'].str.contains('W', na=False)]),
                'abuse_count': len(flagged_claims[flagged_claims['Flag_Type'].str.contains('A', na=False)]),
                'duplicate_count': len(
                    flagged_claims[flagged_claims['Flag_Reason'].str.contains('duplicate claim', na=False)]),
                'male_inappropriate_count': len(
                    flagged_claims[flagged_claims['Flag_Reason'].str.contains('Male cannot claim', na=False)]),
                'female_inappropriate_count': len(
                    flagged_claims[flagged_claims['Flag_Reason'].str.contains('Female cannot claim', na=False)]),
                'file_hash': file_hash,
                'metadata': json.dumps(metadata) if metadata else None,
                'analytics_metrics': json.dumps(self.calculate_analytics_metrics(flagged_data))
            }

            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert analysis data
            cursor.execute('''
                INSERT OR REPLACE INTO analyses VALUES (
                    :id, :analysis_name, :timestamp, :total_claims, :flagged_claims, :flagged_percentage,
                    :total_paid_amount, :flagged_paid_amount, :potential_recovery,
                    :fraud_count, :waste_count, :abuse_count, :duplicate_count,
                    :male_inappropriate_count, :female_inappropriate_count, :file_hash, :metadata, :analytics_metrics
                )
            ''', analysis_data)

            # Save analytics metrics
            analytics_metrics = json.loads(analysis_data['analytics_metrics'])
            for metric_type, metrics in analytics_metrics.items():
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        cursor.execute('''
                            INSERT INTO analytics_metrics VALUES (?, ?, ?, ?)
                        ''', (analysis_id, metric_type, metric_name, float(metric_value)))

            conn.commit()
            conn.close()

            return analysis_id

        except Exception as e:
            st.error(f"Error saving analysis: {str(e)}")
            return None

    def calculate_analytics_metrics(self, flagged_data):
        """Calculate Acturial analytics metrics"""
        metrics = {}

        try:
            # Basic statistical metrics
            if 'TOTAL PAID' in flagged_data.columns:
                amounts = pd.to_numeric(flagged_data['TOTAL PAID'], errors='coerce').dropna()
                metrics['statistical'] = {
                    'mean_amount': float(amounts.mean()),
                    'median_amount': float(amounts.median()),
                    'std_amount': float(amounts.std()),
                    'skewness': float(amounts.skew()),
                    'kurtosis': float(amounts.kurtosis()),
                    'cv': float(amounts.std() / amounts.mean() if amounts.mean() > 0 else 0)
                }

            # Claim pattern metrics
            pattern_metrics = self.analytics.calculate_claim_pattern_metrics(flagged_data)
            if pattern_metrics:
                metrics['pattern'] = pattern_metrics

            return metrics

        except Exception as e:
            st.warning(f"Analytics metrics calculation failed: {str(e)}")
            return {}

    def create_analysis_id(self, data_hash, timestamp):
        """Create unique ID for analysis"""
        unique_string = f"{data_hash}_{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]

    def _save_full_data_backup(self, data, analysis_id):
        """Save full data as compressed pickle file"""
        backup_path = self.data_dir / f"full_data_{analysis_id}.pkl"
        try:
            with open(backup_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            st.warning(f"Could not save full data backup: {e}")

    def get_all_analyses(self, limit=50):
        """Get list of all historical analyses"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT 
                id, analysis_name, timestamp, total_claims, flagged_claims,
                flagged_percentage, flagged_paid_amount, potential_recovery,
                fraud_count, waste_count, abuse_count
            FROM analyses 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))

        columns = [column[0] for column in cursor.description]
        analyses = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return analyses

    def get_analysis_by_id(self, analysis_id):
        """Get detailed analysis by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get main analysis data
        cursor.execute('SELECT * FROM analyses WHERE id = ?', (analysis_id,))
        columns = [column[0] for column in cursor.description]
        analysis_data = dict(zip(columns, cursor.fetchone()))

        if analysis_data.get('metadata'):
            analysis_data['metadata'] = json.loads(analysis_data['metadata'])

        # Get provider insights
        cursor.execute('''
            SELECT * FROM provider_insights 
            WHERE analysis_id = ? 
            ORDER BY flagged_count DESC 
            LIMIT 20
        ''', (analysis_id,))

        provider_columns = [column[0] for column in cursor.description]
        analysis_data['providers'] = [
            dict(zip(provider_columns, row)) for row in cursor.fetchall()
        ]

        # Get procedure insights
        cursor.execute('''
            SELECT * FROM procedure_insights 
            WHERE analysis_id = ? 
            ORDER BY flagged_count DESC 
            LIMIT 20
        ''', (analysis_id,))

        procedure_columns = [column[0] for column in cursor.description]
        analysis_data['procedures'] = [
            dict(zip(procedure_columns, row)) for row in cursor.fetchall()
        ]

        conn.close()
        return analysis_data


class FWADetector:
    def __init__(self):
        self.claims_data = None
        self.tariff_data = None
        self.flagged_claims = None
        self.history_manager = HistoricalAnalysisManager()
        self.analytics = ActurialAnalytics()

    def load_data(self, claims_file, tariff_file):
        """Load claims and tariff data from uploaded files"""
        try:
            self.claims_data = pd.read_excel(claims_file, sheet_name='Sheet1')
            self.tariff_data = pd.read_excel(tariff_file, sheet_name='Sample Tariff')

            # Debug: Show column names and sample data
            st.write("Claims data columns:", self.claims_data.columns.tolist())
            st.write("Tariff data columns:", self.tariff_data.columns.tolist())
            st.write("Sample CLM CODE values from claims:", self.claims_data['CLM CODE'].head(10).tolist())
            st.write("Sample CLM CODE values from tariff:", self.tariff_data['CLM CODE'].head(10).tolist())
            st.write("Sample Gender values:", self.claims_data['Gender'].head(10).tolist())

            # Ensure CLM CODE is string type in both datasets
            if 'CLM CODE' in self.claims_data.columns:
                self.claims_data['CLM CODE'] = self.claims_data['CLM CODE'].astype(str).str.strip()
            if 'CLM CODE' in self.tariff_data.columns:
                self.tariff_data['CLM CODE'] = self.tariff_data['CLM CODE'].astype(str).str.strip()

            return True, "Data loaded successfully"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"

    def detect_fwa(self, analysis_name="Unnamed Analysis"):
        """Main FWA detection logic"""
        if self.claims_data is None or self.tariff_data is None:
            return False, "Please load data first"

        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            self.flagged_claims = self.claims_data.copy()
            self.flagged_claims['FWA_Flag'] = False
            self.flagged_claims['Flag_Reason'] = ''
            self.flagged_claims['Flag_Type'] = ''

            # Existing detection logic
            status_text.text("Running basic FWA checks...")
            self._run_basic_checks()
            progress_bar.progress(40)

            status_text.text("Performing Acturial analytics...")
            self._perform_Acturial_analytics()
            progress_bar.progress(80)

            status_text.text("Saving analysis to history...")

            # Generate file hash for uniqueness
            file_hash = hashlib.md5(
                pd.util.hash_pandas_object(self.claims_data).values.tobytes()
            ).hexdigest()[:16]

            # Prepare metadata
            metadata = {
                'source_file': 'uploaded_file.xlsx',
                'analysis_date': datetime.now().isoformat(),
                'user_notes': '',
                'detection_rules': ['gender', 'age', 'duplicate', 'unit_overdose', 'Acturial_analytics',
                                    'age_chronic_meds']
            }

            # Save to history
            analysis_id = self.history_manager.save_analysis(
                self.flagged_claims,
                analysis_name,
                file_hash,
                metadata
            )

            if analysis_id:
                st.session_state['last_analysis_id'] = analysis_id
                st.session_state['analysis_name'] = analysis_name

            status_text.text("Analysis complete!")
            time.sleep(0.5)
            status_text.empty()

            return True, "FWA analysis completed successfully"

        except Exception as e:
            return False, f"Error during FWA analysis: {str(e)}"

    def _run_basic_checks(self):
        """Run all basic FWA checks"""
        self._check_gender_appropriateness()
        self._check_male_inappropriate_codes()
        self._check_female_inappropriate_codes()
        self._check_age_appropriateness()
        self._check_unit_overdosing()
        self._check_duplicate_claims()
        self._check_age_for_chronic_medications()  # NEW: Added age check for chronic medications

    def _check_age_for_chronic_medications(self):
        """NEW: Check if patients under 17 are claiming chronic medications"""
        try:
            age_flags = 0

            # First, identify chronic medication claims using BASE BENEFIT DESCRIPTION
            chronic_med_mask = self.flagged_claims['BASE BENEFIT DESCRIPTION'].astype(str).str.contains(
                'chronic|maintenance|long-term', case=False, na=False)

            # Check age for each chronic medication claim
            for idx, claim in self.flagged_claims[chronic_med_mask].iterrows():
                current_age = claim.get('CURRENT AGE')

                # Skip if age is missing or not a number
                if pd.isna(current_age):
                    continue

                try:
                    current_age = float(current_age)

                    # Flag if patient is under 17 and claiming chronic medication
                    if current_age < 17:
                        self._add_flag(
                            idx,
                            f"Age inappropriate: Patient age {current_age} under 17 cannot claim Chronic Medications",
                            "F"
                        )
                        age_flags += 1

                except (ValueError, TypeError):
                    continue

            st.write(
                f"Age restriction for chronic medications check: Found {age_flags} age-inappropriate chronic medication claims")

        except Exception as e:
            st.warning(f"Age check for chronic medications failed: {str(e)}")

    def _perform_Acturial_analytics(self):
        """Perform Acturial analytical checks"""
        # Statistical outlier detection moved to analytics dashboard

        # Benford's Law analysis for amounts
        if 'TOTAL PAID' in self.flagged_claims.columns:
            benford_result = self.analytics.calculate_benfords_law(self.flagged_claims['TOTAL PAID'])
            if benford_result:
                st.session_state['benford_result'] = benford_result

        # Provider risk scoring
        provider_risk = self.analytics.calculate_provider_risk_scores(
            self.claims_data,
            self.flagged_claims
        )

        if provider_risk is not None:
            st.session_state['provider_risk'] = provider_risk

            # Flag high-risk providers
            high_risk_providers = provider_risk[provider_risk['risk_level'] == 'High']
            for provider in high_risk_providers.index:
                provider_indices = self.flagged_claims[self.flagged_claims['PROVIDER NAME'] == provider].index
                for idx in provider_indices:
                    self._add_flag(
                        idx,
                        f"High-risk provider: {provider} (risk score: {provider_risk.loc[provider, 'risk_score']:.2f})",
                        "F"
                    )

        # Enhanced medication pattern analysis
        medication_analysis = self.analytics.analyze_medication_patterns(self.claims_data)
        if medication_analysis:
            st.session_state['medication_analysis'] = medication_analysis

        # Benefit pattern analysis
        benefit_analysis = self.analytics.analyze_benefit_patterns(self.claims_data)
        if benefit_analysis:
            st.session_state['benefit_analysis'] = benefit_analysis

    def _check_gender_appropriateness(self):
        """Check if service is appropriate for patient's gender based on tariff"""
        # Debug info
        gender_flags = 0

        for idx, claim in self.flagged_claims.iterrows():
            clm_code = str(claim['CLM CODE']).strip()
            patient_gender = claim['Gender']

            # Convert tariff CLM CODE to string for comparison
            tariff_match = self.tariff_data[self.tariff_data['CLM CODE'].astype(str).str.strip() == clm_code]

            if not tariff_match.empty:
                allowed_gender = str(tariff_match.iloc[0]['AllowedGender']).strip().upper()

                if allowed_gender == 'B':
                    continue
                elif allowed_gender == 'F' and patient_gender != 'Female':
                    self._add_flag(idx, f"Gender inappropriate: Male patient for female-only procedure code {clm_code}",
                                   "F")
                    gender_flags += 1
                elif allowed_gender == 'M' and patient_gender != 'Male':
                    self._add_flag(idx, f"Gender inappropriate: Female patient for male-only procedure code {clm_code}",
                                   "F")
                    gender_flags += 1

        # Debug output
        st.write(f"Gender appropriateness check: Found {gender_flags} gender-inappropriate claims")

    def _check_male_inappropriate_codes(self):
        """Check if male patients are claiming female-only procedure codes"""
        # Define female-only procedure codes as STRINGS
        male_inappropriate_codes = [
            '19121', '52270', '52285', '53210', '53430', '53660', '53665', '55980', '0',
            '56420', '56440', '56441', '56500', '56505', '56510', '56602', '56620',
            '56625', '56641', '56700', '56740', '56750', '57000', '57010', '57100',
            '57105', '57120', '57130', '57210', '57240', '57245', '57250', '57256',
            '57260', '57262', '57265', '57267', '57280', '57288', '57290', '57300',
            '57320', '57410', '57452', '57500', '57510', '57521', '57525', '57530',
            '57540', '57550', '57560', '57600', '57700', '57701', '57720', '58101',
            '58120', '58130', '58140', '58145', '58146', '58150', '58155', '58210',
            '58260', '58265', '58270', '58300', '58301', '58320', '58340', '58350',
            '58351', '58400', '58410', '58500', '58520', '58540', '58600', '58700',
            '58720', '58741', '58743', '58744', '58830', '58840', '58841', '58900',
            '58940', '58945', '58946', '58980', '58984', '58986', '58987', '58990',
            '58993', '58994', '59000', '59050', '59230', '59231', '59232', '59401',
            '59435', '59438', '59439', '59440', '59441', '59443', '59444', '59445',
            '59446', '59455', '59475', '59476', '59477', '59478', '59479', '59483',
            '59484', '59489', '59490', '59492', '59493', '59494', '59495', '59496',
            '59497', '59498', '59499', '59500', '59502', '59503', '59550', '59560',
            '59562', '59861', '59862', '59865'
        ]

        male_flags = 0
        for idx, claim in self.flagged_claims.iterrows():
            clm_code = str(claim['CLM CODE']).strip()
            patient_gender = claim['Gender']

            # Remove leading zeros for comparison
            clm_code_clean = clm_code.lstrip('0')

            # Check if code is in the list (with or without leading zeros)
            if patient_gender == 'Male' and (
                    clm_code in male_inappropriate_codes or clm_code_clean in male_inappropriate_codes):
                self._add_flag(idx, f"Gender inappropriate: Male patient for female-only procedure code {clm_code}",
                               "F")
                male_flags += 1

        # Debug output
        st.write(f"Male inappropriate codes check: Found {male_flags} male-inappropriate claims")

    def _check_female_inappropriate_codes(self):
        """Check if female patients are claiming male-only procedure codes"""
        # Define male-only procedure codes as STRINGS
        female_inappropriate_codes = [
            '19140', '52601', '52610', '52630', '52700', '53215', '53410', '53418',
            '53420', '53425', '53440', '53505', '53515', '53600', '53605', '53620',
            '0', '54000', '54001', '54002', '54015', '54050', '54055', '54060',
            '54065', '54100', '54105', '54110', '54115', '54120', '54125', '54130',
            '54135', '54150', '54160', '54161', '54162', '54165', '54200', '54205',
            '54220', '54300', '54305', '54320', '54325', '54330', '54380', '54385',
            '54390', '54400', '54420', '54430', '54440', '54500', '54505', '54506',
            '54510', '54520', '54530', '54535', '54550', '54560', '54600', '54620',
            '54640', '54645', '54660', '54670', '54680', '54700', '54800', '54820',
            '54830', '54840', '54860', '54861', '54900', '54901', '55000', '55040',
            '55060', '55100', '55120', '55150', '55170', '55200', '55250', '55300',
            '55400', '55450', '55500', '55520', '55530', '55535', '55540', '55600',
            '55605', '55650', '55680', '55700', '55705', '55720', '55725', '55740',
            '55801', '55810', '55821', '55831', '55840', '55845', '55970'
        ]

        female_flags = 0
        for idx, claim in self.flagged_claims.iterrows():
            clm_code = str(claim['CLM CODE']).strip()
            patient_gender = claim['Gender']

            # Remove leading zeros for comparison
            clm_code_clean = clm_code.lstrip('0')

            # Check if code is in the list (with or without leading zeros)
            if patient_gender == 'Female' and (
                    clm_code in female_inappropriate_codes or clm_code_clean in female_inappropriate_codes):
                self._add_flag(idx, f"Gender inappropriate: Female patient for male-only procedure code {clm_code}",
                               "F")
                female_flags += 1

        # Debug output
        st.write(f"Female inappropriate codes check: Found {female_flags} female-inappropriate claims")

    def _check_age_appropriateness(self):
        """Check if service is appropriate for patient's age"""
        age_flags = 0
        for idx, claim in self.flagged_claims.iterrows():
            current_age = claim['CURRENT AGE']
            clm_code = str(claim['CLM CODE']).strip()

            if current_age < 18 and clm_code in ['15831', '15832', '15833']:
                self._add_flag(idx, "Age inappropriate for cosmetic procedure", "F")
                age_flags += 1
            elif current_age > 65 and clm_code in ['15775']:
                self._add_flag(idx, "Questionable procedure for age group", "W")
                age_flags += 1

        st.write(f"Age appropriateness check: Found {age_flags} age-inappropriate claims")

    def _check_unit_overdosing(self):
        """Check if units claimed exceed maximum allowed units"""
        unit_flags = 0
        for idx, claim in self.flagged_claims.iterrows():
            clm_code = str(claim['CLM CODE']).strip()
            units_claimed = claim['UNITS']

            tariff_match = self.tariff_data[self.tariff_data['CLM CODE'].astype(str).str.strip() == clm_code]

            if not tariff_match.empty:
                max_units = tariff_match.iloc[0]['MaxUnits']

                if pd.notna(max_units) and pd.notna(units_claimed):
                    try:
                        max_units = float(max_units)
                        units_claimed = float(units_claimed)
                        if units_claimed > max_units:
                            self._add_flag(idx, f"Units claimed ({units_claimed}) exceed maximum allowed ({max_units})",
                                           "W")
                            unit_flags += 1
                    except (ValueError, TypeError):
                        pass

        st.write(f"Unit overdosing check: Found {unit_flags} unit overdose claims")

    def _check_duplicate_claims(self):
        """Check for duplicate claims based on CLAIM NO and CLAIM LINE NO"""
        duplicate_flags = 0

        if 'CLAIM NO' not in self.flagged_claims.columns:
            st.warning("CLAIM NO column not found. Skipping duplicate detection.")
            return

        if 'CLAIM LINE NO' not in self.flagged_claims.columns:
            st.warning("CLAIM LINE NO column not found. Using only CLAIM NO for duplicate detection.")
            duplicate_mask = self.flagged_claims.duplicated(
                subset=['CLAIM NO'],
                keep=False
            )

            for idx in self.flagged_claims[duplicate_mask].index:
                self._add_flag(idx, "Duplicate claim - same claim number", "F")
                duplicate_flags += 1
            return

        duplicate_mask = self.flagged_claims.duplicated(
            subset=['CLAIM NO', 'CLAIM LINE NO'],
            keep=False
        )

        for idx in self.flagged_claims[duplicate_mask].index:
            self._add_flag(idx, "Exact duplicate claim - same claim number and claim line number", "F")
            duplicate_flags += 1

        st.write(f"Duplicate check: Found {duplicate_flags} duplicate claims")

    def _add_flag(self, index, reason, flag_type):
        """Add FWA flag to claim"""
        self.flagged_claims.at[index, 'FWA_Flag'] = True
        existing_reason = self.flagged_claims.at[index, 'Flag_Reason']

        if existing_reason:
            self.flagged_claims.at[index, 'Flag_Reason'] = existing_reason + "; " + reason
        else:
            self.flagged_claims.at[index, 'Flag_Reason'] = reason

        current_types = self.flagged_claims.at[index, 'Flag_Type']
        if flag_type not in str(current_types):
            if current_types:
                self.flagged_claims.at[index, 'Flag_Type'] = current_types + "," + flag_type
            else:
                self.flagged_claims.at[index, 'Flag_Type'] = flag_type

    def generate_executive_summary(self):
        """Generate executive summary of FWA findings"""
        if self.flagged_claims is None:
            return "No analysis performed yet"

        total_claims = len(self.flagged_claims)
        flagged_claims = self.flagged_claims[self.flagged_claims['FWA_Flag'] == True]
        total_flagged = len(flagged_claims)

        total_paid_amount = self.flagged_claims['TOTAL PAID'].sum()
        flagged_paid_amount = flagged_claims['TOTAL PAID'].sum()

        fraud_flags = flagged_claims[flagged_claims['Flag_Type'].str.contains('F', na=False)]
        waste_flags = flagged_claims[flagged_claims['Flag_Type'].str.contains('W', na=False)]
        abuse_flags = flagged_claims[flagged_claims['Flag_Type'].str.contains('A', na=False)]

        potential_recovery = flagged_claims['TOTAL PAID'].sum()

        male_inappropriate_flags = flagged_claims[
            flagged_claims['Flag_Reason'].str.contains('Male cannot claim', na=False)]
        female_inappropriate_flags = flagged_claims[
            flagged_claims['Flag_Reason'].str.contains('Female cannot claim', na=False)]

        duplicate_flags = flagged_claims[flagged_claims['Flag_Reason'].str.contains('duplicate claim', na=False)]

        # NEW: Count age-inappropriate chronic medication claims
        chronic_age_flags = flagged_claims[
            flagged_claims['Flag_Reason'].str.contains('under 17 cannot claim Chronic Medications', na=False)
        ]
        chronic_age_count = len(chronic_age_flags)

        summary = f"""
        # FWA Detection Executive Summary

        ## Overview
        - **Total Claims Analyzed**: {total_claims:,}
        - **Flagged Claims**: {total_flagged:,} ({total_flagged / total_claims * 100:.1f}%)
        - **Total Paid Amount**: ${total_paid_amount:,.2f}
        - **Flagged Paid Amount**: ${flagged_paid_amount:,.2f} ({flagged_paid_amount / total_paid_amount * 100:.1f}%)
        - **Potential Recovery**: ${potential_recovery:,.2f}

        ## Breakdown by FWA Category
        - **Fraud (F)**: {len(fraud_flags):,} claims
        - **Waste (W)**: {len(waste_flags):,} claims  
        - **Abuse (A)**: {len(abuse_flags):,} claims

        ## Duplicate Claims
        - **Exact Duplicates**: {len(duplicate_flags):,} claims (same claim & line numbers)

        ## Gender-Based Inappropriate Claims
        - **Male claiming female-only procedures**: {len(male_inappropriate_flags):,} claims
        - **Female claiming male-only procedures**: {len(female_inappropriate_flags):,} claims

        ## Age-Based Inappropriate Claims
        - **Patients under 17 claiming chronic medications**: {chronic_age_count:,} claims

        ## Key Findings
        {self._generate_key_findings(flagged_claims)}

        ## Recommendations
        1. **Immediate Action**: Review all Fraud (F) flagged claims for potential recovery
        2. **Duplicate Resolution**: Address exact duplicate claims with same claim and line numbers
        3. **Gender Compliance**: Address gender-inappropriate claims through provider education
        4. **Age Compliance**: Review claims from patients under 17 for chronic medications
        5. **Process Improvement**: Address Waste (W) patterns through billing guidelines
        6. **Continuous Monitoring**: Establish ongoing monitoring for high-risk providers and procedures
        7. **Medication Review**: Audit medication claims, especially high-cost and duplicate prescriptions
        8. **Benefit Analysis**: Monitor unusual benefit patterns and provider specialization
        """

        return summary

    def _generate_key_findings(self, flagged_claims):
        """Generate key findings from flagged claims"""
        if len(flagged_claims) == 0:
            return "No significant findings detected."

        findings = []

        top_providers = flagged_claims['PROVIDER NAME'].value_counts().head(3)
        if len(top_providers) > 0:
            findings.append(
                f"- **Top flagged providers**: {', '.join([f'{provider} ({count} claims)' for provider, count in top_providers.items()])}")

        common_codes = flagged_claims['CLM CODE'].value_counts().head(3)
        if len(common_codes) > 0:
            code_descriptions = []
            for code, count in common_codes.items():
                desc_match = self.tariff_data[self.tariff_data['CLM CODE'] == code]
                if not desc_match.empty:
                    desc = desc_match.iloc[0]['CODE DESCRIPTION']
                    code_descriptions.append(f"Code {code} ({desc}): {count} claims")
                else:
                    code_descriptions.append(f"Code {code}: {count} claims")
            findings.append(f"- **Most common flagged procedures**: {', '.join(code_descriptions)}")

        common_reasons = flagged_claims['Flag_Reason'].value_counts().head(5)
        if len(common_reasons) > 0:
            reasons_list = []
            for reason, count in common_reasons.items():
                short_reason = reason.split(';')[0][:50] + '...' if len(reason) > 50 else reason.split(';')[0]
                reasons_list.append(f"{short_reason} ({count})")
            findings.append(f"- **Common flag reasons**: {', '.join(reasons_list)}")

        financial_by_type = flagged_claims.groupby('Flag_Type')['TOTAL PAID'].sum()
        if len(financial_by_type) > 0:
            financial_summary = []
            for flag_type, amount in financial_by_type.items():
                financial_summary.append(f"{flag_type}: ${amount:,.2f}")
            findings.append(f"- **Financial impact by type**: {', '.join(financial_summary)}")

        male_claims = flagged_claims[flagged_claims['Flag_Reason'].str.contains('Male cannot claim', na=False)]
        female_claims = flagged_claims[flagged_claims['Flag_Reason'].str.contains('Female cannot claim', na=False)]

        if len(male_claims) > 0:
            findings.append(
                f"- **Male inappropriate claims**: {len(male_claims)} claims totaling ${male_claims['TOTAL PAID'].sum():,.2f}")
        if len(female_claims) > 0:
            findings.append(
                f"- **Female inappropriate claims**: {len(female_claims)} claims totaling ${female_claims['TOTAL PAID'].sum():,.2f}")

        duplicate_claims = flagged_claims[flagged_claims['Flag_Reason'].str.contains('duplicate claim', na=False)]
        if len(duplicate_claims) > 0:
            findings.append(f"- **Exact duplicates**: {len(duplicate_claims)} claims with same claim and line numbers")

        # Add chronic medication age violations
        chronic_age_violations = flagged_claims[
            flagged_claims['Flag_Reason'].str.contains('under 17 cannot claim Chronic Medications', na=False)
        ]
        if len(chronic_age_violations) > 0:
            findings.append(
                f"- **Age violations for chronic medications**: {len(chronic_age_violations)} claims from patients under 17")

        # Add medication findings if available
        if 'medication_analysis' in st.session_state:
            med_analysis = st.session_state['medication_analysis']
            if 'total_medication_claims' in med_analysis:
                findings.append(
                    f"- **Medication claims**: {med_analysis['total_medication_claims']:,} total medication claims")

            if 'chronic_medication_claims' in med_analysis:
                findings.append(
                    f"- **Chronic medication claims**: {med_analysis['chronic_medication_claims']:,} claims")

            if 'acute_medication_claims' in med_analysis:
                findings.append(f"- **Acute medication claims**: {med_analysis['acute_medication_claims']:,} claims")

            if 'otc_medication_claims' in med_analysis:
                findings.append(f"- **OTC medication claims**: {med_analysis['otc_medication_claims']:,} claims")

            if 'duplicate_medication_claims' in med_analysis and med_analysis['duplicate_medication_claims'] > 0:
                findings.append(
                    f"- **Duplicate medication claims**: {med_analysis['duplicate_medication_claims']:,} potential duplicates")

        # Add benefit findings if available
        if 'benefit_analysis' in st.session_state:
            benefit_analysis = st.session_state['benefit_analysis']
            if 'total_benefit_categories' in benefit_analysis:
                findings.append(
                    f"- **Benefit categories**: {benefit_analysis['total_benefit_categories']} different benefit types")

            if 'benefit_categories' in benefit_analysis:
                med_count = benefit_analysis['benefit_categories'].get('MEDICATION', 0)
                if med_count > 0:
                    findings.append(f"- **Medication benefits**: {med_count:,} claims categorized as medication")

        return "\n".join(findings)


# ============================================================================
# ACTUARIAL DASHBOARD FUNCTIONS
# ============================================================================

# Helper functions for Actuarial Dashboard
def clean_column_names(df):
    """Clean and standardize column names"""
    cleaned_columns = []
    for col in df.columns:
        col_str = str(col)
        cleaned = col_str.strip().upper().replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')
        cleaned_columns.append(cleaned)
    df.columns = cleaned_columns
    return df


def parse_claims_file_actuarial(file):
    """Parse uploaded claims Excel file for actuarial dashboard"""
    try:
        xls = pd.ExcelFile(file)
        all_data = []
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                df = clean_column_names(df)
                df['source_sheet'] = sheet_name
                all_data.append(df)
            except Exception as e:
                st.warning(f"Warning: Could not read sheet '{sheet_name}': {str(e)}")
                continue

        if not all_data:
            st.error("No data could be read from the Excel file")
            return None

        df = pd.concat(all_data, ignore_index=True, sort=False)

        # Standardize column names
        column_mapping = {
            'MEMBERNO': 'MEMBER_NO',
            'MEMBER_NUMBER': 'MEMBER_NO',
            'CLAIM_AMOUNT': 'AMOUNT_CLAIMED',
            'AMOUNTCLAIMED': 'AMOUNT_CLAIMED',
            'TOTALPAID': 'TOTAL_PAID',
            'PAIDAMOUNT': 'TOTAL_PAID',
            'SERVICEDATE': 'SERVICE_DATE',
            'DATE_OF_SERVICE': 'SERVICE_DATE',
            'PROVIDER': 'PROVIDER_NAME',
            'PROVIDERNAME': 'PROVIDER_NAME',
            'AGE': 'CURRENT_AGE',
            'PATIENT_AGE': 'CURRENT_AGE',
            'GENDER': 'GENDER',
            'SEX': 'GENDER'
        }

        df.columns = [column_mapping.get(col, col) for col in df.columns]

        # Ensure required columns exist
        required_columns = ['MEMBER_NO', 'AMOUNT_CLAIMED', 'TOTAL_PAID']
        for col in required_columns:
            if col not in df.columns:
                st.warning(f"Column '{col}' not found in data. Creating placeholder.")
                df[col] = 0

        # Convert numeric columns
        numeric_columns = ['AMOUNT_CLAIMED', 'TOTAL_PAID', 'CURRENT_AGE']
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                except:
                    df[col] = 0

        # Convert date columns - improved handling
        date_columns = ['SERVICE_DATE']
        for col in date_columns:
            if col in df.columns:
                try:
                    # First, convert to string to handle mixed formats
                    df[col] = df[col].astype(str)
                    # Then parse with flexible format
                    df[col] = pd.to_datetime(df[col], errors='coerce', format='mixed')

                    # Check for parsing issues
                    invalid_dates = df[col].isna().sum()
                    if invalid_dates > 0:
                        st.warning(
                            f"Could not parse {invalid_dates} dates in column '{col}'. These records will have null dates.")

                except Exception as e:
                    st.warning(f"Could not parse dates in column '{col}': {str(e)}")
                    # Try alternative parsing method
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        st.warning(f"Alternative parsing also failed for column '{col}'")
                        df[col] = pd.NaT

        return df

    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        return None


def parse_premium_file_actuarial(file):
    """Parse uploaded premium Excel file"""
    try:
        xls = pd.ExcelFile(file)
        df = pd.read_excel(xls)
        df = clean_column_names(df)

        # Map column names based on the sample file structure
        column_mapping = {
            'MEMBER': 'MEMBER_NO',
            'MEMBERNO': 'MEMBER_NO',
            'NAME': 'MEMBER_NAME',
            'DEPENDANT_TYPE': 'DEPENDANT_TYPE',
            'RELATIONSHIP': 'RELATIONSHIP',
            'ID_NUMBER': 'ID_NUMBER',
            'EMPLOYEE_NUMBER': 'EMPLOYEE_NUMBER',
            'BIRTHDATE': 'BIRTH_DATE',
            'BIRTH_DATE': 'BIRTH_DATE',
            'DATE_OF_BIRTH': 'BIRTH_DATE',
            'DOB': 'BIRTH_DATE',
            'PAYER_NO': 'PAYER_NO',
            'PAYER_NAME': 'PAYER_NAME',
            'JOIN_DATE': 'JOIN_DATE',
            'EFFECTIVE_DATE': 'JOIN_DATE',
            'TERMINATION_DATE': 'TERMINATION_DATE',
            'END_DATE': 'TERMINATION_DATE',
            'PRODUCT_NAME': 'PRODUCT_NAME',
            'PLAN': 'PRODUCT_NAME',
            'PREMIUM': 'PREMIUM_AMOUNT',
            'PREMIUM_AMOUNT': 'PREMIUM_AMOUNT',
            'AMOUNT': 'PREMIUM_AMOUNT'
        }

        df.columns = [column_mapping.get(col, col) for col in df.columns]

        # Ensure required columns exist
        if 'MEMBER_NO' not in df.columns:
            st.error("Premium file must contain 'MEMBER' or 'MEMBER_NO' column")
            return None

        if 'PREMIUM_AMOUNT' not in df.columns:
            # Try to find premium column
            for col in df.columns:
                if 'PREMIUM' in col:
                    df = df.rename(columns={col: 'PREMIUM_AMOUNT'})
                    break
            if 'PREMIUM_AMOUNT' not in df.columns:
                st.error("Premium file must contain premium amount column")
                return None

        # Convert numeric columns
        if 'PREMIUM_AMOUNT' in df.columns:
            df['PREMIUM_AMOUNT'] = pd.to_numeric(df['PREMIUM_AMOUNT'], errors='coerce').fillna(0)

        # Convert date columns
        date_columns = ['BIRTH_DATE', 'JOIN_DATE', 'TERMINATION_DATE']
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    st.warning(f"Could not parse dates in column '{col}'")

        # Calculate age if birth date exists
        if 'BIRTH_DATE' in df.columns:
            df['AGE'] = ((datetime.now() - df['BIRTH_DATE']).dt.days / 365.25).astype(int)

        # Add calculated columns
        df['ACTIVE'] = df['TERMINATION_DATE'].isna() | (df['TERMINATION_DATE'] > datetime.now())

        return df

    except Exception as e:
        st.error(f"Error parsing premium file: {str(e)}")
        return None


def calculate_premium_metrics_actuarial(df):
    """Calculate premium metrics for actuarial dashboard"""
    metrics = {}

    try:
        # Basic statistics
        metrics['total_members'] = len(df)
        metrics['active_members'] = df['ACTIVE'].sum() if 'ACTIVE' in df.columns else len(df)

        if 'MEMBER_NO' in df.columns:
            metrics['unique_members'] = df['MEMBER_NO'].nunique()
        else:
            metrics['unique_members'] = 0

        # Financial metrics
        if 'PREMIUM_AMOUNT' in df.columns:
            metrics['total_premium'] = float(df['PREMIUM_AMOUNT'].sum())
            metrics['avg_premium'] = float(df['PREMIUM_AMOUNT'].mean())
            metrics['max_premium'] = float(df['PREMIUM_AMOUNT'].max())
            metrics['min_premium'] = float(df['PREMIUM_AMOUNT'].min())
        else:
            metrics['total_premium'] = 0
            metrics['avg_premium'] = 0
            metrics['max_premium'] = 0
            metrics['min_premium'] = 0

        # Product analysis
        if 'PRODUCT_NAME' in df.columns and 'PREMIUM_AMOUNT' in df.columns:
            try:
                product_stats = df.groupby('PRODUCT_NAME').agg({
                    'PREMIUM_AMOUNT': 'sum',
                    'MEMBER_NO': 'nunique'
                }).reset_index()
                product_stats = product_stats.sort_values('PREMIUM_AMOUNT', ascending=False)
                metrics['product_analysis'] = product_stats
            except:
                metrics['product_analysis'] = pd.DataFrame()

        # Payer analysis
        if 'PAYER_NAME' in df.columns and 'PREMIUM_AMOUNT' in df.columns:
            try:
                payer_stats = df.groupby('PAYER_NAME').agg({
                    'PREMIUM_AMOUNT': 'sum',
                    'MEMBER_NO': 'nunique'
                }).reset_index()
                payer_stats = payer_stats.sort_values('PREMIUM_AMOUNT', ascending=False)
                metrics['payer_analysis'] = payer_stats
            except:
                metrics['payer_analysis'] = pd.DataFrame()

        # Age distribution
        if 'AGE' in df.columns:
            try:
                df['age_group'] = pd.cut(
                    df['AGE'],
                    bins=[0, 18, 30, 40, 50, 60, 70, 100],
                    labels=['0-18', '19-30', '31-40', '41-50', '51-60', '61-70', '71+'],
                    include_lowest=True
                )
                age_dist = df.groupby('age_group').agg({
                    'MEMBER_NO': 'nunique',
                    'PREMIUM_AMOUNT': 'sum'
                }).reset_index()
                age_dist = age_dist.rename(columns={'MEMBER_NO': 'member_count'})
                metrics['age_distribution'] = age_dist
            except:
                metrics['age_distribution'] = pd.DataFrame()

        # Join date analysis (monthly trends)
        if 'JOIN_DATE' in df.columns and 'PREMIUM_AMOUNT' in df.columns:
            try:
                df['join_month'] = df['JOIN_DATE'].dt.to_period('M')
                monthly_stats = df.groupby('join_month').agg({
                    'PREMIUM_AMOUNT': 'sum',
                    'MEMBER_NO': 'nunique'
                }).reset_index()
                monthly_stats['join_month'] = monthly_stats['join_month'].astype(str)
                metrics['monthly_trends'] = monthly_stats
            except:
                metrics['monthly_trends'] = pd.DataFrame()

        # Termination analysis
        if 'TERMINATION_DATE' in df.columns:
            try:
                terminated = df[df['TERMINATION_DATE'].notna()]
                metrics['terminated_members'] = len(terminated)
                if len(df) > 0:
                    metrics['termination_rate'] = len(terminated) / len(df)
                else:
                    metrics['termination_rate'] = 0
            except:
                metrics['terminated_members'] = 0
                metrics['termination_rate'] = 0

        return metrics

    except Exception as e:
        st.error(f"Error calculating premium metrics: {str(e)}")
        return metrics


def calculate_claims_metrics_actuarial(df):
    """Calculate claims metrics for actuarial dashboard"""
    metrics = {}

    try:
        # Basic statistics
        metrics['total_claims'] = len(df)

        if 'MEMBER_NO' in df.columns:
            metrics['unique_members'] = df['MEMBER_NO'].nunique()
        else:
            metrics['unique_members'] = 0

        if 'PROVIDER_NAME' in df.columns:
            metrics['unique_providers'] = df['PROVIDER_NAME'].nunique()
        else:
            metrics['unique_providers'] = 0

        # Financial metrics
        if 'AMOUNT_CLAIMED' in df.columns:
            metrics['total_claimed'] = float(df['AMOUNT_CLAIMED'].sum())
            metrics['avg_claim_amount'] = float(df['AMOUNT_CLAIMED'].mean())
            metrics['max_claim'] = float(df['AMOUNT_CLAIMED'].max())
            metrics['min_claim'] = float(df['AMOUNT_CLAIMED'].min())
        else:
            metrics['total_claimed'] = 0
            metrics['avg_claim_amount'] = 0
            metrics['max_claim'] = 0
            metrics['min_claim'] = 0

        if 'TOTAL_PAID' in df.columns:
            metrics['total_paid'] = float(df['TOTAL_PAID'].sum())
            if metrics['total_claimed'] > 0:
                metrics['paid_ratio'] = float(metrics['total_paid'] / metrics['total_claimed'])
            else:
                metrics['paid_ratio'] = 0
        else:
            metrics['total_paid'] = 0
            metrics['paid_ratio'] = 0

        # Claim frequency by provider
        if 'PROVIDER_NAME' in df.columns and 'AMOUNT_CLAIMED' in df.columns and 'TOTAL_PAID' in df.columns:
            try:
                provider_stats = df.groupby('PROVIDER_NAME').agg({
                    'AMOUNT_CLAIMED': 'sum',
                    'TOTAL_PAID': 'sum',
                    'MEMBER_NO': 'nunique'
                }).reset_index()
                provider_stats = provider_stats.sort_values('AMOUNT_CLAIMED', ascending=False).head(10)
                metrics['top_providers'] = provider_stats
            except:
                metrics['top_providers'] = pd.DataFrame()

        # Monthly trends
        if 'SERVICE_DATE' in df.columns and 'AMOUNT_CLAIMED' in df.columns:
            try:
                df['month_year'] = df['SERVICE_DATE'].dt.to_period('M')
                monthly_stats = df.groupby('month_year').agg({
                    'AMOUNT_CLAIMED': 'sum',
                    'TOTAL_PAID': 'sum',
                    'MEMBER_NO': 'nunique'
                }).reset_index()
                monthly_stats['month_year'] = monthly_stats['month_year'].astype(str)
                metrics['monthly_trends'] = monthly_stats
            except:
                metrics['monthly_trends'] = pd.DataFrame()

        return metrics

    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return metrics


# ============================================================================
# IBNR CALCULATION FUNCTIONS
# ============================================================================

def calculate_ibnr_excel_chain_ladder(df):
    """
    Replicate the Excel Chain Ladder Model from IBNR Model.xlsx
    This follows the exact methodology from the provided Excel template
    """
    if df is None or df.empty or 'SERVICE_DATE' not in df.columns or 'AMOUNT_CLAIMED' not in df.columns:
        return 0, {}

    try:
        # Create copy for processing
        df_clean = df.copy()

        # Convert service date to datetime and ensure proper format
        df_clean['SERVICE_DATE'] = pd.to_datetime(df_clean['SERVICE_DATE'], errors='coerce')
        df_clean = df_clean.dropna(subset=['SERVICE_DATE'])

        if df_clean.empty:
            return 0, {}

        # Extract year and month for accident period
        df_clean['accident_year'] = df_clean['SERVICE_DATE'].dt.year
        df_clean['accident_month'] = df_clean['SERVICE_DATE'].dt.month

        # Calculate development lag (months from accident date to now)
        current_date = pd.Timestamp.now()
        df_clean['development_lag'] = ((current_date - df_clean['SERVICE_DATE']).dt.days / 30.44).astype(int)

        # Cap development lag at 30 months (as per Excel model)
        df_clean['development_lag'] = df_clean['development_lag'].clip(upper=30)

        # Create incremental triangle (similar to Excel structure)
        # Group by accident period (monthly) and development lag
        inc_triangle = df_clean.pivot_table(
            index=['accident_year', 'accident_month'],
            columns='development_lag',
            values='AMOUNT_CLAIMED',
            aggfunc='sum',
            fill_value=0
        )

        # Sort columns by lag
        inc_triangle = inc_triangle.reindex(sorted(inc_triangle.columns), axis=1)

        # Create cumulative triangle (similar to Excel formulas)
        cum_triangle = inc_triangle.cumsum(axis=1)

        # Calculate development factors (age-to-age factors)
        # Similar to Excel's D0-D1, D1-D2, etc. calculations
        development_factors = {}

        for i in range(len(cum_triangle.columns) - 1):
            current_lag = cum_triangle.columns[i]
            next_lag = cum_triangle.columns[i + 1]

            # Calculate sum of cumulative claims at next lag
            numerator = cum_triangle[next_lag].sum()
            denominator = cum_triangle[current_lag].sum()

            if denominator > 0:
                factor = numerator / denominator
            else:
                factor = 1.0

            development_factors[f'D{current_lag}-D{next_lag}'] = factor

        # Calculate tail factor (for beyond observed development)
        # Excel appears to use a tail factor - for simplicity, using 1.0
        tail_factor = 1.0

        # Project ultimate claims using chain ladder method
        projected_ultimate = []

        for i, (index, row) in enumerate(cum_triangle.iterrows()):
            # Get latest observed cumulative value
            latest_observed = row.iloc[-1]

            # Calculate remaining development factors to apply
            remaining_factors = 1.0
            remaining_lags = len(cum_triangle.columns) - i - 1

            for j in range(remaining_lags):
                factor_key = f'D{cum_triangle.columns[j]}-D{cum_triangle.columns[j + 1]}'
                if factor_key in development_factors:
                    remaining_factors *= development_factors[factor_key]
                else:
                    # Apply tail factor for beyond observed
                    remaining_factors *= tail_factor

            ultimate_claim = latest_observed * remaining_factors
            projected_ultimate.append(ultimate_claim)

        # Calculate total reported and ultimate
        total_reported = cum_triangle.iloc[:, -1].sum()
        total_ultimate = sum(projected_ultimate)

        # Calculate IBNR
        ibnr_estimate = total_ultimate - total_reported

        # Calculate additional metrics similar to Excel
        latest_lag = cum_triangle.columns[-1]
        if cum_triangle[latest_lag].sum() > 0:
            ibnr_ratio = (ibnr_estimate / cum_triangle[latest_lag].sum()) * 100
        else:
            ibnr_ratio = 0

        return ibnr_estimate, {
            'method': 'Excel Chain Ladder Model',
            'total_reported': total_reported,
            'total_ultimate': total_ultimate,
            'ibnr_estimate': ibnr_estimate,
            'ibnr_ratio': ibnr_ratio,
            'development_factors': development_factors,
            'tail_factor': tail_factor,
            'incremental_triangle': inc_triangle,
            'cumulative_triangle': cum_triangle,
            'projected_ultimate': projected_ultimate
        }

    except Exception as e:
        st.error(f"Error in Excel Chain Ladder calculation: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return 0, {}


def calculate_development_triangle_actuarial(df, accident_period='M', development_period='M'):
    """Create a development triangle for claims data"""
    if df is None or 'SERVICE_DATE' not in df.columns or 'AMOUNT_CLAIMED' not in df.columns:
        return None, None, None

    try:
        # Create copy to avoid modifying original
        df_temp = df.copy()

        # Clean the SERVICE_DATE column - handle missing/invalid dates
        if 'SERVICE_DATE' in df_temp.columns:
            # Convert to datetime, coerce errors to NaT
            df_temp['SERVICE_DATE'] = pd.to_datetime(df_temp['SERVICE_DATE'], errors='coerce')

            # Check for missing dates
            missing_dates = df_temp['SERVICE_DATE'].isna().sum()
            if missing_dates > 0:
                st.warning(
                    f"Warning: {missing_dates} records have missing or invalid service dates. These records will be excluded from the development triangle.")
                # Remove rows with missing dates for triangle calculation
                df_temp = df_temp.dropna(subset=['SERVICE_DATE'])

            # Check for future dates
            current_date = pd.Timestamp.now()
            future_dates = (df_temp['SERVICE_DATE'] > current_date).sum()
            if future_dates > 0:
                st.warning(
                    f"Warning: {future_dates} records have future service dates. These dates will be capped at current date.")
                # Cap future dates at current date
                df_temp.loc[df_temp['SERVICE_DATE'] > current_date, 'SERVICE_DATE'] = current_date

        # If no valid dates remain after cleaning, return None
        if len(df_temp) == 0:
            st.error("No valid service dates found for development triangle calculation.")
            return None, None, None

        # Extract accident period (when claim occurred)
        df_temp['accident_period'] = df_temp['SERVICE_DATE'].dt.to_period(accident_period)

        # Calculate development lag (in periods from accident date to reporting/payment)
        current_date = pd.Timestamp.now()

        # Ensure we have valid datetime values before calculating lag
        valid_date_mask = df_temp['SERVICE_DATE'].notna()

        # Initialize development lag with zeros
        df_temp['development_lag'] = 0

        # Calculate lag only for valid dates
        if valid_date_mask.any():
            # Calculate days difference, handle any errors
            days_diff = (current_date - df_temp.loc[valid_date_mask, 'SERVICE_DATE']).dt.days

            # Replace any negative or invalid values with 0
            days_diff = days_diff.clip(lower=0)

            # Calculate months (using 30.44 days per month)
            months_diff = (days_diff / 30.44)

            # Convert to integer, handling any NaN values
            months_int = np.floor(months_diff).astype('Int64')  # Use nullable integer type

            # Assign back to dataframe
            df_temp.loc[valid_date_mask, 'development_lag'] = months_int.fillna(0).astype(int)

        # Create development period labels
        df_temp['development_period'] = df_temp['development_lag'].apply(lambda x: f"L{int(x)}")

        # Create incremental triangle
        inc_triangle = df_temp.pivot_table(
            index='accident_period',
            columns='development_period',
            values='AMOUNT_CLAIMED',
            aggfunc='sum',
            fill_value=0
        )

        # Sort columns by lag (L0, L1, L2, etc.)
        # Extract numeric part from column names for sorting
        def get_lag_number(col_name):
            try:
                return int(col_name[1:])  # Remove 'L' prefix
            except:
                return 0

        # Sort columns by lag number
        sorted_cols = sorted(inc_triangle.columns, key=get_lag_number)
        inc_triangle = inc_triangle.reindex(sorted_cols, axis=1)

        # Create cumulative triangle
        cum_triangle = inc_triangle.cumsum(axis=1)

        # Calculate development factors (age-to-age factors)
        dev_factors = {}
        for i in range(len(cum_triangle.columns) - 1):
            current_col = cum_triangle.columns[i]
            next_col = cum_triangle.columns[i + 1]

            # Calculate development factor for this lag
            # Use only non-zero denominators
            numerator = cum_triangle[next_col].sum()
            denominator = cum_triangle[current_col].sum()

            if denominator > 0:
                dev_factor = numerator / denominator
            else:
                dev_factor = 1.0

            dev_factors[f"{current_col}->{next_col}"] = {
                'factor': dev_factor,
                'periods': i + 1
            }

        return inc_triangle, cum_triangle, dev_factors

    except Exception as e:
        st.error(f"Error creating development triangle: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None, None


def calculate_ibnr_chain_ladder_actuarial(cum_triangle):
    """Calculate IBNR using Chain Ladder method"""
    if cum_triangle is None or cum_triangle.empty:
        return 0, {}

    try:
        n_periods = len(cum_triangle.columns)
        n_accident = len(cum_triangle.index)

        # Calculate age-to-age factors
        dev_factors = []
        for i in range(n_periods - 1):
            numerator = cum_triangle.iloc[:, i + 1].sum()
            denominator = cum_triangle.iloc[:, i].sum()
            if denominator > 0:
                dev_factor = numerator / denominator
            else:
                dev_factor = 1.0
            dev_factors.append(dev_factor)

        # Calculate tail factor (assume 1.05 for remaining development)
        tail_factor = 1.05

        # Project ultimate claims
        projected_ultimate = []
        for i in range(n_accident):
            current_cum = cum_triangle.iloc[i, -1]
            remaining_factors = 1.0

            # Multiply remaining development factors
            for j in range(n_accident - i - 1):
                if j < len(dev_factors):
                    remaining_factors *= dev_factors[j]
                else:
                    remaining_factors *= tail_factor

            ultimate = current_cum * remaining_factors
            projected_ultimate.append(ultimate)

        # Calculate IBNR
        reported_claims = cum_triangle.iloc[:, -1].sum()
        ultimate_claims = sum(projected_ultimate)
        ibnr = ultimate_claims - reported_claims

        return ibnr, {
            'method': 'Chain Ladder',
            'reported_claims': reported_claims,
            'ultimate_claims': ultimate_claims,
            'development_factors': dev_factors,
            'tail_factor': tail_factor
        }

    except Exception as e:
        st.error(f"Error in Chain Ladder calculation: {str(e)}")
        return 0, {}


def calculate_ibnr_bornhuetter_ferguson_actuarial(df, expected_loss_ratio=0.75):
    """Calculate IBNR using Bornhuetter-Ferguson method"""
    if df is None or df.empty or 'AMOUNT_CLAIMED' not in df.columns:
        return 0, {}

    try:
        # Calculate earned premium (simplified - use average premium per member * number of claims)
        if 'premium_metrics' in st.session_state and 'avg_premium' in st.session_state.premium_metrics:
            avg_premium = st.session_state.premium_metrics['avg_premium']
        else:
            # Estimate average premium if not available
            avg_premium = 100  # Default assumption

        num_claims = len(df)
        earned_premium = avg_premium * num_claims

        # Calculate expected ultimate losses
        expected_ultimate = earned_premium * expected_loss_ratio

        # Get reported claims
        reported_claims = df['AMOUNT_CLAIMED'].sum()

        # Calculate development pattern (simplified)
        if 'SERVICE_DATE' in df.columns:
            current_date = pd.Timestamp.now()
            df_temp = df.copy()
            df_temp['months_elapsed'] = ((current_date - df_temp['SERVICE_DATE']).dt.days / 30.44).astype(int)

            # Calculate percentage developed by age
            age_groups = pd.cut(df_temp['months_elapsed'], bins=[0, 3, 6, 9, 12, 24, 36, float('inf')])
            developed_by_age = df_temp.groupby(age_groups)['AMOUNT_CLAIMED'].sum() / reported_claims

            # Estimate remaining development
            remaining_development = 1.0 - developed_by_age.sum()
        else:
            remaining_development = 0.3  # Default assumption

        # Calculate IBNR using BF method
        ibnr = (expected_ultimate - reported_claims) * remaining_development

        return ibnr, {
            'method': 'Bornhuetter-Ferguson',
            'expected_loss_ratio': expected_loss_ratio,
            'earned_premium': earned_premium,
            'expected_ultimate': expected_ultimate,
            'reported_claims': reported_claims,
            'remaining_development': remaining_development
        }

    except Exception as e:
        st.error(f"Error in Bornhuetter-Ferguson calculation: {str(e)}")
        return 0, {}


def calculate_ibnr_frequency_severity_actuarial(df):
    """Calculate IBNR using Frequency-Severity method"""
    if df is None or df.empty:
        return 0, {}

    try:
        # Extract claim frequencies by month
        if 'SERVICE_DATE' in df.columns:
            df_temp = df.copy()
            df_temp['month'] = df_temp['SERVICE_DATE'].dt.to_period('M')
            monthly_counts = df_temp.groupby('month').size()

            # Fit Poisson distribution to claim frequency
            lambda_f


```
