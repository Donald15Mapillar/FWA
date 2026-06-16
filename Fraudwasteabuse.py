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
</style>
""", unsafe_allow_html=True)


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
        """Load claims and tariff data from uploaded files with flexible column matching"""
        try:
            # Read files
            self.claims_data = pd.read_excel(claims_file, sheet_name='Sheet1')
            self.tariff_data = pd.read_excel(tariff_file, sheet_name='Sample Tariff')

            # Define required columns and their possible variations for claims
            required_columns = {
                'CLM CODE': ['CLM CODE', 'CLM_CODE', 'Claim Code', 'CLAIM CODE', 'Procedure Code'],
                'Gender': ['Gender', 'SEX', 'GENDER'],
                'CURRENT AGE': ['CURRENT AGE', 'Age', 'AGE', 'Patient Age'],
                'UNITS': ['UNITS', 'Units', 'QUANTITY'],
                'TOTAL PAID': ['TOTAL PAID', 'Total Paid', 'PAID AMOUNT', 'Amount Paid'],
                'PROVIDER NAME': ['PROVIDER NAME', 'Provider', 'PROVIDER'],
                'MEMBER NO': ['MEMBER NO', 'Member ID', 'MEMBER_ID', 'Patient ID'],
                'SERVICE DATE': ['SERVICE DATE', 'Service Date', 'DATE OF SERVICE', 'ASSESS DATE', 'Assess Date'],
                'BASE BENEFIT DESCRIPTION': ['BASE BENEFIT DESCRIPTION', 'Benefit Description', 'DESCRIPTION'],
                'CLAIM NO': ['CLAIM NO', 'Claim Number', 'CLAIM_NUMBER'],
                'CLAIM LINE NO': ['CLAIM LINE NO', 'Line Number', 'CLAIM_LINE']
            }

            # Map actual columns to expected names
            column_mapping = {}
            missing = []
            for expected, variants in required_columns.items():
                found = False
                for col in self.claims_data.columns:
                    if col.strip().upper() in [v.upper() for v in variants]:
                        column_mapping[expected] = col
                        found = True
                        break
                if not found:
                    missing.append(expected)

            if missing:
                error_msg = f"Missing required columns: {', '.join(missing)}. Please ensure your file contains these columns."
                return False, error_msg

            # Rename columns to standard names for internal use
            self.claims_data.rename(columns={v: k for k, v in column_mapping.items()}, inplace=True)

            # Similarly for tariff data (only need CLM CODE and maybe AllowedGender, MaxUnits, CODE DESCRIPTION)
            tariff_required = {
                'CLM CODE': ['CLM CODE', 'CLM_CODE', 'Procedure Code'],
                'AllowedGender': ['AllowedGender', 'Gender Allowed', 'GENDER_ALLOWED'],
                'MaxUnits': ['MaxUnits', 'Maximum Units', 'MAX_UNITS'],
                'CODE DESCRIPTION': ['CODE DESCRIPTION', 'Description', 'PROC_DESC']
            }
            tariff_mapping = {}
            tariff_missing = []
            for expected, variants in tariff_required.items():
                found = False
                for col in self.tariff_data.columns:
                    if col.strip().upper() in [v.upper() for v in variants]:
                        tariff_mapping[expected] = col
                        found = True
                        break
                if not found:
                    tariff_missing.append(expected)

            if tariff_missing:
                error_msg = f"Missing required tariff columns: {', '.join(tariff_missing)}"
                return False, error_msg

            self.tariff_data.rename(columns={v: k for k, v in tariff_mapping.items()}, inplace=True)

            # Ensure CLM CODE is string type in both datasets
            self.claims_data['CLM CODE'] = self.claims_data['CLM CODE'].astype(str).str.strip()
            self.tariff_data['CLM CODE'] = self.tariff_data['CLM CODE'].astype(str).str.strip()

            # Debug info (now safe)
            st.write("✅ Data loaded successfully!")
            st.write("Claims columns after mapping:", list(self.claims_data.columns))
            st.write("Tariff columns after mapping:", list(self.tariff_data.columns))

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
            if 'BASE BENEFIT DESCRIPTION' not in self.flagged_claims.columns:
                st.warning("BASE BENEFIT DESCRIPTION column not found. Skipping age check for chronic medications.")
                return

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


# NEW: Function to create gender pie chart
def create_gender_pie_chart(flagged_data):
    """Create pie chart showing gender distribution of flagged claims"""
    if flagged_data is None or len(flagged_data) == 0:
        return

    flagged = flagged_data[flagged_data['FWA_Flag'] == True]

    if len(flagged) == 0:
        st.info("No flagged claims to analyze for gender distribution.")
        return

    if 'Gender' not in flagged.columns:
        st.warning("Gender column not found in flagged data.")
        return

    # Count genders in flagged claims
    gender_counts = flagged['Gender'].value_counts()

    # Create pie chart
    fig = go.Figure(data=[
        go.Pie(
            labels=gender_counts.index,
            values=gender_counts.values,
            hole=0.3,
            marker=dict(colors=['#667eea', '#764ba2', '#ff6b6b']),
            textinfo='label+percent+value',
            hoverinfo='label+percent+value'
        )
    ])

    fig.update_layout(
        title={
            'text': "Gender Distribution in Flagged Claims",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=400,
        showlegend=True
    )

    # Display the chart in a styled container
    st.markdown("""
    <div class="gender-pie-container">
        <h4>📊 Gender Analysis of Flagged Claims</h4>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(fig, use_container_width=True)

    # Add summary statistics
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Flagged Claims", len(flagged))

    with col2:
        if len(gender_counts) > 0:
            most_common_gender = gender_counts.index[0]
            st.metric("Most Flagged Gender", most_common_gender)

    # Add gender-specific insights
    st.markdown("### 👤 Gender-Specific Insights")

    if len(gender_counts) > 1:
        # Calculate gender distribution percentages
        total_flagged = len(flagged)
        for gender, count in gender_counts.items():
            percentage = (count / total_flagged * 100) if total_flagged > 0 else 0
            st.markdown(f"- **{gender}**: {count:,} claims ({percentage:.1f}%)")

        # Identify if there's a significant gender imbalance
        max_gender = gender_counts.idxmax()
        max_count = gender_counts.max()
        min_gender = gender_counts.idxmin()
        min_count = gender_counts.min()

        if total_flagged > 10:  # Only show if we have enough data
            imbalance_ratio = max_count / min_count if min_count > 0 else max_count
            if imbalance_ratio > 2:  # More than 2:1 ratio
                st.markdown(f"""
                <div class="age-warning">
                    ⚠️ <strong>Gender Imbalance Alert:</strong> {max_gender} claims are {imbalance_ratio:.1f}x 
                    more frequent than {min_gender} claims. This may require investigation.
                </div>
                """, unsafe_allow_html=True)


def create_Acturial_analytics_dashboard(flagged_data, claims_data=None):
    """Create comprehensive Acturial analytics dashboard with enhanced medication and benefit analysis"""
    if flagged_data is None or len(flagged_data) == 0:
        st.warning("No data available for analytics")
        return

    analytics = ActurialAnalytics()

    # Create tabs for different analytics views - ENHANCED with Benefits tab
    analytics_tabs = st.tabs([
        "📊 Statistical Analysis",
        "📈 Temporal Patterns",
        "🎯 Provider Analytics",
        "💊 Enhanced Medication Analytics",
        "🏥 Benefit Analysis",
        "📉 Benford's Analysis"
    ])

    with analytics_tabs[0]:
        st.markdown("### 📊 Statistical Analysis of Claims")

        # Statistical Outlier Detection Section
        st.markdown("---")
        st.markdown("#### 🔍 Statistical Outlier Detection")

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            method = st.selectbox(
                "Select outlier detection method:",
                ["IQR Method", "Z-Score Method", "Isolation Forest"],
                index=0,
                key="outlier_method"
            )

        with col2:
            # Determine available columns for analysis
            available_columns = []
            if 'TOTAL PAID' in flagged_data.columns:
                available_columns.append('TOTAL PAID')
            if 'AMOUNT CLAIMED' in flagged_data.columns:
                available_columns.append('AMOUNT CLAIMED')
            if 'UNITS' in flagged_data.columns:
                available_columns.append('UNITS')

            column_to_analyze = st.selectbox(
                "Select column for outlier analysis:",
                available_columns,
                index=0,
                key="outlier_column"
            )

        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_outlier_analysis = st.button("🚀 Run Outlier Analysis", type="primary", key="run_outlier_btn")

        if run_outlier_analysis:
            # Convert method name to parameter
            method_param = 'iqr' if method == "IQR Method" else 'zscore' if method == "Z-Score Method" else 'isolation_forest'

            # Run outlier detection
            with st.spinner(f"Running {method} outlier detection on {column_to_analyze}..."):
                outliers_result = analytics.detect_statistical_outliers(
                    flagged_data,
                    column_to_analyze,
                    method=method_param
                )

                if outliers_result:
                    st.session_state['outliers_result'] = outliers_result
                    st.session_state['outliers_column'] = column_to_analyze
                    st.session_state['outliers_method'] = method
                    st.success(f"✅ Outlier detection complete! Found {outliers_result['count']} outliers.")

        # Display statistical analysis
        st.markdown("---")
        st.markdown("#### 📈 Statistical Analysis of Claimed Amounts")

        if 'TOTAL PAID' in flagged_data.columns:
            col1, col2, col3 = st.columns(3)

            with col1:
                amounts = pd.to_numeric(flagged_data['TOTAL PAID'], errors='coerce').dropna()
                fig_dist = px.histogram(
                    amounts,
                    nbins=50,
                    title="Distribution of Total Paid Amounts",
                    labels={'value': 'Total Paid', 'count': 'Frequency'}
                )
                fig_dist.update_layout(height=400)
                st.plotly_chart(fig_dist, use_container_width=True)

            with col2:
                fig_box = px.box(
                    flagged_data,
                    y='TOTAL PAID',
                    title="Box Plot of Total Paid Amounts",
                    points="outliers"
                )
                fig_box.update_layout(height=400)
                st.plotly_chart(fig_box, use_container_width=True)

            with col3:
                st.markdown("##### Statistical Summary")
                stats_df = pd.DataFrame({
                    'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
                    'Value': [
                        len(amounts),
                        f"${amounts.mean():,.2f}",
                        f"${amounts.median():,.2f}",
                        f"${amounts.std():,.2f}",
                        f"${amounts.min():,.2f}",
                        f"${amounts.max():,.2f}",
                        f"{amounts.skew():.4f}",
                        f"{amounts.kurtosis():.4f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

                cv = amounts.std() / amounts.mean() if amounts.mean() > 0 else 0
                st.metric("Coefficient of Variation", f"{cv:.3f}")

    with analytics_tabs[1]:
        st.markdown("### 📈 Temporal Patterns Analysis")

        if 'SERVICE DATE' in flagged_data.columns:
            try:
                flagged_data['SERVICE_DATE_DT'] = pd.to_datetime(flagged_data['SERVICE DATE'], errors='coerce')
                temporal_data = flagged_data.dropna(subset=['SERVICE_DATE_DT'])

                col1, col2 = st.columns(2)

                with col1:
                    daily_counts = temporal_data.groupby(temporal_data['SERVICE_DATE_DT'].dt.date).size()
                    fig_daily = px.line(
                        x=daily_counts.index,
                        y=daily_counts.values,
                        title="Daily Claims Trend",
                        labels={'x': 'Date', 'y': 'Number of Claims'}
                    )
                    fig_daily.update_layout(height=400)
                    st.plotly_chart(fig_daily, use_container_width=True)

                with col2:
                    temporal_data['WEEKDAY'] = temporal_data['SERVICE_DATE_DT'].dt.day_name()
                    weekday_counts = temporal_data['WEEKDAY'].value_counts().reindex([
                        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
                    ])

                    fig_weekday = px.bar(
                        x=weekday_counts.index,
                        y=weekday_counts.values,
                        title="Claims by Day of Week",
                        labels={'x': 'Day', 'y': 'Number of Claims'}
                    )
                    fig_weekday.update_layout(height=400)
                    st.plotly_chart(fig_weekday, use_container_width=True)

                temporal_data['MONTH'] = temporal_data['SERVICE_DATE_DT'].dt.month_name()
                month_counts = temporal_data['MONTH'].value_counts()

                fig_month = px.bar(
                    x=month_counts.index,
                    y=month_counts.values,
                    title="Claims by Month",
                    labels={'x': 'Month', 'y': 'Number of Claims'}
                )
                fig_month.update_layout(height=400)
                st.plotly_chart(fig_month, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not analyze temporal patterns: {str(e)}")

    with analytics_tabs[2]:
        st.markdown("### 🎯 Provider Analytics")

        if 'PROVIDER NAME' in flagged_data.columns:
            provider_claims = flagged_data['PROVIDER NAME'].value_counts().head(20)

            fig_providers = px.bar(
                x=provider_claims.values,
                y=provider_claims.index,
                orientation='h',
                title="Top 20 Providers by Claim Volume",
                labels={'x': 'Number of Claims', 'y': 'Provider'}
            )
            fig_providers.update_layout(height=500)
            st.plotly_chart(fig_providers, use_container_width=True)

            if 'TOTAL PAID' in flagged_data.columns:
                provider_amounts = flagged_data.groupby('PROVIDER NAME')['TOTAL PAID'].sum().nlargest(10)

                fig_amounts = px.bar(
                    x=provider_amounts.values,
                    y=provider_amounts.index,
                    orientation='h',
                    title="Top 10 Providers by Total Amount Paid",
                    labels={'x': 'Total Amount Paid ($)', 'y': 'Provider'}
                )
                fig_amounts.update_layout(height=400)
                st.plotly_chart(fig_amounts, use_container_width=True)

            if 'provider_risk' in st.session_state:
                provider_risk = st.session_state['provider_risk']

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### Top 10 High-Risk Providers")
                    high_risk = provider_risk[provider_risk['risk_level'] == 'High'].head(10)

                    if not high_risk.empty:
                        fig_risk = px.bar(
                            x=high_risk['risk_score'],
                            y=high_risk.index,
                            orientation='h',
                            title="High-Risk Providers",
                            labels={'x': 'Risk Score', 'y': 'Provider'}
                        )
                        fig_risk.update_layout(height=400)
                        st.plotly_chart(fig_risk, use_container_width=True)

                with col2:
                    st.markdown("##### Risk Level Distribution")
                    risk_dist = provider_risk['risk_level'].value_counts()

                    fig_risk_dist = px.pie(
                        values=risk_dist.values,
                        names=risk_dist.index,
                        title="Provider Risk Level Distribution"
                    )
                    fig_risk_dist.update_layout(height=400)
                    st.plotly_chart(fig_risk_dist, use_container_width=True)

    with analytics_tabs[3]:
        # ENHANCED: Medication Analytics Tab
        st.markdown("### 💊 Enhanced Medication Analytics")

        # Run enhanced medication analysis if not already done
        if 'medication_analysis' not in st.session_state:
            with st.spinner("Analyzing medication patterns..."):
                medication_analysis = analytics.analyze_medication_patterns(flagged_data)
                if medication_analysis:
                    st.session_state['medication_analysis'] = medication_analysis

        if 'medication_analysis' in st.session_state:
            medication_analysis = st.session_state['medication_analysis']

            # Medication Overview
            st.markdown("#### 📊 Medication Analytics Overview")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if 'total_medication_claims' in medication_analysis:
                    st.metric("Total Medication Claims", f"{medication_analysis.get('total_medication_claims', 0):,}")

            with col2:
                if 'medication_percentage' in medication_analysis:
                    st.metric("Medication % of All Claims",
                              f"{medication_analysis.get('medication_percentage', 0):.1f}%")

            with col3:
                if 'duplicate_medication_claims' in medication_analysis:
                    st.metric("Duplicate Medication Claims",
                              f"{medication_analysis.get('duplicate_medication_claims', 0):,}")

            with col4:
                if 'polypharmacy_members' in medication_analysis:
                    st.metric("Polypharmacy Patients", f"{medication_analysis.get('polypharmacy_members', 0):,}")

            # Medication Type Breakdown
            st.markdown("#### 📈 Medication Type Analysis")

            # Create cards for each medication type
            col1, col2, col3 = st.columns(3)

            with col1:
                if 'chronic_medication_claims' in medication_analysis:
                    st.markdown(f"""
                    <div class="medication-card chronic-bg">
                        <div class="medication-header">🩺 Chronic Medications</div>
                        <div style="font-size: 1.5rem; font-weight: bold;">{medication_analysis['chronic_medication_claims']:,}</div>
                        <div>Claims</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if 'chronic_cost_stats' in medication_analysis:
                        st.metric("Total Cost",
                                  f"${medication_analysis['chronic_cost_stats'].get('total_cost', 0):,.0f}")

            with col2:
                if 'acute_medication_claims' in medication_analysis:
                    st.markdown(f"""
                    <div class="medication-card acute-bg">
                        <div class="medication-header">🚨 Acute Medications</div>
                        <div style="font-size: 1.5rem; font-weight: bold;">{medication_analysis['acute_medication_claims']:,}</div>
                        <div>Claims</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if 'acute_cost_stats' in medication_analysis:
                        st.metric("Total Cost", f"${medication_analysis['acute_cost_stats'].get('total_cost', 0):,.0f}")

            with col3:
                if 'otc_medication_claims' in medication_analysis:
                    st.markdown(f"""
                    <div class="medication-card otc-bg">
                        <div class="medication-header">💊 OTC Medications</div>
                        <div style="font-size: 1.5rem; font-weight: bold;">{medication_analysis['otc_medication_claims']:,}</div>
                        <div>Claims</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if 'otc_cost_stats' in medication_analysis:
                        st.metric("Total Cost", f"${medication_analysis['otc_cost_stats'].get('total_cost', 0):,.0f}")

            # NEW: Age Compliance in Chronic Medications
            if 'chronic_under_17_count' in medication_analysis and medication_analysis['chronic_under_17_count'] > 0:
                st.markdown("#### ⚠️ Age Compliance in Chronic Medications")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    <div class="medication-chronic">
                        <strong>🔴 Age Compliance Issue:</strong> 
                        {medication_analysis['chronic_under_17_count']} chronic medication claims from patients under 17 years old
                    </div>
                    """, unsafe_allow_html=True)

                    if 'chronic_under_17_percentage' in medication_analysis:
                        st.metric(
                            "Percentage of Chronic Claims",
                            f"{medication_analysis['chronic_under_17_percentage']:.1f}%"
                        )

                with col2:
                    # Create a simple bar chart for age distribution
                    if 'CURRENT AGE' in flagged_data.columns:
                        chronic_ages = flagged_data[
                            (flagged_data['BASE BENEFIT DESCRIPTION'].astype(str).str.contains('chronic', case=False,
                                                                                               na=False)) &
                            (flagged_data['FWA_Flag'] == True)
                            ]

                        if len(chronic_ages) > 0:
                            under_17 = len(
                                chronic_ages[pd.to_numeric(chronic_ages['CURRENT AGE'], errors='coerce') < 17])
                            over_17 = len(chronic_ages) - under_17

                            fig_age = px.bar(
                                x=['Under 17', '17 and Over'],
                                y=[under_17, over_17],
                                title="Chronic Medication Claims by Age Group",
                                color=['Under 17', '17 and Over'],
                                color_discrete_map={'Under 17': '#ff6b6b', '17 and Over': '#667eea'}
                            )
                            fig_age.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig_age, use_container_width=True)

            # Top Medication Benefits
            if 'top_medication_benefits' in medication_analysis:
                st.markdown("#### 🏆 Top Medication Benefits")

                top_benefits = medication_analysis['top_medication_benefits']
                benefits_df = pd.DataFrame({
                    'Benefit Description': list(top_benefits.keys()),
                    'Claim Count': list(top_benefits.values())
                })

                fig_top_meds = px.bar(
                    benefits_df,
                    x='Claim Count',
                    y='Benefit Description',
                    orientation='h',
                    title="Top 10 Medication Benefits by Claim Count",
                    color='Claim Count',
                    color_continuous_scale='Blues'
                )
                fig_top_meds.update_layout(height=500)
                st.plotly_chart(fig_top_meds, use_container_width=True)

            # Provider Medication Analysis
            if 'top_medication_providers' in medication_analysis:
                st.markdown("#### 🏥 Top Medication Prescribers")

                top_providers = medication_analysis['top_medication_providers']
                providers_df = pd.DataFrame({
                    'Provider': list(top_providers.keys()),
                    'Medication Claims': list(top_providers.values())
                })

                fig_med_providers = px.bar(
                    providers_df,
                    x='Medication Claims',
                    y='Provider',
                    orientation='h',
                    title="Top Providers by Medication Claims",
                    color='Medication Claims',
                    color_continuous_scale='Greens'
                )
                fig_med_providers.update_layout(height=400)
                st.plotly_chart(fig_med_providers, use_container_width=True)

            # Medication Cost Analysis
            if 'medication_cost_stats' in medication_analysis:
                st.markdown("#### 💰 Medication Cost Analysis")

                cost_stats = medication_analysis['medication_cost_stats']
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Medication Cost", f"${cost_stats.get('total_cost', 0):,.0f}")

                with col2:
                    st.metric("Average Cost per Claim", f"${cost_stats.get('avg_cost', 0):,.2f}")

                with col3:
                    st.metric("Cost per Member", f"${cost_stats.get('cost_per_member', 0):,.2f}")

                with col4:
                    st.metric("Cost per Provider", f"${cost_stats.get('cost_per_provider', 0):,.2f}")

            # Polypharmacy Analysis
            if 'polypharmacy_members' in medication_analysis and medication_analysis['polypharmacy_members'] > 0:
                st.markdown("#### ⚠️ Polypharmacy Analysis")

                st.markdown(f"""
                <div class="medication-highlight">
                    <strong>⚠️ Alert:</strong> {medication_analysis['polypharmacy_members']} patients are on multiple medications (polypharmacy)
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Polypharmacy Patients", f"{medication_analysis['polypharmacy_members']:,}")

                with col2:
                    if 'polypharmacy_rate' in medication_analysis:
                        st.metric("Polypharmacy Rate", f"{medication_analysis['polypharmacy_rate']:.1f}%")

                # Show common combinations
                if 'common_polypharmacy_combinations' in medication_analysis:
                    st.markdown("**Common Medication Combinations:**")
                    for member, meds in list(medication_analysis['common_polypharmacy_combinations'].items())[:5]:
                        st.write(f"- Member {member}: {len(meds)} medications including {', '.join(meds[:3])}")

            # FWA Indicators
            if 'fwa_indicators' in medication_analysis and medication_analysis['fwa_indicators']:
                st.markdown("#### 🚨 Potential FWA Indicators")

                fwa_indicators = medication_analysis['fwa_indicators']

                if 'frequent_same_medication' in fwa_indicators:
                    st.markdown(f"""
                    <div class="medication-chronic">
                        <strong>🔴 High-frequency medication use:</strong> {fwa_indicators['frequent_same_medication']} cases of frequent same medication prescriptions
                    </div>
                    """, unsafe_allow_html=True)

                if 'high_cost_otc_claims' in fwa_indicators:
                    st.markdown(f"""
                    <div class="medication-otc">
                        <strong>🟠 High-cost OTC claims:</strong> {fwa_indicators['high_cost_otc_claims']} OTC claims over $100
                    </div>
                    """, unsafe_allow_html=True)

                if 'providers_excessive_med_variety' in fwa_indicators:
                    st.markdown(f"""
                    <div class="medication-acute">
                        <strong>🟡 Excessive medication variety:</strong> {fwa_indicators['providers_excessive_med_variety']} providers prescribing >10 medication types
                    </div>
                    """, unsafe_allow_html=True)

            # Recommendations
            st.markdown("#### 💡 Pharmaceutical Fraud Prevention Recommendations")

            recommendations = [
                "**Chronic Medication Monitoring**: Track long-term medication usage for appropriateness",
                "**Acute Medication Validation**: Verify necessity of short-term medications",
                "**OTC Cost Controls**: Implement limits on OTC medication costs"
            ]

            # Add conditional recommendations
            if medication_analysis.get('polypharmacy_members', 0) > 0:
                recommendations.append(
                    "**Polypharmacy Review**: Audit patients on multiple medications for potential interactions")

            if medication_analysis.get('duplicate_medication_claims', 0) > 0:
                recommendations.append("**Duplicate Detection**: Investigate duplicate medication claims")

            # Add age compliance recommendation
            if medication_analysis.get('chronic_under_17_count', 0) > 0:
                recommendations.append(
                    "**Age Compliance Audit**: Review all chronic medication claims for patients under 17 years old")

            recommendations.extend([
                "**Provider Profiling**: Monitor providers with unusual medication prescribing patterns",
                "**Cost Benchmarking**: Compare medication costs against industry averages",
                "**Benefit Analysis**: Review medication benefit descriptions for consistency"
            ])

            for rec in recommendations:
                st.markdown(f"• {rec}")

        else:
            st.warning(
                "No medication data available for analysis. Ensure your dataset contains 'BASE BENEFIT DESCRIPTION' column.")

    with analytics_tabs[4]:
        # NEW: Benefit Analysis Tab
        st.markdown("### 🏥 Benefit Pattern Analysis")

        # Run benefit analysis if not already done
        if 'benefit_analysis' not in st.session_state:
            with st.spinner("Analyzing benefit patterns..."):
                benefit_analysis = analytics.analyze_benefit_patterns(flagged_data)
                if benefit_analysis:
                    st.session_state['benefit_analysis'] = benefit_analysis

        if 'benefit_analysis' in st.session_state:
            benefit_analysis = st.session_state['benefit_analysis']

            # Benefit Overview
            st.markdown("#### 📊 Benefit Analysis Overview")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if 'total_benefit_categories' in benefit_analysis:
                    st.metric("Benefit Categories", f"{benefit_analysis.get('total_benefit_categories', 0):,}")

            with col2:
                if 'benefit_categories' in benefit_analysis:
                    med_count = benefit_analysis['benefit_categories'].get('MEDICATION', 0)
                    st.metric("Medication Benefits", f"{med_count:,}")

            with col3:
                if 'members_multiple_benefits' in benefit_analysis:
                    st.metric("Multi-Benefit Members", f"{benefit_analysis.get('members_multiple_benefits', 0):,}")

            with col4:
                if 'avg_benefits_per_member' in benefit_analysis:
                    st.metric("Avg Benefits per Member", f"{benefit_analysis.get('avg_benefits_per_member', 0):.1f}")

            # Benefit Category Distribution
            if 'benefit_categories' in benefit_analysis:
                st.markdown("#### 📈 Benefit Category Distribution")

                categories = benefit_analysis['benefit_categories']
                categories_df = pd.DataFrame({
                    'Category': list(categories.keys()),
                    'Claim Count': list(categories.values())
                })

                fig_categories = px.pie(
                    categories_df,
                    values='Claim Count',
                    names='Category',
                    title="Distribution of Benefit Categories",
                    hole=0.4
                )
                fig_categories.update_layout(height=500)
                st.plotly_chart(fig_categories, use_container_width=True)

            # Top Benefits by Cost
            if 'top_benefits_by_cost' in benefit_analysis:
                st.markdown("#### 💰 Top Benefits by Total Cost")

                top_benefits_cost = benefit_analysis['top_benefits_by_cost']
                benefits_cost_df = pd.DataFrame({
                    'Benefit Description': list(top_benefits_cost.keys()),
                    'Total Cost ($)': list(top_benefits_cost.values())
                })

                fig_cost = px.bar(
                    benefits_cost_df,
                    x='Total Cost ($)',
                    y='Benefit Description',
                    orientation='h',
                    title="Top 10 Benefits by Total Cost",
                    color='Total Cost ($)',
                    color_continuous_scale='Viridis'
                )
                fig_cost.update_layout(height=500)
                st.plotly_chart(fig_cost, use_container_width=True)

            # Provider Benefit Specialization
            if 'top_specialized_providers' in benefit_analysis:
                st.markdown("#### 🎯 Provider Benefit Specialization")

                specialized_providers = benefit_analysis['top_specialized_providers']
                specialization_df = pd.DataFrame({
                    'Provider': list(specialized_providers.keys()),
                    'Specialization Score': list(specialized_providers.values())
                })

                fig_specialization = px.bar(
                    specialization_df,
                    x='Specialization Score',
                    y='Provider',
                    orientation='h',
                    title="Top 10 Specialized Providers (High concentration in one benefit)",
                    color='Specialization Score',
                    color_continuous_scale='Reds'
                )
                fig_specialization.update_layout(height=400)
                st.plotly_chart(fig_specialization, use_container_width=True)

            # Monthly Trends by Benefit
            if 'monthly_trends_by_benefit' in benefit_analysis:
                st.markdown("#### 📅 Monthly Benefit Trends")

                monthly_trends = benefit_analysis['monthly_trends_by_benefit']

                # Create line chart for top benefits
                trend_data = []
                for benefit, monthly_counts in monthly_trends.items():
                    for month, count in monthly_counts.items():
                        trend_data.append({
                            'Benefit': benefit,
                            'Month': str(month),
                            'Claim Count': count
                        })

                if trend_data:
                    trend_df = pd.DataFrame(trend_data)
                    fig_trends = px.line(
                        trend_df,
                        x='Month',
                        y='Claim Count',
                        color='Benefit',
                        title="Monthly Claim Trends by Top Benefits",
                        markers=True
                    )
                    fig_trends.update_layout(height=400)
                    st.plotly_chart(fig_trends, use_container_width=True)

            # High-Cost Benefit Anomalies
            if 'potential_high_cost_anomalies' in benefit_analysis:
                anomalies = benefit_analysis['potential_high_cost_anomalies']
                if anomalies:
                    st.markdown("#### ⚠️ High-Cost Benefit Anomalies")

                    st.markdown(f"""
                    <div class="benefit-analysis">
                        <strong>🚨 Alert:</strong> Found {len(anomalies)} benefit categories with unusually high costs
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**Potential Anomalies:**")
                    for anomaly in anomalies[:10]:  # Show top 10
                        st.write(f"- {anomaly}")

            # Recommendations
            st.markdown("#### 💡 Benefit Analysis Recommendations")

            benefit_recommendations = [
                "**Benefit Utilization Review**: Monitor unusual patterns in benefit utilization",
                "**Cost Control**: Implement controls for high-cost benefit categories",
                "**Provider Education**: Educate providers on appropriate benefit coding"
            ]

            # Add conditional recommendations
            if benefit_analysis.get('top_specialized_providers'):
                benefit_recommendations.append(
                    "**Specialization Monitoring**: Review highly specialized providers for appropriateness")

            benefit_recommendations.extend([
                "**Member Education**: Educate members on appropriate benefit utilization",
                "**Trend Analysis**: Continuously monitor benefit utilization trends",
                "**Benchmarking**: Compare benefit patterns against industry benchmarks"
            ])

            for rec in benefit_recommendations:
                st.markdown(f"• {rec}")

        else:
            st.warning(
                "No benefit analysis data available. Ensure your dataset contains 'BASE BENEFIT DESCRIPTION' column.")

    with analytics_tabs[5]:
        st.markdown("### 📉 Benford's Law Analysis")

        if 'TOTAL PAID' in flagged_data.columns:
            benford_result = analytics.calculate_benfords_law(flagged_data['TOTAL PAID'])

            if benford_result:
                col1, col2 = st.columns(2)

                with col1:
                    digits = list(range(1, 10))
                    observed = [benford_result['observed'].get(d, 0) for d in digits]
                    expected = [benford_result['expected'].get(d, 0) for d in digits]

                    fig_benford = go.Figure()

                    fig_benford.add_trace(go.Bar(
                        x=digits,
                        y=observed,
                        name='Observed',
                        marker_color='#667eea'
                    ))

                    fig_benford.add_trace(go.Scatter(
                        x=digits,
                        y=expected,
                        name='Expected (Benford)',
                        mode='lines+markers',
                        line=dict(color='#ff6b6b', width=3)
                    ))

                    fig_benford.update_layout(
                        title="Benford's Law Analysis - First Digit Distribution",
                        xaxis_title="First Digit",
                        yaxis_title="Frequency (%)",
                        height=400,
                        showlegend=True
                    )

                    st.plotly_chart(fig_benford, use_container_width=True)

                with col2:
                    st.markdown("##### Statistical Analysis")

                    metrics_df = pd.DataFrame({
                        'Metric': ['Chi-Square Statistic', 'P-Value', 'Anomaly Score', 'Analysis'],
                        'Value': [
                            f"{benford_result['chi_square']:.4f}",
                            f"{benford_result['p_value']:.6f}",
                            f"{benford_result['anomaly_score']:.3f}",
                            "Potential Anomaly" if benford_result['p_value'] < 0.05 else "Normal Distribution"
                        ]
                    })

                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                    if benford_result['p_value'] < 0.05:
                        st.error("""
                        **⚠️ Potential Data Anomaly Detected**

                        The distribution of first digits significantly deviates from Benford's Law (p < 0.05).
                        This could indicate:
                        - Potential data manipulation
                        - Unusual billing patterns
                        - Systematic fraud
                        - Data entry anomalies
                        """)
                    else:
                        st.success("""
                        **✅ Normal Distribution**

                        The distribution of first digits follows Benford's Law (p ≥ 0.05).
                        This suggests natural, unmanipulated data patterns.
                        """)


def create_key_insights_panel(flagged_data):
    """Create panel with key analytical insights"""
    if flagged_data is None or len(flagged_data) == 0:
        return

    flagged = flagged_data[flagged_data['FWA_Flag'] == True]

    if len(flagged) == 0:
        return

    st.markdown("### 🔍 Key Analytical Insights")

    insights = []

    # Insight 1: Top flagged providers
    if 'PROVIDER NAME' in flagged.columns:
        top_providers = flagged['PROVIDER NAME'].value_counts().head(3)
        if len(top_providers) > 0:
            provider_list = ", ".join([f"{p} ({c})" for p, c in top_providers.items()])
            insights.append(f"**Top flagged providers**: {provider_list}")

    # Insight 2: Financial impact
    if 'TOTAL PAID' in flagged.columns:
        total_flagged = flagged['TOTAL PAID'].sum()
        total_all = flagged_data['TOTAL PAID'].sum()
        percentage = (total_flagged / total_all * 100) if total_all > 0 else 0
        insights.append(f"**Financial impact**: ${total_flagged:,.2f} flagged ({percentage:.1f}% of total)")

    # Insight 3: Medication insights
    if 'medication_analysis' in st.session_state:
        medication_analysis = st.session_state['medication_analysis']

        if 'total_medication_claims' in medication_analysis:
            insights.append(
                f"**Medication claims**: {medication_analysis['total_medication_claims']:,} medication-related claims")

        if 'medication_percentage' in medication_analysis and medication_analysis['medication_percentage'] > 20:
            insights.append(
                f"**High medication percentage**: {medication_analysis['medication_percentage']:.1f}% of all claims")

        if 'polypharmacy_members' in medication_analysis and medication_analysis['polypharmacy_members'] > 10:
            insights.append(
                f"**Polypharmacy concern**: {medication_analysis['polypharmacy_members']} patients on multiple medications")

        if 'duplicate_medication_claims' in medication_analysis and medication_analysis[
            'duplicate_medication_claims'] > 10:
            insights.append(
                f"**Duplicate medications**: {medication_analysis['duplicate_medication_claims']} potential duplicate claims")

        # NEW: Age compliance insights
        if 'chronic_under_17_count' in medication_analysis and medication_analysis['chronic_under_17_count'] > 0:
            insights.append(
                f"**Age compliance issue**: {medication_analysis['chronic_under_17_count']} chronic medication claims from patients under 17")

    # Insight 4: Benefit insights
    if 'benefit_analysis' in st.session_state:
        benefit_analysis = st.session_state['benefit_analysis']

        if 'total_benefit_categories' in benefit_analysis:
            insights.append(
                f"**Benefit variety**: {benefit_analysis['total_benefit_categories']} different benefit categories")

        if 'potential_high_cost_anomalies' in benefit_analysis and benefit_analysis['potential_high_cost_anomalies']:
            insights.append(
                f"**High-cost anomalies**: {len(benefit_analysis['potential_high_cost_anomalies'])} benefit categories with unusual costs")

    # Display insights
    for insight in insights:
        st.markdown(f"""
        <div class="insight-card">
            {insight}
        </div>
        """, unsafe_allow_html=True)


def get_download_link(df, filename):
    """Generate a download link for DataFrame"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Flagged_Claims')

        if 'TOTAL PAID' in df.columns:
            amounts = pd.to_numeric(df['TOTAL PAID'], errors='coerce').dropna()
            analytics_summary = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
                'Value': [
                    f"${amounts.mean():,.2f}",
                    f"${amounts.median():,.2f}",
                    f"${amounts.std():,.2f}",
                    f"${amounts.min():,.2f}",
                    f"${amounts.max():,.2f}",
                    f"{amounts.skew():.4f}",
                    f"{amounts.kurtosis():.4f}"
                ]
            })
            analytics_summary.to_excel(writer, index=False, sheet_name='Analytics_Summary')

    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" class="summary-box" style="display: inline-block; padding: 0.5rem 1rem; background: #667eea; color: white; text-decoration: none; border-radius: 5px; font-weight: 500;">📥 Download Excel File</a>'
    return href


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

                # NEW: Gender Pie Chart Section
                st.markdown("### 👥 Gender Distribution Analysis")
                create_gender_pie_chart(flagged_data)

                # NEW: Age Compliance Warning for Chronic Medications
                if 'medication_analysis' in st.session_state:
                    medication_analysis = st.session_state['medication_analysis']
                    if 'chronic_under_17_count' in medication_analysis and medication_analysis[
                        'chronic_under_17_count'] > 0:
                        chronic_count = medication_analysis['chronic_under_17_count']
                        st.markdown(f"""
                        <div class="age-warning">
                            ⚠️ <strong>Age Compliance Alert:</strong> Found {chronic_count} chronic medication claims 
                            from patients under 17 years old. These have been flagged for review.
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
                                                    ["All", "Gender Issues", "Duplicate Claims", "Unit Overdosing",
                                                     "Age Chronic Meds"])

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
                elif issue_filter == "Age Chronic Meds":
                    display_data = display_data[
                        display_data['Flag_Reason'].str.contains('under 17 cannot claim Chronic Medications', na=False)]

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
                            <li>Gender distribution analysis</li>
                            <li>Age compliance findings for chronic medications</li>
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
                        if 'chronic_under_17_count' in medication_analysis and medication_analysis[
                            'chronic_under_17_count'] > 0:
                            insights.append(
                                f"👶 **Age compliance issue**: {medication_analysis['chronic_under_17_count']} chronic medication claims from patients under 17")

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

                if st.session_state.get('medication_analysis', {}).get('chronic_under_17_count', 0) > 0:
                    recommendations.append(
                        "**Age Compliance Audit**: Review all chronic medication claims for patients under 17 years old")

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

                st.markdown("""
                **👥 NEW: Gender & Age Analytics**
                - Gender distribution pie charts
                - Age compliance for chronic medications
                - Gender-based claim pattern analysis
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


Question 1

please note that Household Income Survey data file is already inported
This question is based on Household Income Survey data obtained from 4048 households. The variable Income represents the average total income earned by the household . Expenditure is the variable for household expenditure level , given the income.Age is the household head. Gender, is gender for the household head coded as 1 for male and 0 for female. Employment is the dummy variable for employment status of the household head.It asssumed that value 1 if the household head is employed and 0 if unemployed.Farm is the farming income.
    a)Test for the normality for the variables Age ,Income, Expenditure and Farm using the Shapiro Wilk test .Interpret the results, Export as word file.