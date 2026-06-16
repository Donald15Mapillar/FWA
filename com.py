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

    /* Gender pie chart styling */
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


class ActuarialAnalytics:
    """Actuarial analytical methods for FWA detection"""

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

            # Age analysis for chronic medications
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
        self.analytics = ActuarialAnalytics()  # Updated class name
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
        """Save analysis results to database with Actuarial analytics, and save full data as Excel"""
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

            # Save full data as Excel backup
            self._save_full_data_excel(flagged_data, analysis_id)

            return analysis_id

        except Exception as e:
            st.error(f"Error saving analysis: {str(e)}")
            return None

    def _save_full_data_excel(self, data, analysis_id):
        """Save full flagged data as Excel file"""
        excel_path = self.data_dir / f"full_data_{analysis_id}.xlsx"
        try:
            data.to_excel(excel_path, index=False)
            return True
        except Exception as e:
            st.warning(f"Could not save full data Excel backup: {e}")
            return False

    def calculate_analytics_metrics(self, flagged_data):
        """Calculate Actuarial analytics metrics and ensure JSON serializable"""
        metrics = {}

        try:
            # Basic statistical metrics
            if 'TOTAL PAID' in flagged_data.columns:
                amounts = pd.to_numeric(flagged_data['TOTAL PAID'], errors='coerce').dropna()
                if len(amounts) > 0:
                    metrics['statistical'] = {
                        'mean_amount': float(amounts.mean()),
                        'median_amount': float(amounts.median()),
                        'std_amount': float(amounts.std()),
                        'skewness': float(amounts.skew()),
                        'kurtosis': float(amounts.kurtosis()),
                        'cv': float(amounts.std() / amounts.mean()) if amounts.mean() != 0 else 0.0
                    }

            # Claim pattern metrics
            pattern_metrics = self.analytics.calculate_claim_pattern_metrics(flagged_data)
            if pattern_metrics:
                # Convert any numpy values in pattern_metrics to Python native types
                metrics['pattern'] = self._convert_to_serializable(pattern_metrics)

            return metrics

        except Exception as e:
            st.warning(f"Analytics metrics calculation failed: {str(e)}")
            return {}

    def _convert_to_serializable(self, obj):
        """Recursively convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_serializable(obj.tolist())
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return obj

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
        self.analytics = ActuarialAnalytics()  # Updated class name

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

            status_text.text("Performing Actuarial analytics...")
            self._perform_Actuarial_analytics()
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
                'detection_rules': ['gender', 'age', 'duplicate', 'unit_overdose', 'Actuarial_analytics',
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
        self._check_age_for_chronic_medications()  # Added age check for chronic medications

    def _check_age_for_chronic_medications(self):
        """Check if patients under 17 are claiming chronic medications"""
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

    def _perform_Actuarial_analytics(self):
        """Perform Actuarial analytical checks"""
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
        # ... (unchanged) ...
        pass

    def _check_male_inappropriate_codes(self):
        # ... (unchanged) ...
        pass

    def _check_female_inappropriate_codes(self):
        # ... (unchanged) ...
        pass

    def _check_age_appropriateness(self):
        # ... (unchanged) ...
        pass

    def _check_unit_overdosing(self):
        # ... (unchanged) ...
        pass

    def _check_duplicate_claims(self):
        # ... (unchanged) ...
        pass

    def _add_flag(self, index, reason, flag_type):
        # ... (unchanged) ...
        pass

    def generate_executive_summary(self):
        # ... (unchanged) ...
        pass

    def _generate_key_findings(self, flagged_claims):
        # ... (unchanged) ...
        pass


# Function to create gender pie chart
def create_gender_pie_chart(flagged_data):
    # ... (unchanged) ...
    pass


def display_flagged_claims_panel(flagged_data):
    # ... (unchanged) ...
    pass


def create_Actuarial_analytics_dashboard(flagged_data, claims_data=None):
    """Create comprehensive Actuarial analytics dashboard with enhanced medication and benefit analysis"""
    # ... (unchanged, but function name updated) ...
    pass


def create_key_insights_panel(flagged_data):
    # ... (unchanged) ...
    pass


def get_download_link(df, filename):
    # ... (unchanged) ...
    pass


def display_history_panel(history_manager):
    """Display historical analyses in a table."""
    st.markdown("### 📜 Past Analyses")
    analyses = history_manager.get_all_analyses(limit=20)
    if not analyses:
        st.info("No historical analyses found. Run an analysis first.")
        return

    # Convert to DataFrame for display
    df_history = pd.DataFrame(analyses)
    # Format columns
    df_history['timestamp'] = pd.to_datetime(df_history['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    df_history['flagged_percentage'] = df_history['flagged_percentage'].round(1).astype(str) + '%'
    df_history['flagged_paid_amount'] = df_history['flagged_paid_amount'].apply(lambda x: f"${x:,.2f}")
    df_history['potential_recovery'] = df_history['potential_recovery'].apply(lambda x: f"${x:,.2f}")

    # Rename columns for display
    df_history = df_history.rename(columns={
        'analysis_name': 'Analysis Name',
        'timestamp': 'Date/Time',
        'total_claims': 'Total Claims',
        'flagged_claims': 'Flagged Claims',
        'flagged_percentage': 'Flagged %',
        'flagged_paid_amount': 'Flagged Amount',
        'potential_recovery': 'Potential Recovery',
        'fraud_count': 'Fraud',
        'waste_count': 'Waste',
        'abuse_count': 'Abuse'
    })

    # Select columns to show
    display_cols = ['Analysis Name', 'Date/Time', 'Total Claims', 'Flagged Claims', 'Flagged %', 'Flagged Amount',
                    'Potential Recovery', 'Fraud', 'Waste', 'Abuse']
    st.dataframe(df_history[display_cols], use_container_width=True, hide_index=True)

    # Optionally, allow selecting an analysis to view details (could be expanded later)
    st.caption("Full data for each analysis is saved in the 'historical_data' folder with the analysis ID.")


def main():
    # Header with modern design – now includes logo on the left
    col_logo, col_title, col_badge = st.columns([1, 3, 1])

    with col_logo:
        logo_path = "C:/Users/tmapillar/OneDrive - Generation Health/Desktop/Donald Final draft/Final FWA 2025/Generation health logo.ico"
        if Path(logo_path).exists():
            st.image(logo_path, width=80)
        else:
            st.markdown("### 🏥")

    with col_title:
        st.markdown('<div class="main-header">FWA Detection System</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Actuarial Analytics for Fraud, Waste & Abuse Detection in Healthcare Claims</div>',
            unsafe_allow_html=True)

    with col_badge:
        st.markdown("")
        st.markdown("")
        st.markdown("###  Actuarial Analytics")

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
        display_history_panel(st.session_state.fwa_detector.history_manager)

    elif st.session_state.tariff_uploaded:
        if st.session_state.analysis_run and hasattr(st.session_state.fwa_detector,
                                                     'flagged_claims') and st.session_state.fwa_detector.flagged_claims is not None:
            flagged_data = st.session_state.fwa_detector.flagged_claims

            # Create modern tabs with enhanced analytics
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Executive Dashboard",
                "🚩 Flagged Claims",
                "📈 Actuarial Analytics",
                "🔍 Detailed View",
                "📥 Export",
                "📋 History",
                "🎯 Insights"
            ])

            with tab1:
                # ... (all content from previous version, but make sure any displayed "Actuarial" text is updated) ...
                # I'll assume the inner functions already use correct spelling; otherwise, adjust them.
                pass

            with tab2:
                st.markdown("### 🚩 Flagged Claims Analysis")
                display_flagged_claims_panel(flagged_data)

            with tab3:
                create_Actuarial_analytics_dashboard(flagged_data, st.session_state.fwa_detector.claims_data)

            with tab4:
                st.markdown("### 🔍 Detailed Claims Analysis")
                # ... (filter options and dataframe) ...
                pass

            with tab5:
                st.markdown("### 📥 Export Results")
                # ... (export button) ...
                pass

            with tab6:
                st.markdown("### 📋 Analysis History")
                display_history_panel(st.session_state.fwa_detector.history_manager)

            with tab7:
                st.markdown("### 🎯 Analytical Insights")
                # ... (insights and recommendations) ...
                pass

        else:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem;">
                <h2>🚀 Ready to Analyze Claims</h2>
                <p style="font-size: 1.1rem; color: #6c757d; max-width: 600px; margin: 0 auto 2rem;">
                    Upload your claims data and run the analysis to uncover potential fraud, waste, and abuse patterns.
                </p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📋 Actuarial Analytics Capabilities", expanded=True):
                # ... (capabilities list) ...
                pass

    else:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <h2> Welcome to FWA Detection System</h2>
            <p style="font-size: 1.1rem; color: #6c757d; max-width: 700px; margin: 0 auto 2rem;">
                Actuarial analytics platform for detecting Fraud, Waste, and Abuse in healthcare claims. 
                Get started by uploading your tariff file in the sidebar.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()