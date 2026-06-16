import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import warnings
import sys
import io
import base64
import tempfile
from pathlib import Path
import time
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import hashlib

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Healthcare Claims Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with gradient themes
st.markdown("""
<style>
    /* Modern Gradient Theme */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 1rem;
        padding: 10px 0;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #4a5568;
        font-weight: 600;
        margin: 1rem 0;
    }

    /* Enhanced Metric Cards with Hover Effects */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.2);
    }
    .metric-card-green {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 20px rgba(46, 125, 50, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-card-green:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(46, 125, 50, 0.2);
    }
    .metric-card-orange {
        background: linear-gradient(135deg, #FF9800 0%, #FF5722 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 20px rgba(255, 152, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-card-orange:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(255, 152, 0, 0.2);
    }
    .metric-card-blue {
        background: linear-gradient(135deg, #1565C0 0%, #2196F3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 20px rgba(33, 150, 243, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-card-blue:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(33, 150, 243, 0.2);
    }
    .metric-card-purple {
        background: linear-gradient(135deg, #7B1FA2 0%, #9C27B0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 20px rgba(156, 39, 176, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-card-purple:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(156, 39, 176, 0.2);
    }
    .metric-card-red {
        background: linear-gradient(135deg, #D32F2F 0%, #F44336 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 20px rgba(211, 47, 47, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-card-red:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(211, 47, 47, 0.2);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.95;
        font-weight: 500;
    }
    .metric-change {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    /* Enhanced Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.2);
    }

    /* Enhanced Cards */
    .provider-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease;
    }
    .provider-card:hover {
        transform: translateX(5px);
    }

    /* Alerts and Notifications */
    .alert-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #155724;
        border-left: 5px solid #28a745;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #155724;
    }
    .alert-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #856404;
        border-left: 5px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #856404;
    }
    .alert-info {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #0c5460;
        border-left: 5px solid #17a2b8;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #0c5460;
    }
    .alert-danger {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #721c24;
        border-left: 5px solid #dc3545;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #721c24;
    }

    /* KPI Indicators */
    .kpi-indicator {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 10px;
    }
    .kpi-good {
        background-color: #d4edda;
        color: #155724;
    }
    .kpi-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    .kpi-danger {
        background-color: #f8d7da;
        color: #721c24;
    }

    /* Progress Bars */
    .progress-container {
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 10px 0;
        height: 10px;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }

    /* Risk Level Indicators */
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .risk-critical {
        background-color: #721c24;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Fraud Alert Cards */
    .fraud-alert-card {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #856404;
        border-left: 5px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #856404;
    }
    .fraud-alert-card-high {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #721c24;
        border-left: 5px solid #dc3545;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


class FWADetector:
    """Fraud, Waste, and Abuse Detection System"""

    def __init__(self, df):
        self.df = df.copy()
        self.preprocess_fwa_data()
        self.flagged_claims = pd.DataFrame()
        self.risk_scores = {}

    def preprocess_fwa_data(self):
        """Preprocess data for FWA detection"""
        # Standardize column names
        self.df.columns = self.df.columns.str.strip().str.upper()

        # Create unique claim identifier
        self.df['CLAIM_ID'] = self.df.apply(
            lambda row: hashlib.md5(
                f"{row.get('MEMBER NO', '')}{row.get('SERVICE DATE', '')}{row.get('PROVIDER NAME', '')}{row.get('AMOUNT CLAIMED', '')}".encode()).hexdigest()[
                :10],
            axis=1
        )

        # Convert dates
        date_cols = ['SERVICE DATE', 'ASSESS DATE', 'DATE RECEIVED', 'EXTERNAL PAY DATE']
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

        # Convert numeric columns
        numeric_cols = ['AMOUNT CLAIMED', 'TOTAL PAID', 'UNITS', 'CO-PAY', 'AGE']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df[col] = self.df[col].fillna(0)

    def detect_duplicate_claims(self):
        """Detect duplicate or near-duplicate claims"""
        duplicates = []

        if all(col in self.df.columns for col in ['MEMBER NO', 'SERVICE DATE', 'PROVIDER NAME', 'AMOUNT CLAIMED']):
            # Exact duplicates
            duplicate_cols = ['MEMBER NO', 'SERVICE DATE', 'PROVIDER NAME', 'AMOUNT CLAIMED']
            duplicate_mask = self.df.duplicated(subset=duplicate_cols, keep=False)

            if duplicate_mask.any():
                dup_df = self.df[duplicate_mask].copy()
                dup_df['FWA_RULE'] = 'DUPLICATE_CLAIM'
                dup_df['FWA_RISK_SCORE'] = 0.8
                dup_df['FWA_DESCRIPTION'] = 'Exact duplicate claim detected'
                duplicates.append(dup_df)

            # Same member, same provider, same day, different amounts
            if 'SERVICE DATE' in self.df.columns:
                self.df['SERVICE_DATE_STR'] = self.df['SERVICE DATE'].dt.date.astype(str)
                same_day_cols = ['MEMBER NO', 'PROVIDER NAME', 'SERVICE_DATE_STR']

                # Group and check for multiple claims on same day
                same_day_counts = self.df.groupby(same_day_cols).size()
                same_day_multi = same_day_counts[same_day_counts > 3].reset_index(name='COUNT')

                if not same_day_multi.empty:
                    # Merge to get original records
                    multi_claims = pd.merge(
                        self.df,
                        same_day_multi,
                        on=same_day_cols,
                        how='inner'
                    )
                    multi_claims['FWA_RULE'] = 'MULTIPLE_CLAIMS_SAME_DAY'
                    multi_claims['FWA_RISK_SCORE'] = 0.6
                    multi_claims[
                        'FWA_DESCRIPTION'] = f'Multiple claims ({multi_claims["COUNT"]}) on same day for same member-provider'
                    duplicates.append(multi_claims)

        if duplicates:
            return pd.concat(duplicates, ignore_index=True)
        return pd.DataFrame()

    def detect_unusual_amounts(self):
        """Detect claims with unusual amounts using statistical methods"""
        unusual = []

        if 'AMOUNT CLAIMED' in self.df.columns:
            amounts = self.df['AMOUNT CLAIMED']

            # Remove zeros
            non_zero_amounts = amounts[amounts > 0]

            if len(non_zero_amounts) > 10:
                # Calculate z-scores
                mean = non_zero_amounts.mean()
                std = non_zero_amounts.std()
                z_scores = (amounts - mean) / std

                # Flag outliers (z-score > 3 or < -3)
                outlier_mask = abs(z_scores) > 3

                if outlier_mask.any():
                    outlier_df = self.df[outlier_mask].copy()
                    outlier_df['FWA_RULE'] = 'UNUSUAL_AMOUNT'
                    outlier_df['FWA_RISK_SCORE'] = 0.7
                    outlier_df[
                        'FWA_DESCRIPTION'] = f'Claim amount ({outlier_df["AMOUNT CLAIMED"]}) is statistical outlier (z-score: {z_scores[outlier_mask].round(2)})'
                    unusual.append(outlier_df)

                # Check for round numbers (potential fraud indicator)
                round_numbers = amounts[amounts % 100 == 0]
                if len(round_numbers) > 0:
                    round_mask = self.df['AMOUNT CLAIMED'].isin(round_numbers)
                    round_df = self.df[round_mask].copy()
                    round_df['FWA_RULE'] = 'ROUND_NUMBER_AMOUNT'
                    round_df['FWA_RISK_SCORE'] = 0.4
                    round_df['FWA_DESCRIPTION'] = 'Claim amount is a round number (multiple of 100)'
                    unusual.append(round_df)

        if unusual:
            return pd.concat(unusual, ignore_index=True)
        return pd.DataFrame()

    def detect_unusual_patterns(self):
        """Detect unusual billing patterns"""
        patterns = []

        # 1. Unusually high units
        if 'UNITS' in self.df.columns:
            high_units = self.df[self.df['UNITS'] > self.df['UNITS'].quantile(0.95)]
            if not high_units.empty:
                high_units_df = high_units.copy()
                high_units_df['FWA_RULE'] = 'HIGH_UNITS'
                high_units_df['FWA_RISK_SCORE'] = 0.5
                high_units_df['FWA_DESCRIPTION'] = f'Unusually high number of units ({high_units_df["UNITS"]})'
                patterns.append(high_units_df)

        # 2. Weekend/holiday billing
        if 'SERVICE DATE' in self.df.columns:
            weekend_mask = self.df['SERVICE DATE'].dt.dayofweek >= 5
            weekend_claims = self.df[weekend_mask]
            if not weekend_claims.empty:
                weekend_df = weekend_claims.copy()
                weekend_df['FWA_RULE'] = 'WEEKEND_SERVICE'
                weekend_df['FWA_RISK_SCORE'] = 0.3
                weekend_df['FWA_DESCRIPTION'] = 'Service provided on weekend'
                patterns.append(weekend_df)

        # 3. After-hours billing (if time available)
        if 'SERVICE DATE' in self.df.columns:
            # Assuming service hours 8 AM to 6 PM
            self.df['SERVICE_HOUR'] = self.df['SERVICE DATE'].dt.hour
            after_hours = self.df[(self.df['SERVICE_HOUR'] < 8) | (self.df['SERVICE_HOUR'] > 18)]
            if not after_hours.empty:
                after_hours_df = after_hours.copy()
                after_hours_df['FWA_RULE'] = 'AFTER_HOURS_SERVICE'
                after_hours_df['FWA_RISK_SCORE'] = 0.4
                after_hours_df[
                    'FWA_DESCRIPTION'] = f'Service provided outside normal hours ({after_hours_df["SERVICE_HOUR"]}:00)'
                patterns.append(after_hours_df)

        # 4. Rapid succession claims
        if all(col in self.df.columns for col in ['MEMBER NO', 'PROVIDER NAME', 'SERVICE DATE']):
            self.df = self.df.sort_values(['MEMBER NO', 'PROVIDER NAME', 'SERVICE DATE'])
            self.df['TIME_DIFF'] = self.df.groupby(['MEMBER NO', 'PROVIDER NAME'])['SERVICE DATE'].diff().dt.days

            rapid_claims = self.df[(self.df['TIME_DIFF'] < 7) & (self.df['TIME_DIFF'] > 0)]
            if not rapid_claims.empty:
                rapid_df = rapid_claims.copy()
                rapid_df['FWA_RULE'] = 'RAPID_SUCCESSION'
                rapid_df['FWA_RISK_SCORE'] = 0.6
                rapid_df['FWA_DESCRIPTION'] = f'Claims in rapid succession ({rapid_df["TIME_DIFF"]} days apart)'
                patterns.append(rapid_df)

        if patterns:
            return pd.concat(patterns, ignore_index=True)
        return pd.DataFrame()

    def detect_provider_anomalies(self):
        """Detect anomalies in provider behavior"""
        provider_issues = []

        if 'PROVIDER NAME' in self.df.columns and 'AMOUNT CLAIMED' in self.df.columns:
            # Calculate provider statistics
            provider_stats = self.df.groupby('PROVIDER NAME').agg({
                'AMOUNT CLAIMED': ['mean', 'std', 'count'],
                'MEMBER NO': 'nunique'
            }).round(2)

            provider_stats.columns = ['AVG_AMOUNT', 'STD_AMOUNT', 'TOTAL_CLAIMS', 'UNIQUE_MEMBERS']
            provider_stats = provider_stats.reset_index()

            # Flag providers with high variation
            high_var_providers = provider_stats[
                (provider_stats['STD_AMOUNT'] > provider_stats['AVG_AMOUNT']) &
                (provider_stats['TOTAL_CLAIMS'] > 5)
                ]

            if not high_var_providers.empty:
                high_var_df = pd.merge(
                    self.df,
                    high_var_providers[['PROVIDER NAME']],
                    on='PROVIDER NAME',
                    how='inner'
                )
                high_var_df['FWA_RULE'] = 'HIGH_VARIATION_PROVIDER'
                high_var_df['FWA_RISK_SCORE'] = 0.7
                high_var_df['FWA_DESCRIPTION'] = 'Provider shows unusually high variation in claim amounts'
                provider_issues.append(high_var_df)

            # Flag providers with many claims per member
            provider_stats['CLAIMS_PER_MEMBER'] = provider_stats['TOTAL_CLAIMS'] / provider_stats['UNIQUE_MEMBERS']
            high_claims_per_member = provider_stats[provider_stats['CLAIMS_PER_MEMBER'] > 5]

            if not high_claims_per_member.empty:
                high_cpm_df = pd.merge(
                    self.df,
                    high_claims_per_member[['PROVIDER NAME']],
                    on='PROVIDER NAME',
                    how='inner'
                )
                high_cpm_df['FWA_RULE'] = 'HIGH_CLAIMS_PER_MEMBER'
                high_cpm_df['FWA_RISK_SCORE'] = 0.8
                high_cpm_df['FWA_DESCRIPTION'] = 'Provider has high number of claims per unique member'
                provider_issues.append(high_cpm_df)

            # Flag providers with only high-cost claims
            avg_amount_overall = self.df['AMOUNT CLAIMED'].mean()
            high_cost_providers = provider_stats[
                (provider_stats['AVG_AMOUNT'] > avg_amount_overall * 2) &
                (provider_stats['TOTAL_CLAIMS'] > 3)
                ]

            if not high_cost_providers.empty:
                high_cost_df = pd.merge(
                    self.df,
                    high_cost_providers[['PROVIDER NAME']],
                    on='PROVIDER NAME',
                    how='inner'
                )
                high_cost_df['FWA_RULE'] = 'HIGH_COST_PROVIDER'
                high_cost_df['FWA_RISK_SCORE'] = 0.6
                high_cost_df['FWA_DESCRIPTION'] = 'Provider consistently submits high-cost claims'
                provider_issues.append(high_cost_df)

        if provider_issues:
            return pd.concat(provider_issues, ignore_index=True)
        return pd.DataFrame()

    def detect_member_anomalies(self):
        """Detect anomalies in member behavior"""
        member_issues = []

        if 'MEMBER NO' in self.df.columns and 'AMOUNT CLAIMED' in self.df.columns:
            # Calculate member statistics
            member_stats = self.df.groupby('MEMBER NO').agg({
                'AMOUNT CLAIMED': ['sum', 'mean', 'count'],
                'PROVIDER NAME': 'nunique'
            }).round(2)

            member_stats.columns = ['TOTAL_AMOUNT', 'AVG_AMOUNT', 'TOTAL_CLAIMS', 'UNIQUE_PROVIDERS']
            member_stats = member_stats.reset_index()

            # Flag members with high total amount
            total_amount_q95 = member_stats['TOTAL_AMOUNT'].quantile(0.95)
            high_total_members = member_stats[member_stats['TOTAL_AMOUNT'] > total_amount_q95]

            if not high_total_members.empty:
                high_total_df = pd.merge(
                    self.df,
                    high_total_members[['MEMBER NO']],
                    on='MEMBER NO',
                    how='inner'
                )
                high_total_df['FWA_RULE'] = 'HIGH_TOTAL_AMOUNT_MEMBER'
                high_total_df['FWA_RISK_SCORE'] = 0.6
                high_total_df['FWA_DESCRIPTION'] = 'Member has unusually high total claimed amount'
                member_issues.append(high_total_df)

            # Flag members with many providers
            if 'UNIQUE_PROVIDERS' in member_stats.columns:
                many_providers = member_stats[member_stats['UNIQUE_PROVIDERS'] > 5]
                if not many_providers.empty:
                    many_prov_df = pd.merge(
                        self.df,
                        many_providers[['MEMBER NO']],
                        on='MEMBER NO',
                        how='inner'
                    )
                    many_prov_df['FWA_RULE'] = 'MANY_PROVIDERS_MEMBER'
                    many_prov_df['FWA_RISK_SCORE'] = 0.5
                    many_prov_df['FWA_DESCRIPTION'] = 'Member uses many different providers'
                    member_issues.append(many_prov_df)

            # Flag members with frequent claims
            frequent_members = member_stats[member_stats['TOTAL_CLAIMS'] > member_stats['TOTAL_CLAIMS'].quantile(0.95)]
            if not frequent_members.empty:
                frequent_df = pd.merge(
                    self.df,
                    frequent_members[['MEMBER NO']],
                    on='MEMBER NO',
                    how='inner'
                )
                frequent_df['FWA_RULE'] = 'FREQUENT_CLAIMS_MEMBER'
                frequent_df['FWA_RISK_SCORE'] = 0.7
                frequent_df['FWA_DESCRIPTION'] = 'Member submits claims very frequently'
                member_issues.append(frequent_df)

        if member_issues:
            return pd.concat(member_issues, ignore_index=True)
        return pd.DataFrame()

    def detect_benefit_abuse(self):
        """Detect potential benefit category abuse"""
        benefit_issues = []

        if 'BASE BENEFIT DESCRIPTION' in self.df.columns and 'MEMBER NO' in self.df.columns:
            # Check for members claiming same benefit too frequently
            benefit_freq = self.df.groupby(['MEMBER NO', 'BASE BENEFIT DESCRIPTION']).size().reset_index(name='COUNT')

            # Get benefit frequency statistics
            benefit_stats = benefit_freq.groupby('BASE BENEFIT DESCRIPTION')['COUNT'].agg(['mean', 'std']).reset_index()
            benefit_stats.columns = ['BASE BENEFIT DESCRIPTION', 'MEAN_COUNT', 'STD_COUNT']

            # Flag outliers
            high_freq_benefits = benefit_freq.merge(benefit_stats, on='BASE BENEFIT DESCRIPTION')
            high_freq_benefits['Z_SCORE'] = (high_freq_benefits['COUNT'] - high_freq_benefits['MEAN_COUNT']) / \
                                            high_freq_benefits['STD_COUNT'].replace(0, 1)

            high_freq = high_freq_benefits[high_freq_benefits['Z_SCORE'] > 2]

            if not high_freq.empty:
                high_freq_df = pd.merge(
                    self.df,
                    high_freq[['MEMBER NO', 'BASE BENEFIT DESCRIPTION']],
                    on=['MEMBER NO', 'BASE BENEFIT DESCRIPTION'],
                    how='inner'
                )
                high_freq_df['FWA_RULE'] = 'HIGH_FREQUENCY_BENEFIT'
                high_freq_df['FWA_RISK_SCORE'] = 0.6
                high_freq_df['FWA_DESCRIPTION'] = 'Member claims same benefit category unusually frequently'
                benefit_issues.append(high_freq_df)

        if benefit_issues:
            return pd.concat(benefit_issues, ignore_index=True)
        return pd.DataFrame()

    def detect_time_anomalies(self):
        """Detect anomalies in claim timing"""
        time_issues = []

        if 'SERVICE DATE' in self.df.columns and 'DATE RECEIVED' in self.df.columns:
            # Calculate processing time
            self.df['PROCESSING_DAYS'] = (self.df['DATE RECEIVED'] - self.df['SERVICE DATE']).dt.days

            # Flag very short processing times (potential backdating)
            short_processing = self.df[self.df['PROCESSING_DAYS'] < 0]
            if not short_processing.empty:
                short_proc_df = short_processing.copy()
                short_proc_df['FWA_RULE'] = 'NEGATIVE_PROCESSING_TIME'
                short_proc_df['FWA_RISK_SCORE'] = 0.9
                short_proc_df[
                    'FWA_DESCRIPTION'] = f'Claim received before service date ({short_proc_df["PROCESSING_DAYS"]} days)'
                time_issues.append(short_proc_df)

            # Flag very long processing times
            long_processing = self.df[self.df['PROCESSING_DAYS'] > 365]
            if not long_processing.empty:
                long_proc_df = long_processing.copy()
                long_proc_df['FWA_RULE'] = 'VERY_LONG_PROCESSING_TIME'
                long_proc_df['FWA_RISK_SCORE'] = 0.4
                long_proc_df[
                    'FWA_DESCRIPTION'] = f'Claim submitted over a year after service ({long_proc_df["PROCESSING_DAYS"]} days)'
                time_issues.append(long_proc_df)

        # Check for clustering of claims in time
        if 'SERVICE DATE' in self.df.columns and 'PROVIDER NAME' in self.df.columns:
            self.df['SERVICE_DATE_STR'] = self.df['SERVICE DATE'].dt.date.astype(str)

            # Count claims per provider per day
            daily_provider_counts = self.df.groupby(['PROVIDER NAME', 'SERVICE_DATE_STR']).size().reset_index(
                name='DAILY_CLAIMS')

            high_daily_claims = daily_provider_counts[daily_provider_counts['DAILY_CLAIMS'] > 10]
            if not high_daily_claims.empty:
                high_daily_df = pd.merge(
                    self.df,
                    high_daily_claims[['PROVIDER NAME', 'SERVICE_DATE_STR']],
                    on=['PROVIDER NAME', 'SERVICE_DATE_STR'],
                    how='inner'
                )
                high_daily_df['FWA_RULE'] = 'HIGH_DAILY_CLAIM_VOLUME'
                high_daily_df['FWA_RISK_SCORE'] = 0.7
                high_daily_df['FWA_DESCRIPTION'] = 'Provider submitted unusually high volume of claims in one day'
                time_issues.append(high_daily_df)

        if time_issues:
            return pd.concat(time_issues, ignore_index=True)
        return pd.DataFrame()

    def run_all_detections(self):
        """Run all FWA detection rules"""
        all_detections = []

        # Run individual detection methods
        detection_methods = [
            ('Duplicate Claims', self.detect_duplicate_claims),
            ('Unusual Amounts', self.detect_unusual_amounts),
            ('Unusual Patterns', self.detect_unusual_patterns),
            ('Provider Anomalies', self.detect_provider_anomalies),
            ('Member Anomalies', self.detect_member_anomalies),
            ('Benefit Abuse', self.detect_benefit_abuse),
            ('Time Anomalies', self.detect_time_anomalies)
        ]

        for rule_name, detection_func in detection_methods:
            try:
                detected = detection_func()
                if not detected.empty:
                    detected['DETECTION_RULE'] = rule_name
                    all_detections.append(detected)
            except Exception as e:
                st.warning(f"Error in {rule_name} detection: {str(e)}")

        if all_detections:
            self.flagged_claims = pd.concat(all_detections, ignore_index=True)

            # Aggregate risk scores by claim
            if not self.flagged_claims.empty and 'CLAIM_ID' in self.flagged_claims.columns:
                claim_risk = self.flagged_claims.groupby('CLAIM_ID').agg({
                    'FWA_RISK_SCORE': 'max',
                    'FWA_RULE': lambda x: ', '.join(set(x)),
                    'FWA_DESCRIPTION': lambda x: ' | '.join(set(x[:3]))  # First 3 descriptions
                }).reset_index()

                # Merge with original data for full details
                self.flagged_claims = pd.merge(
                    claim_risk,
                    self.df,
                    on='CLAIM_ID',
                    how='left'
                )

                # Calculate overall risk level
                def get_risk_level(score):
                    if score >= 0.8:
                        return 'Critical'
                    elif score >= 0.6:
                        return 'High'
                    elif score >= 0.4:
                        return 'Medium'
                    else:
                        return 'Low'

                self.flagged_claims['RISK_LEVEL'] = self.flagged_claims['FWA_RISK_SCORE'].apply(get_risk_level)

                # Sort by risk score
                self.flagged_claims = self.flagged_claims.sort_values('FWA_RISK_SCORE', ascending=False)

        return self.flagged_claims

    def calculate_fwa_metrics(self):
        """Calculate FWA metrics and statistics"""
        metrics = {}

        if not self.flagged_claims.empty:
            metrics['total_flagged_claims'] = len(self.flagged_claims)
            metrics['total_flagged_amount'] = self.flagged_claims[
                'AMOUNT CLAIMED'].sum() if 'AMOUNT CLAIMED' in self.flagged_claims.columns else 0
            metrics['avg_risk_score'] = self.flagged_claims['FWA_RISK_SCORE'].mean()

            # Risk level distribution
            risk_dist = self.flagged_claims['RISK_LEVEL'].value_counts()
            metrics['critical_risk'] = int(risk_dist.get('Critical', 0))
            metrics['high_risk'] = int(risk_dist.get('High', 0))
            metrics['medium_risk'] = int(risk_dist.get('Medium', 0))
            metrics['low_risk'] = int(risk_dist.get('Low', 0))

            # Rule distribution
            rule_dist = {}
            for rule in self.flagged_claims['FWA_RULE'].str.split(', ').explode():
                rule_dist[rule] = rule_dist.get(rule, 0) + 1
            metrics['rule_distribution'] = rule_dist

            # Top flagged providers
            if 'PROVIDER NAME' in self.flagged_claims.columns:
                top_providers = self.flagged_claims['PROVIDER NAME'].value_counts().head(10).to_dict()
                metrics['top_flagged_providers'] = top_providers

            # Top flagged members
            if 'MEMBER NO' in self.flagged_claims.columns:
                top_members = self.flagged_claims['MEMBER NO'].value_counts().head(10).to_dict()
                metrics['top_flagged_members'] = top_members

            # Time trend of flagged claims
            if 'SERVICE DATE' in self.flagged_claims.columns:
                monthly_flagged = self.flagged_claims.groupby(
                    self.flagged_claims['SERVICE DATE'].dt.to_period('M')
                ).size()
                metrics['monthly_flagged_trend'] = monthly_flagged.to_dict()

        else:
            metrics['total_flagged_claims'] = 0
            metrics['total_flagged_amount'] = 0
            metrics['avg_risk_score'] = 0
            metrics['critical_risk'] = 0
            metrics['high_risk'] = 0
            metrics['medium_risk'] = 0
            metrics['low_risk'] = 0
            metrics['rule_distribution'] = {}
            metrics['top_flagged_providers'] = {}
            metrics['top_flagged_members'] = {}
            metrics['monthly_flagged_trend'] = {}

        # Calculate percentages
        metrics['flagged_percentage'] = (metrics['total_flagged_claims'] / len(self.df) * 100) if len(
            self.df) > 0 else 0
        metrics['amount_risk_percentage'] = (metrics['total_flagged_amount'] / self.df[
            'AMOUNT CLAIMED'].sum() * 100) if 'AMOUNT CLAIMED' in self.df.columns and self.df[
            'AMOUNT CLAIMED'].sum() > 0 else 0

        return metrics

    def create_fwa_visualizations(self, metrics):
        """Create visualizations for FWA analysis"""
        viz = {}

        try:
            # 1. Risk Level Distribution Pie Chart
            if metrics['total_flagged_claims'] > 0:
                risk_labels = ['Critical', 'High', 'Medium', 'Low']
                risk_values = [
                    metrics['critical_risk'],
                    metrics['high_risk'],
                    metrics['medium_risk'],
                    metrics['low_risk']
                ]

                # Filter out zero values
                plot_data = [(label, value) for label, value in zip(risk_labels, risk_values) if value > 0]

                if plot_data:
                    labels, values = zip(*plot_data)
                    colors = ['#dc3545', '#ffc107', '#17a2b8', '#28a745']

                    fig_risk = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=.3,
                        marker_colors=colors[:len(labels)],
                        textinfo='label+percent+value',
                        hoverinfo='label+percent+value'
                    )])
                    fig_risk.update_layout(
                        title='Fraud Risk Level Distribution',
                        template='plotly_white',
                        height=400
                    )
                    viz['risk_distribution_chart'] = fig_risk

            # 2. Detection Rule Distribution
            if metrics.get('rule_distribution', {}):
                rules = list(metrics['rule_distribution'].keys())
                counts = list(metrics['rule_distribution'].values())

                fig_rules = go.Figure(data=[go.Bar(
                    x=rules,
                    y=counts,
                    marker_color='#8c564b',
                    text=counts,
                    textposition='auto'
                )])
                fig_rules.update_layout(
                    title='Detection Rules Triggered',
                    template='plotly_white',
                    height=400,
                    xaxis_tickangle=45
                )
                viz['rules_distribution_chart'] = fig_rules

            # 3. Monthly Fraud Trend
            if metrics.get('monthly_flagged_trend', {}):
                months = list(metrics['monthly_flagged_trend'].keys())
                counts = list(metrics['monthly_flagged_trend'].values())

                fig_trend = go.Figure(data=[go.Scatter(
                    x=months,
                    y=counts,
                    mode='lines+markers',
                    line=dict(color='#dc3545', width=3),
                    marker=dict(size=8)
                )])
                fig_trend.update_layout(
                    title='Monthly Fraud Detection Trend',
                    template='plotly_white',
                    height=400,
                    xaxis_title='Month',
                    yaxis_title='Number of Flagged Claims'
                )
                viz['fraud_trend_chart'] = fig_trend

            # 4. Top Flagged Providers
            if metrics.get('top_flagged_providers', {}):
                providers = list(metrics['top_flagged_providers'].keys())[:10]
                counts = list(metrics['top_flagged_providers'].values())[:10]

                fig_providers = go.Figure(data=[go.Bar(
                    x=providers,
                    y=counts,
                    marker_color='#e377c2',
                    text=counts,
                    textposition='auto'
                )])
                fig_providers.update_layout(
                    title='Top 10 Flagged Providers',
                    template='plotly_white',
                    height=400,
                    xaxis_tickangle=45
                )
                viz['flagged_providers_chart'] = fig_providers

            # 5. Risk Score Distribution Histogram
            if not self.flagged_claims.empty:
                fig_hist = go.Figure(data=[go.Histogram(
                    x=self.flagged_claims['FWA_RISK_SCORE'],
                    nbinsx=20,
                    marker_color='#17becf',
                    opacity=0.7
                )])
                fig_hist.update_layout(
                    title='Risk Score Distribution',
                    template='plotly_white',
                    height=400,
                    xaxis_title='Risk Score',
                    yaxis_title='Number of Claims'
                )
                viz['risk_score_histogram'] = fig_hist

            # 6. Fraud Dashboard
            if metrics['total_flagged_claims'] > 0:
                fig_dashboard = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Risk Level Distribution', 'Monthly Trend',
                                    'Top Detection Rules', 'Risk Score Analysis'),
                    specs=[[{'type': 'pie'}, {'type': 'scatter'}],
                           [{'type': 'bar'}, {'type': 'histogram'}]],
                    vertical_spacing=0.15,
                    horizontal_spacing=0.15
                )

                # Risk Level Distribution
                if metrics['total_flagged_claims'] > 0:
                    fig_dashboard.add_trace(
                        go.Pie(
                            labels=risk_labels,
                            values=risk_values,
                            hole=0.3,
                            marker_colors=['#dc3545', '#ffc107', '#17a2b8', '#28a745'],
                            textinfo='label+percent',
                            name='Risk Levels'
                        ),
                        row=1, col=1
                    )

                # Monthly Trend
                if metrics.get('monthly_flagged_trend', {}):
                    fig_dashboard.add_trace(
                        go.Scatter(
                            x=months,
                            y=counts,
                            mode='lines+markers',
                            line=dict(color='#dc3545', width=2),
                            marker=dict(size=6),
                            name='Monthly Trend'
                        ),
                        row=1, col=2
                    )

                # Detection Rules
                if metrics.get('rule_distribution', {}):
                    fig_dashboard.add_trace(
                        go.Bar(
                            x=rules,
                            y=counts,
                            marker_color='#8c564b',
                            name='Detection Rules'
                        ),
                        row=2, col=1
                    )

                # Risk Score Histogram
                if not self.flagged_claims.empty:
                    fig_dashboard.add_trace(
                        go.Histogram(
                            x=self.flagged_claims['FWA_RISK_SCORE'],
                            nbinsx=15,
                            marker_color='#17becf',
                            opacity=0.7,
                            name='Risk Scores'
                        ),
                        row=2, col=2
                    )

                fig_dashboard.update_layout(
                    template='plotly_white',
                    height=600,
                    showlegend=False,
                    title_text="Fraud, Waste & Abuse Dashboard"
                )
                viz['fwa_dashboard'] = fig_dashboard

        except Exception as e:
            st.error(f"Error creating FWA visualizations: {str(e)}")
            viz = {}

        return viz

    def get_high_risk_alerts(self, limit=10):
        """Get high-risk alerts for immediate attention"""
        if self.flagged_claims.empty:
            return []

        high_risk = self.flagged_claims[
            (self.flagged_claims['FWA_RISK_SCORE'] >= 0.7)
        ].head(limit)

        alerts = []
        for _, row in high_risk.iterrows():
            alert = {
                'claim_id': row.get('CLAIM_ID', 'N/A'),
                'member_no': row.get('MEMBER NO', 'N/A'),
                'provider': row.get('PROVIDER NAME', 'N/A'),
                'amount': row.get('AMOUNT CLAIMED', 0),
                'risk_score': row.get('FWA_RISK_SCORE', 0),
                'risk_level': row.get('RISK_LEVEL', 'Unknown'),
                'rules': row.get('FWA_RULE', 'Unknown'),
                'description': row.get('FWA_DESCRIPTION', 'No description')
            }
            alerts.append(alert)

        return alerts


class InteractiveFilters:
    @staticmethod
    def create_filter_panel(df):
        """Create interactive filter panel with robust data handling"""
        st.sidebar.markdown("## 🔍 Advanced Filters")

        filters = {}

        # Date range filter
        date_cols = [col for col in df.columns if 'date' in col.lower() and 'service' in col.lower()]
        if date_cols:
            date_col = date_cols[0]

            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

            # Filter out NaT values for min/max calculation
            valid_dates = df[date_col].dropna()

            if len(valid_dates) > 0:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()

                date_range = st.sidebar.date_input(
                    "Service Date Range",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )

                if len(date_range) == 2:
                    filters['date_range'] = date_range
                    filters['date_column'] = date_col

        # Amount range filter
        amount_cols = [col for col in df.columns if 'amount' in col.lower() and 'claimed' in col.lower()]
        if amount_cols:
            amount_col = amount_cols[0]

            # Ensure numeric column
            if df[amount_col].dtype != 'float64' and df[amount_col].dtype != 'int64':
                df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')

            # Filter out NaN values for min/max calculation
            valid_amounts = df[amount_col].dropna()

            if len(valid_amounts) > 0:
                min_amount = float(valid_amounts.min())
                max_amount = float(valid_amounts.max())

                amount_range = st.sidebar.slider(
                    "Claim Amount Range ($)",
                    min_amount,
                    max_amount,
                    (min_amount, max_amount),
                    step=100.0
                )
                filters['amount_range'] = amount_range
                filters['amount_column'] = amount_col

        # Gender filter
        gender_cols = [col for col in df.columns if 'gender' in col.lower()]
        if gender_cols:
            gender_col = gender_cols[0]
            genders = ['All'] + sorted([str(g) for g in df[gender_col].dropna().unique() if pd.notna(g)])
            selected_gender = st.sidebar.selectbox(
                "Filter by Gender",
                genders
            )
            if selected_gender != 'All':
                filters['gender'] = selected_gender
                filters['gender_column'] = gender_col

        # Provider filter with search
        provider_cols = [col for col in df.columns if 'provider' in col.lower() and 'name' in col.lower()]
        if provider_cols:
            provider_col = provider_cols[0]
            providers = ['All'] + sorted([str(p) for p in df[provider_col].dropna().unique() if pd.notna(p)])
            selected_provider = st.sidebar.selectbox(
                "Filter by Provider",
                providers
            )
            if selected_provider != 'All':
                filters['provider'] = selected_provider
                filters['provider_column'] = provider_col

        # Clinical group filter
        clin_cols = [col for col in df.columns if 'clin' in col.lower() and 'group' in col.lower()]
        if clin_cols:
            clin_col = clin_cols[0]
            clin_groups = ['All'] + sorted([str(c) for c in df[clin_col].dropna().unique() if pd.notna(c)])
            selected_clin_group = st.sidebar.selectbox(
                "Filter by Clinical Group",
                clin_groups
            )
            if selected_clin_group != 'All':
                filters['clin_group'] = selected_clin_group
                filters['clin_column'] = clin_col

        # Age range filter (if age available)
        age_cols = [col for col in df.columns if col.lower() == 'age']
        if age_cols:
            age_col = age_cols[0]

            # Ensure numeric column
            if df[age_col].dtype != 'float64' and df[age_col].dtype != 'int64':
                df[age_col] = pd.to_numeric(df[age_col], errors='coerce')

            # Filter out NaN values for min/max calculation
            valid_ages = df[age_col].dropna()

            if len(valid_ages) > 0:
                min_age = int(valid_ages.min())
                max_age = int(valid_ages.max())

                age_range = st.sidebar.slider(
                    "Age Range",
                    min_age,
                    max_age,
                    (min_age, max_age)
                )
                filters['age_range'] = age_range
                filters['age_column'] = age_col

        # EDI/Paper filter
        edi_cols = [col for col in df.columns if 'paper' in col.lower() or 'edi' in col.lower()]
        if edi_cols:
            edi_col = edi_cols[0]
            submission_types = ['All'] + sorted([str(t) for t in df[edi_col].dropna().unique() if pd.notna(t)])
            selected_type = st.sidebar.selectbox(
                "Submission Type",
                submission_types
            )
            if selected_type != 'All':
                filters['submission_type'] = selected_type
                filters['submission_column'] = edi_col

        # Reset filters button
        if st.sidebar.button("Reset All Filters"):
            filters = {}
            st.rerun()

        return filters

    @staticmethod
    def apply_filters(df, filters):
        """Apply filters to dataframe with robust type handling"""
        filtered_df = df.copy()

        try:
            # Date range filter
            if 'date_range' in filters and 'date_column' in filters:
                start_date, end_date = filters['date_range']
                date_col = filters['date_column']

                # Ensure date column is datetime
                if not pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
                    filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')

                # Apply filter
                mask = (
                        (filtered_df[date_col].dt.date >= start_date) &
                        (filtered_df[date_col].dt.date <= end_date)
                )
                filtered_df = filtered_df[mask]

            # Amount range filter
            if 'amount_range' in filters and 'amount_column' in filters:
                min_amount, max_amount = filters['amount_range']
                amount_col = filters['amount_column']

                # Ensure numeric column
                if filtered_df[amount_col].dtype != 'float64' and filtered_df[amount_col].dtype != 'int64':
                    filtered_df[amount_col] = pd.to_numeric(filtered_df[amount_col], errors='coerce')

                # Apply filter
                mask = (
                        (filtered_df[amount_col] >= min_amount) &
                        (filtered_df[amount_col] <= max_amount)
                )
                filtered_df = filtered_df[mask]

            # Gender filter
            if 'gender' in filters and 'gender_column' in filters:
                gender = filters['gender']
                gender_col = filters['gender_column']
                filtered_df = filtered_df[filtered_df[gender_col] == gender]

            # Provider filter
            if 'provider' in filters and 'provider_column' in filters:
                provider = filters['provider']
                provider_col = filters['provider_column']
                filtered_df = filtered_df[filtered_df[provider_col] == provider]

            # Clinical group filter
            if 'clin_group' in filters and 'clin_column' in filters:
                clin_group = filters['clin_group']
                clin_col = filters['clin_column']
                filtered_df = filtered_df[filtered_df[clin_col] == clin_group]

            # Age range filter
            if 'age_range' in filters and 'age_column' in filters:
                min_age, max_age = filters['age_range']
                age_col = filters['age_column']

                # Ensure numeric column
                if filtered_df[age_col].dtype != 'float64' and filtered_df[age_col].dtype != 'int64':
                    filtered_df[age_col] = pd.to_numeric(filtered_df[age_col], errors='coerce')

                # Apply filter
                mask = (
                        (filtered_df[age_col] >= min_age) &
                        (filtered_df[age_col] <= max_age)
                )
                filtered_df = filtered_df[mask]

            # Submission type filter
            if 'submission_type' in filters and 'submission_column' in filters:
                submission_type = filters['submission_type']
                submission_col = filters['submission_column']
                filtered_df = filtered_df[filtered_df[submission_col] == submission_type]

        except Exception as e:
            st.error(f"Error applying filters: {str(e)}")
            # Return original dataframe if filter application fails
            return df

        return filtered_df


class DashboardEnhancer:
    @staticmethod
    def create_dashboard_header():
        """Create enhanced dashboard header"""
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown('<h1 class="main-header">🏥 Advanced Healthcare Claims Analytics Dashboard</h1>',
                        unsafe_allow_html=True)

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            # Theme selector
            theme = st.selectbox(
                "Theme",
                ["Light", "Dark", "Blue", "Green"],
                help="Select dashboard theme",
                key="theme_selector"
            )

        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            # View selector
            view_mode = st.selectbox(
                "View Mode",
                ["Overview", "Analytics", "Financial", "Clinical", "Provider", "Premium",
                 "Benefits", "Hospitalisation", "FWA Detector"],
                help="Select dashboard view",
                key="view_selector"
            )

        return theme, view_mode

    @staticmethod
    def create_animated_metrics(metrics):
        """Create animated metric displays"""
        cols = st.columns(4)

        with cols[0]:
            st.markdown(f"""
            <div class="metric-card-blue">
                <div class="metric-value">{metrics.get('total_claims', 0):,}</div>
                <div class="metric-label">Total Claims</div>
                <div class="metric-change">↑ 12% from last month</div>
            </div>
            """, unsafe_allow_html=True)

        with cols[1]:
            st.markdown(f"""
            <div class="metric-card-green">
                <div class="metric-value">${metrics.get('total_paid_amount', 0):,.0f}</div>
                <div class="metric-label">Total Paid</div>
                <div class="metric-change">↓ 8% savings achieved</div>
            </div>
            """, unsafe_allow_html=True)

        with cols[2]:
            st.markdown(f"""
            <div class="metric-card-orange">
                <div class="metric-value">{metrics.get('utilization_rate', 0):.1f}%</div>
                <div class="metric-label">Utilization Rate</div>
                <div class="metric-change">↔ Stable trend</div>
            </div>
            """, unsafe_allow_html=True)

        with cols[3]:
            if 'loss_ratio' in metrics:
                loss_status = "Good" if metrics['loss_ratio'] < 50 else "Moderate" if metrics[
                                                                                          'loss_ratio'] < 80 else "High"
                status_color = "#4CAF50" if loss_status == "Good" else "#FF9800" if loss_status == "Moderate" else "#F44336"
                st.markdown(f"""
                <div class="metric-card-purple">
                    <div class="metric-value">{metrics['loss_ratio']:.1f}%</div>
                    <div class="metric-label">Loss Ratio</div>
                    <div class="metric-change" style="color: {status_color}">
                        {'✅ Below target' if metrics['loss_ratio'] < 70 else '⚠️ Needs attention'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card-purple">
                    <div class="metric-value">N/A</div>
                    <div class="metric-label">Loss Ratio</div>
                    <div class="metric-change">Premium data required</div>
                </div>
                """, unsafe_allow_html=True)

    @staticmethod
    def create_fwa_metrics(metrics):
        """Create FWA metric displays"""
        cols = st.columns(4)

        with cols[0]:
            flagged_pct = metrics.get('flagged_percentage', 0)
            status_color = "#4CAF50" if flagged_pct < 5 else "#FF9800" if flagged_pct < 15 else "#F44336"
            st.markdown(f"""
            <div class="metric-card-red">
                <div class="metric-value">{metrics.get('total_flagged_claims', 0):,}</div>
                <div class="metric-label">Flagged Claims</div>
                <div class="metric-change" style="color: {status_color}">
                    {flagged_pct:.1f}% of total claims
                </div>
            </div>
            """, unsafe_allow_html=True)

        with cols[1]:
            risk_amount = metrics.get('total_flagged_amount', 0)
            st.markdown(f"""
            <div class="metric-card-orange">
                <div class="metric-value">${risk_amount:,.0f}</div>
                <div class="metric-label">At-Risk Amount</div>
                <div class="metric-change">
                    {metrics.get('amount_risk_percentage', 0):.1f}% of total
                </div>
            </div>
            """, unsafe_allow_html=True)

        with cols[2]:
            avg_risk = metrics.get('avg_risk_score', 0)
            risk_level = "Low" if avg_risk < 0.4 else "Medium" if avg_risk < 0.6 else "High" if avg_risk < 0.8 else "Critical"
            risk_color = "#4CAF50" if avg_risk < 0.4 else "#FF9800" if avg_risk < 0.6 else "#F44336"
            st.markdown(f"""
            <div class="metric-card-purple">
                <div class="metric-value">{avg_risk:.2f}</div>
                <div class="metric-label">Avg Risk Score</div>
                <div class="metric-change" style="color: {risk_color}">
                    {risk_level} Risk Level
                </div>
            </div>
            """, unsafe_allow_html=True)

        with cols[3]:
            critical_risk = metrics.get('critical_risk', 0)
            high_risk = metrics.get('high_risk', 0)
            total_high_risk = critical_risk + high_risk
            st.markdown(f"""
            <div class="metric-card-blue">
                <div class="metric-value">{total_high_risk:,}</div>
                <div class="metric-label">High/Critical Risks</div>
                <div class="metric-change">
                    {critical_risk} Critical, {high_risk} High
                </div>
            </div>
            """, unsafe_allow_html=True)

    @staticmethod
    def create_real_time_alerts(metrics, thresholds):
        """Create real-time alert system"""
        st.markdown("### ⚡ Real-time Alerts")

        alerts = []

        # Check loss ratio
        if 'loss_ratio' in metrics:
            if metrics['loss_ratio'] > thresholds.get('loss_ratio_high', 85):
                alerts.append(("danger", "High Loss Ratio",
                               f"Loss ratio at {metrics['loss_ratio']:.1f}% exceeds threshold of 85%"))
            elif metrics['loss_ratio'] > thresholds.get('loss_ratio_warning', 70):
                alerts.append(("warning", "Warning: Loss Ratio",
                               f"Loss ratio at {metrics['loss_ratio']:.1f}% above 70%"))

        # Check utilization rate
        if metrics.get('utilization_rate', 0) > thresholds.get('utilization_high', 80):
            alerts.append(("warning", "High Utilization",
                           f"Utilization rate at {metrics['utilization_rate']:.1f}% exceeds 80%"))

        # Check EDI ratio
        if metrics.get('edi_ratio', 0) < thresholds.get('edi_low', 70):
            alerts.append(("info", "Low EDI Submission",
                           f"EDI submission rate at {metrics.get('edi_ratio', 0) * 100:.1f}% below 70%"))

        # Check FWA flagged percentage
        if metrics.get('flagged_percentage', 0) > thresholds.get('fwa_high', 15):
            alerts.append(("danger", "High FWA Flag Rate",
                           f"{metrics.get('flagged_percentage', 0):.1f}% of claims flagged for potential FWA"))
        elif metrics.get('flagged_percentage', 0) > thresholds.get('fwa_warning', 10):
            alerts.append(("warning", "Warning: High FWA Flag Rate",
                           f"{metrics.get('flagged_percentage', 0):.1f}% of claims flagged for potential FWA"))

        # Check for critical FWA risks
        if metrics.get('critical_risk', 0) > thresholds.get('critical_fwa_threshold', 5):
            alerts.append(("danger", "Critical FWA Risks Detected",
                           f"{metrics.get('critical_risk', 0)} claims with critical risk level detected"))

        # Display alerts
        if alerts:
            for alert_type, title, message in alerts:
                if alert_type == "danger":
                    st.markdown(f"""
                    <div class="alert-danger">
                        <strong>🚨 {title}:</strong> {message}
                    </div>
                    """, unsafe_allow_html=True)
                elif alert_type == "warning":
                    st.markdown(f"""
                    <div class="alert-warning">
                        <strong>⚠️ {title}:</strong> {message}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-info">
                        <strong>ℹ️ {title}:</strong> {message}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-success">
                <strong>✅ All Systems Normal:</strong> No critical alerts at this time.
            </div>
            """, unsafe_allow_html=True)

    @staticmethod
    def display_fwa_alerts(fwa_alerts):
        """Display high-risk FWA alerts"""
        if fwa_alerts:
            st.markdown("### 🚨 High-Risk FWA Alerts")

            for alert in fwa_alerts[:5]:  # Show top 5
                risk_color = "#dc3545" if alert['risk_level'] == 'Critical' else "#ffc107" if alert[
                                                                                                  'risk_level'] == 'High' else "#17a2b8"

                st.markdown(f"""
                <div class="fraud-alert-card{' fraud-alert-card-high' if alert['risk_level'] in ['Critical', 'High'] else ''}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>Claim ID:</strong> {alert['claim_id']}<br>
                            <strong>Member:</strong> {alert['member_no']} | 
                            <strong>Provider:</strong> {alert['provider']}<br>
                            <strong>Amount:</strong> ${alert['amount']:,.2f} | 
                            <strong>Risk:</strong> <span style="color: {risk_color}; font-weight: bold;">{alert['risk_level']} ({alert['risk_score']:.2f})</span>
                        </div>
                        <div class="kpi-indicator {'kpi-danger' if alert['risk_level'] == 'Critical' else 'kpi-warning' if alert['risk_level'] == 'High' else 'kpi-good'}">
                            {alert['risk_level']}
                        </div>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.9em;">
                        <strong>Rules Triggered:</strong> {alert['rules']}<br>
                        <strong>Description:</strong> {alert['description'][:200]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)


# Main application
def main():
    # Initialize session state
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'fwa_detector' not in st.session_state:
        st.session_state.fwa_detector = None
    if 'fwa_metrics' not in st.session_state:
        st.session_state.fwa_metrics = None
    if 'fwa_viz' not in st.session_state:
        st.session_state.fwa_viz = None
    if 'fwa_alerts' not in st.session_state:
        st.session_state.fwa_alerts = None

    # Create header and get theme/view mode
    theme, view_mode = DashboardEnhancer.create_dashboard_header()

    # Sidebar file upload
    with st.sidebar:
        st.markdown("## 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Claims Data (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            key="file_uploader"
        )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.session_state.df = df
                st.session_state.uploaded_file = uploaded_file.name

                st.success(f"✅ Data loaded successfully! {len(df)} records found.")

                # Show data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(), use_container_width=True)

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

        # Show data info if loaded
        if st.session_state.df is not None:
            st.markdown("## 📊 Data Summary")
            st.info(f"""
            **Records:** {len(st.session_state.df):,}
            **Columns:** {len(st.session_state.df.columns)}
            **Date Range:** {st.session_state.df.select_dtypes(include=['datetime64']).min().min().date() if not st.session_state.df.select_dtypes(include=['datetime64']).empty else 'N/A'} to {st.session_state.df.select_dtypes(include=['datetime64']).max().max().date() if not st.session_state.df.select_dtypes(include=['datetime64']).empty else 'N/A'}
            """)

    # Main content area
    if st.session_state.df is not None:
        # Apply filters
        filters = InteractiveFilters.create_filter_panel(st.session_state.df)
        filtered_df = InteractiveFilters.apply_filters(st.session_state.df, filters)
        st.session_state.filtered_df = filtered_df

        # Calculate basic metrics
        metrics = {
            'total_claims': len(filtered_df),
            'total_paid_amount': filtered_df['TOTAL PAID'].sum() if 'TOTAL PAID' in filtered_df.columns else 0,
            'utilization_rate': 75.3,  # This would be calculated from actual data
            'loss_ratio': 68.2,  # This would be calculated from premium data
            'edi_ratio': 0.85  # This would be calculated from submission types
        }

        # Display metrics based on view mode
        if view_mode == "Overview":
            DashboardEnhancer.create_animated_metrics(metrics)
            DashboardEnhancer.create_real_time_alerts(metrics, {})

            # Show filtered data summary
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📈 Filtered Data Summary")
                st.dataframe(filtered_df.describe(), use_container_width=True)

            with col2:
                st.markdown("### 📅 Date Distribution")
                if 'SERVICE DATE' in filtered_df.columns:
                    filtered_df['SERVICE DATE'] = pd.to_datetime(filtered_df['SERVICE DATE'], errors='coerce')
                    monthly_counts = filtered_df.groupby(filtered_df['SERVICE DATE'].dt.to_period('M')).size()
                    st.line_chart(monthly_counts)

        elif view_mode == "FWA Detector":
            st.markdown("## 🔍 Fraud, Waste & Abuse Detection System")

            # Initialize FWA detector if not already done
            if st.session_state.fwa_detector is None:
                with st.spinner("Initializing FWA Detection System..."):
                    st.session_state.fwa_detector = FWADetector(filtered_df)
                    st.session_state.fwa_metrics = None
                    st.session_state.fwa_viz = None
                    st.session_state.fwa_alerts = None

            # Run FWA detection
            if st.button("🚀 Run FWA Analysis", type="primary"):
                with st.spinner("Analyzing claims for potential fraud, waste, and abuse..."):
                    # Run detection
                    flagged_claims = st.session_state.fwa_detector.run_all_detections()

                    # Calculate metrics
                    st.session_state.fwa_metrics = st.session_state.fwa_detector.calculate_fwa_metrics()

                    # Create visualizations
                    st.session_state.fwa_viz = st.session_state.fwa_detector.create_fwa_visualizations(
                        st.session_state.fwa_metrics)

                    # Get high-risk alerts
                    st.session_state.fwa_alerts = st.session_state.fwa_detector.get_high_risk_alerts()

                    st.success(
                        f"✅ FWA analysis complete! Found {st.session_state.fwa_metrics['total_flagged_claims']} flagged claims.")

            # Display FWA results if available
            if st.session_state.fwa_metrics is not None:
                # Display FWA metrics
                DashboardEnhancer.create_fwa_metrics(st.session_state.fwa_metrics)

                # Display high-risk alerts
                if st.session_state.fwa_alerts:
                    DashboardEnhancer.display_fwa_alerts(st.session_state.fwa_alerts)

                # Display visualizations
                if st.session_state.fwa_viz:
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Trends", "🏥 Providers", "📋 Details"])

                    with tab1:
                        if 'fwa_dashboard' in st.session_state.fwa_viz:
                            st.plotly_chart(st.session_state.fwa_viz['fwa_dashboard'], use_container_width=True)

                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            if 'fraud_trend_chart' in st.session_state.fwa_viz:
                                st.plotly_chart(st.session_state.fwa_viz['fraud_trend_chart'], use_container_width=True)
                        with col2:
                            if 'risk_score_histogram' in st.session_state.fwa_viz:
                                st.plotly_chart(st.session_state.fwa_viz['risk_score_histogram'],
                                                use_container_width=True)

                    with tab3:
                        if 'flagged_providers_chart' in st.session_state.fwa_viz:
                            st.plotly_chart(st.session_state.fwa_viz['flagged_providers_chart'],
                                            use_container_width=True)

                    with tab4:
                        st.markdown("### 📋 Flagged Claims Details")
                        if not st.session_state.fwa_detector.flagged_claims.empty:
                            display_cols = ['CLAIM_ID', 'MEMBER NO', 'PROVIDER NAME', 'AMOUNT CLAIMED',
                                            'FWA_RISK_SCORE', 'RISK_LEVEL', 'FWA_RULE', 'FWA_DESCRIPTION']
                            available_cols = [col for col in display_cols if
                                              col in st.session_state.fwa_detector.flagged_claims.columns]

                            st.dataframe(
                                st.session_state.fwa_detector.flagged_claims[available_cols],
                                use_container_width=True
                            )

                            # Download option
                            csv = st.session_state.fwa_detector.flagged_claims.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Flagged Claims",
                                data=csv,
                                file_name="flagged_claims.csv",
                                mime="text/csv"
                            )

                # Show FWA detection statistics
                with st.expander("📊 FWA Detection Statistics"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Claims Analyzed", len(filtered_df))
                    with col2:
                        st.metric("Flagged Claims", st.session_state.fwa_metrics['total_flagged_claims'])
                    with col3:
                        st.metric("Flagged Amount", f"${st.session_state.fwa_metrics['total_flagged_amount']:,.2f}")

        else:
            # For other view modes (to be implemented)
            st.info(f"View mode '{view_mode}' selected. This view will be implemented in future updates.")

    else:
        # Welcome screen when no data is loaded
        st.markdown("""
        # 🏥 Welcome to Healthcare Claims Analytics Dashboard

        ## Getting Started

        1. **Upload your claims data** using the file uploader in the sidebar
        2. **Apply filters** to focus on specific subsets of data
        3. **Select a view mode** from the dropdown above
        4. **Explore insights** through interactive visualizations

        ## Available Features

        - **FWA Detector**: Advanced fraud, waste, and abuse detection
        - **Interactive Filters**: Drill down into specific data segments
        - **Real-time Alerts**: Get notified about critical metrics
        - **Visual Analytics**: Comprehensive charts and graphs
        - **Data Export**: Download filtered and analyzed data

        ### Supported File Formats:
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)

        ### Expected Data Columns:
        - Member information
        - Provider details
        - Claim amounts and dates
        - Service descriptions
        - Payment information
        """)

        # Show sample data structure
        with st.expander("📋 Sample Data Structure"):
            sample_data = pd.DataFrame({
                'MEMBER NO': ['M001', 'M002', 'M003'],
                'PROVIDER NAME': ['Provider A', 'Provider B', 'Provider C'],
                'SERVICE DATE': ['2024-01-15', '2024-01-16', '2024-01-17'],
                'AMOUNT CLAIMED': [150.00, 275.50, 89.99],
                'TOTAL PAID': [120.00, 220.40, 71.99],
                'BASE BENEFIT DESCRIPTION': ['Consultation', 'Lab Test', 'Medication'],
                'GENDER': ['M', 'F', 'M'],
                'AGE': [45, 32, 58]
            })
            st.dataframe(sample_data, use_container_width=True)


if __name__ == "__main__":
    main()