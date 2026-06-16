# Dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
from scipy import stats
from scipy.optimize import curve_fit
import math

# Page configuration
st.set_page_config(
    page_title="Actuarial Dashboard - Operations",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1a237e;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #283593;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1a237e;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a237e;
        color: white;
    }
    .upload-section {
        border: 3px dashed #ddd;
        border-radius: 10px;
        padding: 3rem;
        text-align: center;
        background: #fafafa;
        margin-bottom: 2rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
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

# Initialize session state
if 'claims_data' not in st.session_state:
    st.session_state.claims_data = None
if 'premium_data' not in st.session_state:
    st.session_state.premium_data = None
if 'claims_metrics' not in st.session_state:
    st.session_state.claims_metrics = None
if 'premium_metrics' not in st.session_state:
    st.session_state.premium_metrics = None
if 'claims_file_name' not in st.session_state:
    st.session_state.claims_file_name = None
if 'premium_file_name' not in st.session_state:
    st.session_state.premium_file_name = None
if 'ibnr_estimate' not in st.session_state:
    st.session_state.ibnr_estimate = None
if 'ibnr_method' not in st.session_state:
    st.session_state.ibnr_method = None
if 'development_triangle' not in st.session_state:
    st.session_state.development_triangle = None
if 'ibnr_results' not in st.session_state:
    st.session_state.ibnr_results = None


# Helper functions for claims data (existing)
def clean_column_names(df):
    """Clean and standardize column names"""
    cleaned_columns = []
    for col in df.columns:
        col_str = str(col)
        cleaned = col_str.strip().upper().replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')
        cleaned_columns.append(cleaned)
    df.columns = cleaned_columns
    return df


def parse_claims_file(file):
    """Parse uploaded claims Excel file"""
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


# Helper functions for premium data
def parse_premium_file(file):
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


def calculate_premium_metrics(df):
    """Calculate premium metrics"""
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


def calculate_claims_metrics(df):
    """Calculate claims metrics"""
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


# EXCEL CHAIN LADDER MODEL FUNCTIONS
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


# IBNR Calculation Functions - UPDATED to fix the date error
def calculate_development_triangle(df, accident_period='M', development_period='M'):
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


def calculate_ibnr_chain_ladder(cum_triangle):
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


def calculate_ibnr_bornhuetter_ferguson(df, expected_loss_ratio=0.75):
    """Calculate IBNR using Bornhuetter-Ferguson method"""
    if df is None or df.empty or 'AMOUNT_CLAIMED' not in df.columns:
        return 0, {}

    try:
        # Calculate earned premium (simplified - use average premium per member * number of claims)
        if st.session_state.premium_metrics and 'avg_premium' in st.session_state.premium_metrics:
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


def calculate_ibnr_frequency_severity(df):
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
            lambda_freq = monthly_counts.mean()

            # Calculate severity distribution
            severities = df['AMOUNT_CLAIMED'].values
            avg_severity = np.mean(severities)
            std_severity = np.std(severities)

            # Estimate IBNR
            months_unreported = 3  # Assume 3 months of unreported claims
            expected_unreported_claims = lambda_freq * months_unreported
            ibnr = expected_unreported_claims * avg_severity

            return ibnr, {
                'method': 'Frequency-Severity',
                'lambda_frequency': lambda_freq,
                'avg_severity': avg_severity,
                'std_severity': std_severity,
                'expected_unreported_claims': expected_unreported_claims,
                'months_unreported': months_unreported
            }
        else:
            return 0, {}

    except Exception as e:
        st.error(f"Error in Frequency-Severity calculation: {str(e)}")
        return 0, {}


def calculate_ibnr_bootstrap(df, n_simulations=1000):
    """Calculate IBNR using Bootstrap method with confidence intervals"""
    if df is None or df.empty:
        return 0, {}

    try:
        # Get development triangle
        inc_triangle, cum_triangle, _ = calculate_development_triangle(df)

        if cum_triangle is None:
            return 0, {}

        # Simple bootstrap simulation
        ibnr_simulations = []
        n_accident = len(cum_triangle.index)
        n_development = len(cum_triangle.columns)

        for _ in range(n_simulations):
            # Resample development factors
            sampled_factors = []
            for i in range(n_development - 1):
                current_col = cum_triangle.iloc[:, i]
                next_col = cum_triangle.iloc[:, i + 1]

                # Bootstrap ratio
                ratios = []
                for j in range(len(current_col)):
                    if current_col.iloc[j] > 0:
                        ratio = next_col.iloc[j] / current_col.iloc[j]
                        ratios.append(ratio)

                if ratios:
                    sampled_factor = np.random.choice(ratios)
                    sampled_factors.append(sampled_factor)
                else:
                    sampled_factors.append(1.0)

            # Calculate ultimate for each accident period
            ultimate_total = 0
            for i in range(n_accident):
                current_value = cum_triangle.iloc[i, -1]
                remaining_factors = 1.0

                # Apply remaining development factors
                for j in range(n_accident - i - 1):
                    if j < len(sampled_factors):
                        remaining_factors *= sampled_factors[j]

                ultimate_total += current_value * remaining_factors

            reported_total = cum_triangle.iloc[:, -1].sum()
            ibnr_sim = ultimate_total - reported_total
            ibnr_simulations.append(ibnr_sim)

        # Calculate statistics
        ibnr_mean = np.mean(ibnr_simulations)
        ibnr_std = np.std(ibnr_simulations)
        ibnr_median = np.median(ibnr_simulations)

        # Calculate confidence intervals
        confidence_95 = np.percentile(ibnr_simulations, [2.5, 97.5])
        confidence_90 = np.percentile(ibnr_simulations, [5, 95])

        return ibnr_mean, {
            'method': 'Bootstrap Simulation',
            'n_simulations': n_simulations,
            'mean': ibnr_mean,
            'median': ibnr_median,
            'std_dev': ibnr_std,
            'confidence_95': confidence_95,
            'confidence_90': confidence_90,
            'simulations': ibnr_simulations
        }

    except Exception as e:
        st.error(f"Error in Bootstrap calculation: {str(e)}")
        return 0, {}


def calculate_composite_ibnr(df, methods=['chain_ladder', 'bornhuetter', 'frequency', 'excel']):
    """Calculate composite IBNR using multiple methods including Excel model"""
    results = {}

    # Chain Ladder
    if 'chain_ladder' in methods:
        inc_triangle, cum_triangle, _ = calculate_development_triangle(df)
        if cum_triangle is not None and not cum_triangle.empty:
            ibnr_cl, details_cl = calculate_ibnr_chain_ladder(cum_triangle)
            results['Chain Ladder'] = {
                'ibnr': ibnr_cl,
                'details': details_cl
            }

    # Bornhuetter-Ferguson
    if 'bornhuetter' in methods:
        ibnr_bf, details_bf = calculate_ibnr_bornhuetter_ferguson(df)
        results['Bornhuetter-Ferguson'] = {
            'ibnr': ibnr_bf,
            'details': details_bf
        }

    # Frequency-Severity
    if 'frequency' in methods:
        ibnr_fs, details_fs = calculate_ibnr_frequency_severity(df)
        results['Frequency-Severity'] = {
            'ibnr': ibnr_fs,
            'details': details_fs
        }

    # Bootstrap
    if 'bootstrap' in methods:
        ibnr_bs, details_bs = calculate_ibnr_bootstrap(df)
        results['Bootstrap'] = {
            'ibnr': ibnr_bs,
            'details': details_bs
        }

    # Excel Chain Ladder Model
    if 'excel' in methods:
        ibnr_excel, details_excel = calculate_ibnr_excel_chain_ladder(df)
        results['Excel Chain Ladder'] = {
            'ibnr': ibnr_excel,
            'details': details_excel
        }

    if not results:
        return 0, {}

    # Calculate weighted average
    weights = {
        'Excel Chain Ladder': 0.40,  # Give more weight to Excel model
        'Chain Ladder': 0.25,
        'Bornhuetter-Ferguson': 0.20,
        'Frequency-Severity': 0.10,
        'Bootstrap': 0.05
    }

    composite_ibnr = 0
    valid_methods = 0

    for method, data in results.items():
        if data['ibnr'] > 0:
            composite_ibnr += data['ibnr'] * weights.get(method, 0.20)
            valid_methods += 1

    if valid_methods > 0:
        # Adjust for number of valid methods
        composite_ibnr = composite_ibnr * (valid_methods / len(methods))
    else:
        composite_ibnr = 0

    return composite_ibnr, {
        'method': 'Composite (Weighted Average)',
        'component_results': results,
        'weights': weights
    }


# Chart functions for premium data
def create_premium_product_chart(product_data):
    """Create premium by product chart"""
    if product_data.empty or 'PRODUCT_NAME' not in product_data.columns:
        return None

    fig = px.bar(
        product_data.head(10),
        x='PRODUCT_NAME',
        y='PREMIUM_AMOUNT',
        title="Premium by Product",
        labels={'PRODUCT_NAME': 'Product', 'PREMIUM_AMOUNT': 'Premium Amount ($)'},
        color='PREMIUM_AMOUNT',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(height=400, template="plotly_white")
    return fig


def create_premium_payer_chart(payer_data):
    """Create premium by payer chart"""
    if payer_data.empty or 'PAYER_NAME' not in payer_data.columns:
        return None

    fig = px.pie(
        payer_data.head(10),
        values='PREMIUM_AMOUNT',
        names='PAYER_NAME',
        title="Premium Distribution by Payer",
        hole=0.4
    )

    fig.update_layout(height=400, template="plotly_white")
    return fig


def create_premium_trend_chart(monthly_data):
    """Create premium trend chart"""
    if monthly_data.empty or 'join_month' not in monthly_data.columns:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_data['join_month'],
        y=monthly_data['PREMIUM_AMOUNT'],
        name='Monthly Premium',
        line=dict(color='#764ba2', width=3),
        fill='tozeroy',
        fillcolor='rgba(118, 75, 162, 0.1)'
    ))

    fig.update_layout(
        title="Premium Enrollment Trends",
        xaxis_title="Month",
        yaxis_title="Premium Amount ($)",
        template="plotly_white",
        hovermode="x unified",
        height=400
    )

    return fig


def create_premium_age_chart(age_data):
    """Create premium age distribution chart"""
    if age_data.empty or 'age_group' not in age_data.columns:
        return None

    fig = px.bar(
        age_data,
        x='age_group',
        y='PREMIUM_AMOUNT',
        title="Premium by Age Group",
        labels={'age_group': 'Age Group', 'PREMIUM_AMOUNT': 'Total Premium ($)'},
        color='member_count',
        color_continuous_scale='Blues'
    )

    fig.update_layout(height=400, template="plotly_white")
    return fig


# Chart functions for claims data
def create_monthly_trend_chart(monthly_data):
    """Create monthly trend chart for claims"""
    if monthly_data.empty or 'month_year' not in monthly_data.columns:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_data['month_year'],
        y=monthly_data['AMOUNT_CLAIMED'],
        name='Amount Claimed',
        line=dict(color='#1a237e', width=3),
        fill='tozeroy',
        fillcolor='rgba(26, 35, 126, 0.1)'
    ))

    if 'TOTAL_PAID' in monthly_data.columns:
        fig.add_trace(go.Scatter(
            x=monthly_data['month_year'],
            y=monthly_data['TOTAL_PAID'],
            name='Amount Paid',
            line=dict(color='#4caf50', width=3),
            fill='tozeroy',
            fillcolor='rgba(76, 175, 80, 0.1)'
        ))

    fig.update_layout(
        title="Monthly Claims Trend",
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        template="plotly_white",
        hovermode="x unified",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_provider_chart(provider_data):
    """Create provider analysis chart for claims"""
    if provider_data.empty or 'PROVIDER_NAME' not in provider_data.columns:
        return None

    # Truncate long provider names for better display
    provider_data = provider_data.copy()
    provider_data['PROVIDER_NAME_SHORT'] = provider_data['PROVIDER_NAME'].apply(
        lambda x: x[:30] + '...' if len(x) > 30 else x
    )

    fig = px.bar(
        provider_data.head(10),
        y='PROVIDER_NAME_SHORT',
        x='AMOUNT_CLAIMED',
        orientation='h',
        title="Top Providers by Claims Amount",
        labels={'PROVIDER_NAME_SHORT': 'Provider', 'AMOUNT_CLAIMED': 'Amount Claimed ($)'},
        color='AMOUNT_CLAIMED',
        color_continuous_scale='Blues',
        hover_data=['PROVIDER_NAME']
    )

    fig.update_layout(
        height=500,
        template="plotly_white",
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


def create_age_distribution_chart(age_data):
    """Create age distribution chart for claims"""
    if age_data.empty or 'age_group' not in age_data.columns:
        return None

    fig = px.pie(
        age_data,
        values='count',
        names='age_group',
        title="Age Distribution of Members",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Blues_r
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400, template="plotly_white")

    return fig


def create_service_type_chart(service_data):
    """Create service type chart for claims"""
    if service_data.empty or 'BASE_BENEFIT_DESCRIPTION' not in service_data.columns:
        return None

    # Truncate long descriptions
    service_data = service_data.copy()
    service_data['SERVICE_SHORT'] = service_data['BASE_BENEFIT_DESCRIPTION'].apply(
        lambda x: x[:40] + '...' if len(x) > 40 else x
    )

    fig = px.treemap(
        service_data.head(10),
        path=['SERVICE_SHORT'],
        values='AMOUNT_CLAIMED',
        title="Service Type Breakdown",
        color='AMOUNT_CLAIMED',
        color_continuous_scale='Blues',
        hover_data=['BASE_BENEFIT_DESCRIPTION']
    )

    fig.update_layout(height=500, template="plotly_white")

    return fig


def create_financial_kpi_chart(metrics):
    """Create financial KPI chart for claims"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Paid Ratio', 'Average Claim', 'Rejection Rate'),
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
    )

    # Paid Ratio
    paid_ratio = metrics.get('paid_ratio', 0) * 100
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=paid_ratio,
            title={'text': "Paid Ratio (%)"},
            domain={'row': 0, 'column': 0},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1a237e"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "darkgray"}
                ]
            }
        ),
        row=1, col=1
    )

    # Average Claim
    avg_claim = metrics.get('avg_claim_amount', 0)
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=avg_claim,
            title={'text': "Average Claim ($)"},
            domain={'row': 0, 'column': 1},
            number={'prefix': "$", 'valueformat': '.2f'}
        ),
        row=1, col=2
    )

    # Rejection Rate
    rejection_rate = metrics.get('rejection_rate', 0) * 100
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=rejection_rate,
            title={'text': "Rejection Rate (%)"},
            domain={'row': 0, 'column': 2},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#f44336"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 40], 'color': "yellow"},
                    {'range': [40, 100], 'color': "lightcoral"}
                ]
            }
        ),
        row=1, col=3
    )

    fig.update_layout(height=300, template="plotly_white")

    return fig


# IBNR Chart Functions
def create_ibnr_comparison_chart(ibnr_results):
    """Create comparison chart of different IBNR methods"""
    if not ibnr_results or ibnr_results is None:
        return None

    methods = []
    values = []

    for method, data in ibnr_results.items():
        if data and 'ibnr' in data:
            methods.append(method)
            values.append(data['ibnr'])

    if not methods:
        return None

    fig = px.bar(
        x=methods,
        y=values,
        title="IBNR Estimates by Method",
        labels={'x': 'Method', 'y': 'IBNR Amount ($)'},
        color=values,
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        height=400,
        template="plotly_white",
        xaxis_tickangle=-45
    )

    return fig


def create_development_triangle_chart(cum_triangle):
    """Create heatmap of development triangle"""
    if cum_triangle is None or cum_triangle.empty:
        return None

    # Convert to dataframe for plotting
    plot_data = cum_triangle.copy()
    plot_data.index = plot_data.index.astype(str)

    fig = px.imshow(
        plot_data,
        title="Claims Development Triangle",
        labels=dict(x="Development Period", y="Accident Period", color="Cumulative Claims"),
        aspect="auto",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        height=500,
        template="plotly_white",
        xaxis_title="Development Lag (Months)",
        yaxis_title="Accident Period"
    )

    return fig


def create_ibnr_confidence_chart(bootstrap_details):
    """Create confidence interval chart for bootstrap method"""
    if not bootstrap_details or 'simulations' not in bootstrap_details:
        return None

    simulations = bootstrap_details['simulations']

    fig = go.Figure()

    # Histogram of simulations
    fig.add_trace(go.Histogram(
        x=simulations,
        nbinsx=50,
        name='IBNR Distribution',
        marker_color='#1a237e',
        opacity=0.7
    ))

    # Add mean line
    mean_val = bootstrap_details.get('mean', 0)
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: ${mean_val:,.0f}",
        annotation_position="top right"
    )

    # Add confidence intervals
    conf_95 = bootstrap_details.get('confidence_95', [0, 0])
    conf_90 = bootstrap_details.get('confidence_90', [0, 0])

    fig.add_vrect(
        x0=conf_95[0], x1=conf_95[1],
        fillcolor="rgba(255, 0, 0, 0.1)",
        line_width=0,
        annotation_text="95% CI",
        annotation_position="top left"
    )

    fig.add_vrect(
        x0=conf_90[0], x1=conf_90[1],
        fillcolor="rgba(0, 255, 0, 0.1)",
        line_width=0,
        annotation_text="90% CI",
        annotation_position="top right"
    )

    fig.update_layout(
        title="Bootstrap IBNR Distribution with Confidence Intervals",
        xaxis_title="IBNR Amount ($)",
        yaxis_title="Frequency",
        height=400,
        template="plotly_white",
        showlegend=False
    )

    return fig


# Excel Chain Ladder specific charts
def create_excel_chain_ladder_chart(ibnr_details):
    """Create visualization for Excel Chain Ladder model"""
    if not ibnr_details or 'cumulative_triangle' not in ibnr_details:
        return None

    cum_triangle = ibnr_details['cumulative_triangle']

    # Create heatmap of cumulative triangle
    fig = px.imshow(
        cum_triangle,
        title="Excel Chain Ladder - Cumulative Claims Triangle",
        labels=dict(x="Development Lag (Months)", y="Accident Period", color="Cumulative Claims"),
        aspect="auto",
        color_continuous_scale="Blues",
        text_auto=False
    )

    fig.update_layout(
        height=500,
        template="plotly_white",
        xaxis_title="Development Lag (Months)",
        yaxis_title="Accident Period (Year-Month)"
    )

    return fig


def create_development_factors_chart(development_factors):
    """Create bar chart of development factors"""
    if not development_factors:
        return None

    # Convert to DataFrame for plotting
    factors_df = pd.DataFrame({
        'Transition': list(development_factors.keys()),
        'Factor': list(development_factors.values())
    })

    fig = px.bar(
        factors_df,
        x='Transition',
        y='Factor',
        title="Development Factors (Age-to-Age)",
        labels={'Transition': 'Lag Transition', 'Factor': 'Development Factor'},
        color='Factor',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        height=400,
        template="plotly_white",
        xaxis_tickangle=-45
    )

    # Add horizontal line at 1.0
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="No Development")

    return fig


# Sidebar
with st.sidebar:
    st.markdown("## 📊 Actuarial Dashboard")
    st.markdown("### Operations Department")
    st.markdown("---")

    # Claims file uploader
    st.markdown("### 📁 Upload Claims Data")
    claims_file = st.file_uploader(
        "Choose claims Excel file",
        type=['xlsx', 'xls', 'csv'],
        key="claims_uploader",
        help="Upload Excel file containing claims data"
    )

    if claims_file is not None:
        if st.button("Process Claims", type="primary", use_container_width=True):
            with st.spinner("Processing claims file..."):
                st.session_state.claims_data = None
                st.session_state.claims_metrics = None
                st.session_state.ibnr_estimate = None
                st.session_state.ibnr_results = None

                if claims_file.name.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(claims_file)
                        df = clean_column_names(df)
                        st.session_state.claims_data = df
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                else:
                    df = parse_claims_file(claims_file)
                    if df is not None:
                        st.session_state.claims_data = df

                if st.session_state.claims_data is not None:
                    st.session_state.claims_metrics = calculate_claims_metrics(st.session_state.claims_data)
                    st.session_state.claims_file_name = claims_file.name
                    st.success("Claims file processed!")

    # Premium file uploader
    st.markdown("### 💰 Upload Premium Data")
    premium_file = st.file_uploader(
        "Choose premium Excel file",
        type=['xlsx', 'xls', 'csv'],
        key="premium_uploader",
        help="Upload Excel file containing premium data"
    )

    if premium_file is not None:
        if st.button("Process Premium", type="secondary", use_container_width=True):
            with st.spinner("Processing premium file..."):
                st.session_state.premium_data = None
                st.session_state.premium_metrics = None

                if premium_file.name.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(premium_file)
                        df = clean_column_names(df)
                        st.session_state.premium_data = df
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                else:
                    df = parse_premium_file(premium_file)
                    if df is not None:
                        st.session_state.premium_data = df

                if st.session_state.premium_data is not None:
                    st.session_state.premium_metrics = calculate_premium_metrics(st.session_state.premium_data)
                    st.session_state.premium_file_name = premium_file.name
                    st.success("Premium file processed!")

    st.markdown("---")

    # Data info
    data_sections = []

    if st.session_state.claims_data is not None:
        data_sections.append(f"""
        **Claims File:** {st.session_state.claims_file_name}
        **Claims Records:** {len(st.session_state.claims_data):,}
        **Unique Members (Claims):** {st.session_state.claims_metrics.get('unique_members', 0):,}
        """)

    if st.session_state.premium_data is not None:
        data_sections.append(f"""
        **Premium File:** {st.session_state.premium_file_name}
        **Premium Records:** {len(st.session_state.premium_data):,}
        **Unique Members (Premium):** {st.session_state.premium_metrics.get('unique_members', 0):,}
        **Total Premium:** ${st.session_state.premium_metrics.get('total_premium', 0):,.2f}
        """)

    if data_sections:
        st.markdown("### 📋 Data Summary")
        st.info("\n".join(data_sections))

        # Combined analysis if both datasets available
        if st.session_state.claims_data is not None and st.session_state.premium_data is not None:
            st.markdown("---")
            st.markdown("### 🔗 Combined Analysis")

            # Calculate combined metrics
            total_claims = st.session_state.claims_metrics.get('total_claimed', 0)
            total_premium = st.session_state.premium_metrics.get('total_premium', 0)

            if total_premium > 0:
                loss_ratio = (total_claims / total_premium) * 100
                st.metric("Loss Ratio", f"{loss_ratio:.1f}%")

            # Common members
            if 'MEMBER_NO' in st.session_state.claims_data.columns and 'MEMBER_NO' in st.session_state.premium_data.columns:
                claims_members = set(st.session_state.claims_data['MEMBER_NO'].unique())
                premium_members = set(st.session_state.premium_data['MEMBER_NO'].unique())
                common_members = len(claims_members.intersection(premium_members))

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Common Members", common_members)
                with col2:
                    coverage = (common_members / len(premium_members) * 100) if len(premium_members) > 0 else 0
                    st.metric("Coverage", f"{coverage:.1f}%")

        # IBNR estimate if available
        if st.session_state.ibnr_estimate is not None:
            st.markdown("---")
            st.markdown("### 📈 IBNR Estimate")
            ibnr_value = st.session_state.ibnr_estimate
            st.metric("Estimated IBNR", f"${ibnr_value:,.0f}")

    st.markdown("---")
    st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Main content
if st.session_state.claims_data is None and st.session_state.premium_data is None:
    # Upload page
    st.markdown('<div class="main-header">Actuarial Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Health Insurance Analysis Platform</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Claims Data")
        st.markdown("Analyze claims patterns and utilization")
        st.markdown("")
        st.markdown("**Expected columns:**")
        st.markdown("- Member Number")
        st.markdown("- Claim Amount")
        st.markdown("- Paid Amount")
        st.markdown("- Service Date")
        st.markdown("- Provider Name")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="upload-section" style="background: #f0f7ff;">', unsafe_allow_html=True)
        st.markdown("### 💰 Upload Premium Data")
        st.markdown("Analyze premium collection and membership")
        st.markdown("")
        st.markdown("**Expected columns:**")
        st.markdown("- Member Number")
        st.markdown("- Premium Amount")
        st.markdown("- Product Name")
        st.markdown("- Payer Name")
        st.markdown("- Join Date")
        st.markdown('</div>', unsafe_allow_html=True)

    # Sample data preview
    with st.expander("📋 Sample Premium Data Format", expanded=True):
        sample_data = {
            'Member': [13574465, 13574474, 13574492],
            'Name': ['MR CHAMUNORWA NYAMAKURA', 'MRS RACHAEL MUTENDI', 'MR MUNYARADZI MOYO'],
            'BirthDate': ['1953/07/09', '1963/03/11', '1998/07/07'],
            'Payer Name': ['SELF PAYING', 'SELF PAYING', 'SELF PAYING'],
            'Join Date': ['2025/02/01', '2025/02/01', '2025/02/01'],
            'Product Name': ['GENLINK', 'GENCORE', 'GENCARE'],
            'Premium': [121, 16, 69]
        }
        st.dataframe(pd.DataFrame(sample_data))

else:
    # Create tabs based on available data
    tab_names = []

    if st.session_state.claims_data is not None:
        tab_names.extend(
            ["📈 Claims Overview", "💰 Claims Financial", "👥 Claims Members", "🏥 Providers", "📊 IBNR Analysis"])

    if st.session_state.premium_data is not None:
        tab_names.extend(["💎 Premium Analysis", "📊 Premium Details"])

    if st.session_state.claims_data is not None and st.session_state.premium_data is not None:
        tab_names.append("🔗 Combined View")

    tab_names.append("📁 Data Export")

    tabs = st.tabs(tab_names)

    # Claims Overview Tab
    if "📈 Claims Overview" in tab_names:
        claims_overview_index = tab_names.index("📈 Claims Overview")
        with tabs[claims_overview_index]:
            st.markdown('<div class="main-header">Claims Overview</div>', unsafe_allow_html=True)
            metrics = st.session_state.claims_metrics

            # Key metrics row 1
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total Claims</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{metrics.get("total_claims", 0):,}</div>',
                            unsafe_allow_html=True)
                st.caption("Number of claim records")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total Claimed</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${metrics.get("total_claimed", 0):,.2f}</div>',
                            unsafe_allow_html=True)
                st.caption("Total amount claimed")
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total Paid</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${metrics.get("total_paid", 0):,.2f}</div>',
                            unsafe_allow_html=True)
                st.caption("Total amount paid")
                st.markdown('</div>', unsafe_allow_html=True)

            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Paid Ratio</div>', unsafe_allow_html=True)
                paid_ratio = metrics.get("paid_ratio", 0) * 100
                st.markdown(f'<div class="metric-value">{paid_ratio:.1f}%</div>', unsafe_allow_html=True)
                st.caption("Paid / Claimed ratio")
                st.markdown('</div>', unsafe_allow_html=True)

            # Key metrics row 2 (with IBNR)
            col5, col6, col7, col8 = st.columns(4)

            with col5:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Unique Members</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{metrics.get("unique_members", 0):,}</div>',
                            unsafe_allow_html=True)
                st.caption("Number of unique members")
                st.markdown('</div>', unsafe_allow_html=True)

            with col6:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Avg Claim</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${metrics.get("avg_claim_amount", 0):,.2f}</div>',
                            unsafe_allow_html=True)
                st.caption("Average claim amount")
                st.markdown('</div>', unsafe_allow_html=True)

            with col7:
                ibnr_value = st.session_state.ibnr_estimate or 0
                st.markdown('<div class="ibnr-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label" style="color: white;">Estimated IBNR</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color: white;">${ibnr_value:,.0f}</div>',
                            unsafe_allow_html=True)
                st.caption("Incurred But Not Reported")
                st.markdown('</div>', unsafe_allow_html=True)

            with col8:
                total_claims = metrics.get("total_claimed", 0)
                ibnr_ratio = (ibnr_value / total_claims * 100) if total_claims > 0 else 0
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">IBNR Ratio</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{ibnr_ratio:.1f}%</div>', unsafe_allow_html=True)
                st.caption("IBNR/Reported Claims")
                st.markdown('</div>', unsafe_allow_html=True)

            # Charts
            if 'monthly_trends' in metrics and not metrics['monthly_trends'].empty:
                fig = create_monthly_trend_chart(metrics['monthly_trends'])
                st.plotly_chart(fig, use_container_width=True)

    # Claims Financial Tab
    if "💰 Claims Financial" in tab_names:
        claims_financial_index = tab_names.index("💰 Claims Financial")
        with tabs[claims_financial_index]:
            st.markdown('<div class="main-header">Claims Financial Analysis</div>', unsafe_allow_html=True)
            metrics = st.session_state.claims_metrics

            # Financial KPIs
            st.markdown('<div class="sub-header">Key Financial Indicators</div>', unsafe_allow_html=True)
            fig = create_financial_kpi_chart(metrics)
            st.plotly_chart(fig, use_container_width=True)

            # Financial details
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="sub-header">Payment Sources Analysis</div>', unsafe_allow_html=True)

                # Check for payment source columns
                payment_sources_data = []

                # Check various payment source columns
                df = st.session_state.claims_data

                # Look for common payment source column names
                payment_columns = {
                    'PAID_FROM_RISK_AMT': 'Risk Pool',
                    'PAID_FROM_THRESHHOLD': 'Threshold',
                    'PAID_FROM_SAVINGS': 'Savings',
                    'PAID_FROM_ACCUMULATOR': 'Accumulator',
                    'COPAY': 'Co-payment',
                    'DEDUCTIBLE': 'Deductible'
                }

                for col, source_name in payment_columns.items():
                    if col in df.columns:
                        try:
                            amount = float(df[col].sum())
                            if amount > 0:
                                payment_sources_data.append({
                                    'Source': source_name,
                                    'Amount': amount,
                                    'Percentage': (amount / metrics.get('total_paid', 1) * 100) if metrics.get(
                                        'total_paid', 0) > 0 else 0
                                })
                        except:
                            pass

                if payment_sources_data:
                    payment_df = pd.DataFrame(payment_sources_data)
                    fig = px.pie(
                        payment_df,
                        values='Amount',
                        names='Source',
                        title="Payment Sources Distribution",
                        color_discrete_sequence=px.colors.sequential.Blues,
                        hover_data=['Percentage']
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)

                    # Payment source table
                    st.markdown("**Payment Source Details:**")
                    payment_df['Amount'] = payment_df['Amount'].apply(lambda x: f"${x:,.2f}")
                    payment_df['Percentage'] = payment_df['Percentage'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(payment_df[['Source', 'Amount', 'Percentage']], use_container_width=True)
                else:
                    st.info(
                        "Payment source data not available. Add columns like PAID_FROM_RISK_AMT, PAID_FROM_THRESHHOLD, etc.")

            with col2:
                st.markdown('<div class="sub-header">Top Claims by Amount</div>', unsafe_allow_html=True)

                if 'AMOUNT_CLAIMED' in df.columns and 'TOTAL_PAID' in df.columns:
                    try:
                        # Get top 10 claims
                        top_claims = df.nlargest(10, 'AMOUNT_CLAIMED')

                        # Select available columns for display
                        display_columns = []
                        column_mapping = {
                            'MEMBER_NO': 'Member ID',
                            'SERVICE_DATE': 'Service Date',
                            'AMOUNT_CLAIMED': 'Amount Claimed ($)',
                            'TOTAL_PAID': 'Amount Paid ($)',
                            'PROVIDER_NAME': 'Provider',
                            'BASE_BENEFIT_DESCRIPTION': 'Service Type'
                        }

                        for col, display_name in column_mapping.items():
                            if col in top_claims.columns:
                                display_columns.append(col)

                        if display_columns:
                            # Create display dataframe with unique column names
                            top_claims_display = top_claims[display_columns].copy()

                            # Apply unique display names
                            rename_dict = {}
                            for col in display_columns:
                                if col in column_mapping:
                                    rename_dict[col] = column_mapping[col]

                            # Rename columns
                            top_claims_display = top_claims_display.rename(columns=rename_dict)

                            # Calculate Paid % as a separate column
                            if 'Amount Claimed ($)' in top_claims_display.columns and 'Amount Paid ($)' in top_claims_display.columns:
                                # Get original values for calculation
                                original_claimed = top_claims['AMOUNT_CLAIMED'].values
                                original_paid = top_claims['TOTAL_PAID'].values

                                # Calculate paid percentage
                                paid_percentages = []
                                for i in range(len(original_claimed)):
                                    if original_claimed[i] > 0:
                                        paid_percentages.append(
                                            f"{(original_paid[i] / original_claimed[i] * 100):.1f}%")
                                    else:
                                        paid_percentages.append("0.0%")

                                # Add Paid % column
                                top_claims_display['Paid %'] = paid_percentages

                            # Format currency columns
                            if 'Amount Claimed ($)' in top_claims_display.columns:
                                top_claims_display['Amount Claimed ($)'] = top_claims_display[
                                    'Amount Claimed ($)'].apply(
                                    lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x
                                )

                            if 'Amount Paid ($)' in top_claims_display.columns:
                                top_claims_display['Amount Paid ($)'] = top_claims_display['Amount Paid ($)'].apply(
                                    lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x
                                )

                            # Format date column if present
                            if 'Service Date' in top_claims_display.columns:
                                top_claims_display['Service Date'] = top_claims_display['Service Date'].apply(
                                    lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else ''
                                )

                            st.dataframe(
                                top_claims_display,
                                use_container_width=True,
                                height=400
                            )
                        else:
                            st.info("No claim data available for display")
                    except Exception as e:
                        st.error(f"Error displaying top claims: {str(e)}")
                        st.info("Try checking the data format and column names")
                else:
                    st.info("Claim amount data not available")

            # Monthly financial analysis
            st.markdown('<div class="sub-header">Monthly Financial Summary</div>', unsafe_allow_html=True)

            if 'monthly_trends' in metrics and not metrics['monthly_trends'].empty:
                monthly_financial = metrics['monthly_trends'].copy()

                # Format for display
                monthly_financial_display = monthly_financial.copy()
                monthly_financial_display['AMOUNT_CLAIMED'] = monthly_financial_display['AMOUNT_CLAIMED'].apply(
                    lambda x: f"${x:,.2f}")
                monthly_financial_display['TOTAL_PAID'] = monthly_financial_display['TOTAL_PAID'].apply(
                    lambda x: f"${x:,.2f}")

                if 'AMOUNT_CLAIMED' in monthly_financial.columns and 'TOTAL_PAID' in monthly_financial.columns:
                    monthly_financial_display['Paid %'] = (
                            (monthly_financial['TOTAL_PAID'] / monthly_financial['AMOUNT_CLAIMED']) * 100).round(
                        1).apply(lambda x: f"{x:.1f}%")
                    monthly_financial_display['Variance'] = (
                            monthly_financial['AMOUNT_CLAIMED'] - monthly_financial['TOTAL_PAID']).apply(
                        lambda x: f"${x:,.2f}")

                monthly_financial_display = monthly_financial_display.rename(columns={
                    'month_year': 'Month',
                    'AMOUNT_CLAIMED': 'Amount Claimed',
                    'TOTAL_PAID': 'Amount Paid',
                    'MEMBER_NO': 'Unique Members'
                })

                st.dataframe(
                    monthly_financial_display,
                    use_container_width=True
                )

                # Monthly financial chart
                if len(monthly_financial) > 1:
                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        x=monthly_financial['month_year'],
                        y=monthly_financial['AMOUNT_CLAIMED'],
                        name='Amount Claimed',
                        marker_color='#1a237e'
                    ))

                    fig.add_trace(go.Bar(
                        x=monthly_financial['month_year'],
                        y=monthly_financial['TOTAL_PAID'],
                        name='Amount Paid',
                        marker_color='#4caf50'
                    ))

                    fig.update_layout(
                        title="Monthly Claims vs Paid Amounts",
                        xaxis_title="Month",
                        yaxis_title="Amount ($)",
                        barmode='group',
                        template="plotly_white",
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Monthly financial data not available")

    # Claims Members Tab
    if "👥 Claims Members" in tab_names:
        claims_members_index = tab_names.index("👥 Claims Members")
        with tabs[claims_members_index]:
            st.markdown('<div class="main-header">Member Analytics</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="sub-header">Member Demographics</div>', unsafe_allow_html=True)

                # Check for age data
                df = st.session_state.claims_data
                if 'CURRENT_AGE' in df.columns:
                    try:
                        # Create age groups
                        df_temp = df.copy()
                        df_temp['AGE_GROUP'] = pd.cut(
                            df_temp['CURRENT_AGE'],
                            bins=[0, 18, 30, 40, 50, 60, 70, 100],
                            labels=['0-18', '19-30', '31-40', '41-50', '51-60', '61-70', '71+'],
                            include_lowest=True
                        )
                        age_dist = df_temp.groupby('AGE_GROUP').size().reset_index(name='count')

                        fig = px.bar(
                            age_dist,
                            x='AGE_GROUP',
                            y='count',
                            title="Age Distribution of Claimants",
                            labels={'AGE_GROUP': 'Age Group', 'count': 'Number of Claims'},
                            color='count',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.info("Could not create age distribution chart")
                else:
                    st.info("Age data not available for members")

                # Gender distribution
                if 'GENDER' in df.columns:
                    gender_dist = df['GENDER'].value_counts().reset_index()
                    gender_dist.columns = ['gender', 'count']

                    if not gender_dist.empty:
                        fig = px.pie(
                            gender_dist,
                            values='count',
                            names='gender',
                            title="Gender Distribution",
                            color_discrete_sequence=px.colors.sequential.Blues,
                            hole=0.4
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No gender data available")
                else:
                    st.info("Gender data not available")

            with col2:
                st.markdown('<div class="sub-header">Top Members by Claims</div>', unsafe_allow_html=True)

                if 'MEMBER_NO' in df.columns and 'AMOUNT_CLAIMED' in df.columns:
                    try:
                        # Calculate member-level statistics
                        member_stats = df.groupby('MEMBER_NO').agg({
                            'AMOUNT_CLAIMED': ['sum', 'count'],
                            'TOTAL_PAID': 'sum'
                        }).reset_index()

                        # Flatten column names
                        member_stats.columns = ['MEMBER_NO', 'TOTAL_CLAIMED', 'CLAIM_COUNT', 'TOTAL_PAID']

                        # Get top members by claim count
                        top_members = member_stats.nlargest(10, 'CLAIM_COUNT')

                        # Create bar chart
                        fig = px.bar(
                            top_members,
                            x='MEMBER_NO',
                            y='CLAIM_COUNT',
                            title="Top Members by Number of Claims",
                            labels={'MEMBER_NO': 'Member ID', 'CLAIM_COUNT': 'Number of Claims'},
                            color='TOTAL_CLAIMED',
                            color_continuous_scale='Viridis',
                            hover_data=['TOTAL_CLAIMED', 'TOTAL_PAID']
                        )
                        fig.update_layout(xaxis={'type': 'category'})
                        st.plotly_chart(fig, use_container_width=True)

                        # Member details table
                        st.markdown("**Top Members Details:**")
                        member_details = top_members.copy()
                        member_details['TOTAL_CLAIMED'] = member_details['TOTAL_CLAIMED'].apply(lambda x: f"${x:,.2f}")
                        member_details['TOTAL_PAID'] = member_details['TOTAL_PAID'].apply(lambda x: f"${x:,.2f}")
                        member_details['Paid %'] = ((member_details['TOTAL_PAID'].str.replace('$', '').str.replace(',',
                                                                                                                   '').astype(
                            float) /
                                                     member_details['TOTAL_CLAIMED'].str.replace('$', '').str.replace(
                                                         ',', '').astype(float)) * 100).round(1).apply(
                            lambda x: f"{x:.1f}%")

                        st.dataframe(
                            member_details[['MEMBER_NO', 'CLAIM_COUNT', 'TOTAL_CLAIMED', 'TOTAL_PAID', 'Paid %']],
                            use_container_width=True
                        )
                    except Exception as e:
                        st.info(f"Error analyzing member data: {str(e)}")
                else:
                    st.info("Member ID or claim amount data not available")

            # Member segmentation analysis
            st.markdown('<div class="sub-header">Member Segmentation Analysis</div>', unsafe_allow_html=True)

            if 'MEMBER_NO' in df.columns and 'AMOUNT_CLAIMED' in df.columns:
                try:
                    # Create member segments based on claim frequency
                    member_claims = df.groupby('MEMBER_NO').agg({
                        'AMOUNT_CLAIMED': 'sum',
                        'TOTAL_PAID': 'sum'
                    }).reset_index()
                    member_claims['claim_count'] = df.groupby('MEMBER_NO').size().values

                    # Segment members
                    member_claims['Segment'] = pd.cut(
                        member_claims['claim_count'],
                        bins=[0, 1, 3, 5, float('inf')],
                        labels=['One-time (1 claim)', 'Occasional (2-3 claims)', 'Regular (4-5 claims)',
                                'Frequent (6+ claims)']
                    )

                    # Calculate segment summary
                    segment_summary = member_claims.groupby('Segment').agg({
                        'MEMBER_NO': 'count',
                        'AMOUNT_CLAIMED': 'sum',
                        'TOTAL_PAID': 'sum',
                        'claim_count': 'mean'
                    }).reset_index()

                    segment_summary = segment_summary.rename(columns={
                        'MEMBER_NO': 'Member Count',
                        'AMOUNT_CLAIMED': 'Total Claimed',
                        'TOTAL_PAID': 'Total Paid',
                        'claim_count': 'Avg Claims per Member'
                    })

                    # Create visualization
                    col3, col4 = st.columns(2)

                    with col3:
                        fig = px.sunburst(
                            segment_summary,
                            path=['Segment'],
                            values='Total Claimed',
                            title="Member Segmentation by Claim Frequency",
                            color='Member Count',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col4:
                        # Format segment summary for display
                        display_summary = segment_summary.copy()
                        display_summary['Total Claimed'] = display_summary['Total Claimed'].apply(
                            lambda x: f"${x:,.2f}")
                        display_summary['Total Paid'] = display_summary['Total Paid'].apply(lambda x: f"${x:,.2f}")
                        display_summary['Avg Claims per Member'] = display_summary['Avg Claims per Member'].apply(
                            lambda x: f"{x:.1f}")

                        st.dataframe(
                            display_summary,
                            use_container_width=True
                        )
                except Exception as e:
                    st.info(f"Member segmentation analysis not available: {str(e)}")
            else:
                st.info("Member segmentation data not available")

    # Providers Tab
    if "🏥 Providers" in tab_names:
        providers_index = tab_names.index("🏥 Providers")
        with tabs[providers_index]:
            st.markdown('<div class="main-header">Provider Analysis</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="sub-header">Top 10 Providers by Claims Volume</div>', unsafe_allow_html=True)

                df = st.session_state.claims_data
                if 'PROVIDER_NAME' in df.columns and 'AMOUNT_CLAIMED' in df.columns:
                    try:
                        # Calculate provider statistics
                        provider_stats = df.groupby('PROVIDER_NAME').agg({
                            'AMOUNT_CLAIMED': 'sum',
                            'TOTAL_PAID': 'sum',
                            'MEMBER_NO': 'nunique'
                        }).reset_index()

                        provider_stats = provider_stats.sort_values('AMOUNT_CLAIMED', ascending=False).head(10)
                        provider_stats['Paid %'] = (
                                provider_stats['TOTAL_PAID'] / provider_stats['AMOUNT_CLAIMED'] * 100).round(1)

                        # Create grouped bar chart
                        fig = px.bar(
                            provider_stats,
                            x='PROVIDER_NAME',
                            y=['AMOUNT_CLAIMED', 'TOTAL_PAID'],
                            title="Top Providers - Claimed vs Paid Amounts",
                            barmode='group',
                            labels={'value': 'Amount ($)', 'variable': 'Type', 'PROVIDER_NAME': 'Provider'},
                            color_discrete_map={'AMOUNT_CLAIMED': '#1a237e', 'TOTAL_PAID': '#4caf50'}
                        )

                        # Truncate long provider names for better display
                        fig.update_layout(
                            xaxis={'tickangle': -45, 'tickmode': 'array'},
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.info(f"Error creating provider chart: {str(e)}")
                else:
                    st.info("Provider name or claim amount data not available")

            with col2:
                st.markdown('<div class="sub-header">Provider Efficiency Analysis</div>', unsafe_allow_html=True)

                if 'PROVIDER_NAME' in df.columns and 'AMOUNT_CLAIMED' in df.columns and 'TOTAL_PAID' in df.columns:
                    try:
                        provider_stats = df.groupby('PROVIDER_NAME').agg({
                            'AMOUNT_CLAIMED': 'sum',
                            'TOTAL_PAID': 'sum',
                            'MEMBER_NO': 'nunique'
                        }).reset_index()

                        # Calculate efficiency metrics
                        provider_stats['Efficiency'] = (
                                provider_stats['TOTAL_PAID'] / provider_stats['AMOUNT_CLAIMED'] * 100).round(1)
                        provider_stats['Avg Claim per Member'] = (
                                provider_stats['AMOUNT_CLAIMED'] / provider_stats['MEMBER_NO']).round(2)

                        # Filter for providers with sufficient data
                        filtered_providers = provider_stats[provider_stats['AMOUNT_CLAIMED'] > 0].nlargest(15,
                                                                                                           'AMOUNT_CLAIMED')

                        if not filtered_providers.empty:
                            fig = px.scatter(
                                filtered_providers,
                                x='AMOUNT_CLAIMED',
                                y='Efficiency',
                                size='MEMBER_NO',
                                color='Avg Claim per Member',
                                hover_name='PROVIDER_NAME',
                                title="Provider Efficiency Analysis",
                                labels={
                                    'AMOUNT_CLAIMED': 'Total Claimed ($)',
                                    'Efficiency': 'Paid %',
                                    'MEMBER_NO': 'Unique Members',
                                    'Avg Claim per Member': 'Avg Claim/Member ($)'
                                },
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Insufficient data for provider efficiency analysis")
                    except Exception as e:
                        st.info(f"Error analyzing provider efficiency: {str(e)}")
                else:
                    st.info("Provider efficiency data not available")

            # Provider details table
            st.markdown('<div class="sub-header">Provider Performance Details</div>', unsafe_allow_html=True)

            if 'PROVIDER_NAME' in df.columns and 'AMOUNT_CLAIMED' in df.columns:
                try:
                    provider_details = df.groupby('PROVIDER_NAME').agg({
                        'AMOUNT_CLAIMED': ['sum', 'mean', 'count'],
                        'TOTAL_PAID': 'sum',
                        'MEMBER_NO': 'nunique'
                    }).reset_index()

                    # Flatten column names
                    provider_details.columns = ['PROVIDER_NAME', 'Total_Claimed', 'Avg_Claim', 'Claim_Count',
                                                'Total_Paid', 'Unique_Members']

                    # Calculate additional metrics
                    provider_details['Paid_%'] = (
                            provider_details['Total_Paid'] / provider_details['Total_Claimed'] * 100).round(1)
                    provider_details['Claims_per_Member'] = (
                            provider_details['Claim_Count'] / provider_details['Unique_Members']).round(2)

                    # Sort by total claimed
                    provider_details = provider_details.sort_values('Total_Claimed', ascending=False).head(20)

                    # Format for display
                    display_details = provider_details.copy()
                    display_details['Total_Claimed'] = display_details['Total_Claimed'].apply(lambda x: f"${x:,.2f}")
                    display_details['Total_Paid'] = display_details['Total_Paid'].apply(lambda x: f"${x:,.2f}")
                    display_details['Avg_Claim'] = display_details['Avg_Claim'].apply(lambda x: f"${x:,.2f}")
                    display_details['Paid_%'] = display_details['Paid_%'].apply(lambda x: f"{x:.1f}%")

                    display_details = display_details.rename(columns={
                        'PROVIDER_NAME': 'Provider',
                        'Total_Claimed': 'Total Claimed',
                        'Total_Paid': 'Total Paid',
                        'Avg_Claim': 'Average Claim',
                        'Claim_Count': 'Claim Count',
                        'Unique_Members': 'Unique Members',
                        'Paid_%': 'Paid %',
                        'Claims_per_Member': 'Claims/Member'
                    })

                    st.dataframe(
                        display_details,
                        use_container_width=True,
                        height=500
                    )
                except Exception as e:
                    st.info(f"Provider details not available: {str(e)}")
            else:
                st.info("Provider performance data not available")

            # Provider type analysis (if available)
            if 'PROVIDER_TYPE' in df.columns or 'PROVIDER_CATEGORY' in df.columns:
                st.markdown('<div class="sub-header">Provider Type Analysis</div>', unsafe_allow_html=True)

                provider_type_col = 'PROVIDER_TYPE' if 'PROVIDER_TYPE' in df.columns else 'PROVIDER_CATEGORY'

                try:
                    type_analysis = df.groupby(provider_type_col).agg({
                        'AMOUNT_CLAIMED': 'sum',
                        'TOTAL_PAID': 'sum',
                        'MEMBER_NO': 'nunique',
                        'PROVIDER_NAME': 'nunique'
                    }).reset_index()

                    type_analysis = type_analysis.sort_values('AMOUNT_CLAIMED', ascending=False)

                    col5, col6 = st.columns(2)

                    with col5:
                        fig = px.bar(
                            type_analysis.head(10),
                            x=provider_type_col,
                            y='AMOUNT_CLAIMED',
                            title="Claims by Provider Type",
                            labels={provider_type_col: 'Provider Type', 'AMOUNT_CLAIMED': 'Total Claimed ($)'},
                            color='AMOUNT_CLAIMED',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col6:
                        # Format for display
                        display_type = type_analysis.copy()
                        display_type['AMOUNT_CLAIMED'] = display_type['AMOUNT_CLAIMED'].apply(lambda x: f"${x:,.2f}")
                        display_type['TOTAL_PAID'] = display_type['TOTAL_PAID'].apply(lambda x: f"${x:,.2f}")
                        display_type = display_type.rename(columns={
                            provider_type_col: 'Provider Type',
                            'AMOUNT_CLAIMED': 'Total Claimed',
                            'TOTAL_PAID': 'Total Paid',
                            'MEMBER_NO': 'Unique Members',
                            'PROVIDER_NAME': 'Unique Providers'
                        })

                        st.dataframe(
                            display_type,
                            use_container_width=True
                        )
                except:
                    st.info("Provider type analysis not available")

    # IBNR Analysis Tab - UPDATED with Excel Chain Ladder Model
    if "📊 IBNR Analysis" in tab_names:
        ibnr_tab_index = tab_names.index("📊 IBNR Analysis")
        with tabs[ibnr_tab_index]:
            st.markdown('<div class="main-header">IBNR Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">Incurred But Not Reported Claims Estimation</div>',
                        unsafe_allow_html=True)

            # Excel Model Upload Section
            st.markdown("### 📋 Excel Chain Ladder Model")
            col_upload, col_info = st.columns([2, 1])

            with col_upload:
                excel_model_file = st.file_uploader(
                    "Upload Excel Chain Ladder Model (Optional)",
                    type=['xlsx'],
                    help="Upload your IBNR Model.xlsx file for comparison"
                )

            with col_info:
                st.info("""
                **Excel Model Info:**
                - Based on Chain Ladder method
                - Uses development factors
                - Projects cumulative claims to ultimate
                """)

            # Main IBNR Calculation
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🎯 IBNR Calculation Methods")

                # Method selection - include Excel model
                selected_methods = st.multiselect(
                    "Select IBNR calculation methods:",
                    options=['Excel Chain Ladder', 'Chain Ladder', 'Bornhuetter-Ferguson', 'Frequency-Severity',
                             'Bootstrap'],
                    default=['Excel Chain Ladder', 'Chain Ladder', 'Bornhuetter-Ferguson']
                )

                # Parameters
                with st.expander("⚙️ Method Parameters", expanded=False):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        loss_ratio = st.slider("Expected Loss Ratio", 0.1, 1.0, 0.75, 0.05)
                        months_unreported = st.slider("Months of Unreported Claims", 1, 12, 3)

                    with col_b:
                        n_simulations = st.number_input("Bootstrap Simulations", 100, 10000, 1000, step=100)
                        tail_factor = st.number_input("Tail Factor", 1.0, 2.0, 1.05, 0.01)

                # Calculate button
                if st.button("Calculate IBNR", type="primary", use_container_width=True):
                    with st.spinner("Calculating IBNR using selected methods..."):
                        # Convert selected methods to format for composite function
                        method_map = {
                            'Excel Chain Ladder': 'excel',
                            'Chain Ladder': 'chain_ladder',
                            'Bornhuetter-Ferguson': 'bornhuetter',
                            'Frequency-Severity': 'frequency',
                            'Bootstrap': 'bootstrap'
                        }

                        selected_methods_formatted = [method_map[m] for m in selected_methods if m in method_map]

                        # Calculate IBNR
                        composite_ibnr, ibnr_details = calculate_composite_ibnr(
                            st.session_state.claims_data,
                            methods=selected_methods_formatted
                        )

                        st.session_state.ibnr_estimate = composite_ibnr
                        st.session_state.ibnr_method = ibnr_details.get('method', 'Composite')

                        if 'component_results' in ibnr_details:
                            st.session_state.ibnr_results = ibnr_details['component_results']
                        else:
                            st.session_state.ibnr_results = None

                        st.success(f"IBNR calculated successfully!")
                        st.rerun()

            with col2:
                st.markdown("### 📊 IBNR Estimate")

                if st.session_state.ibnr_estimate is not None:
                    # Display IBNR metrics
                    st.markdown('<div class="ibnr-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label" style="color: white;">Estimated IBNR</div>',
                                unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-value" style="color: white; font-size: 2.5rem;">${st.session_state.ibnr_estimate:,.0f}</div>',
                        unsafe_allow_html=True)
                    st.caption(f"Method: {st.session_state.ibnr_method}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Calculate IBNR ratio
                    reported_claims = st.session_state.claims_metrics.get('total_claimed', 0)
                    if reported_claims > 0:
                        ibnr_ratio = (st.session_state.ibnr_estimate / reported_claims) * 100

                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">IBNR Ratio</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-value">{ibnr_ratio:.1f}%</div>', unsafe_allow_html=True)
                        st.caption("IBNR as % of Reported Claims")
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">Estimated Ultimate</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="metric-value">${reported_claims + st.session_state.ibnr_estimate:,.0f}</div>',
                            unsafe_allow_html=True)
                        st.caption("Reported + IBNR")
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Click 'Calculate IBNR' to generate estimates")

            # Display Excel Model Results if available
            ibnr_results = st.session_state.ibnr_results
            if ibnr_results and 'Excel Chain Ladder' in ibnr_results:
                st.markdown("### 📐 Excel Chain Ladder Model Details")

                excel_details = ibnr_results['Excel Chain Ladder']['details']

                # Display development factors
                if 'development_factors' in excel_details:
                    st.markdown("**Development Factors:**")
                    factors_df = pd.DataFrame({
                        'Transition': list(excel_details['development_factors'].keys()),
                        'Factor': list(excel_details['development_factors'].values())
                    })
                    st.dataframe(factors_df, use_container_width=True)

                # Display triangles
                col3, col4 = st.columns(2)

                with col3:
                    if 'incremental_triangle' in excel_details:
                        inc_triangle = excel_details['incremental_triangle']
                        st.markdown("**Incremental Triangle:**")
                        st.dataframe(inc_triangle.style.format("${:,.0f}"), use_container_width=True, height=300)

                with col4:
                    if 'cumulative_triangle' in excel_details:
                        cum_triangle = excel_details['cumulative_triangle']
                        st.markdown("**Cumulative Triangle:**")
                        st.dataframe(cum_triangle.style.format("${:,.0f}"), use_container_width=True, height=300)

                # Display summary metrics
                st.markdown("**Model Summary:**")
                summary_data = {
                    'Metric': ['Total Reported Claims', 'Estimated Ultimate Claims', 'IBNR Reserve', 'IBNR Ratio'],
                    'Value': [
                        f"${excel_details.get('total_reported', 0):,.2f}",
                        f"${excel_details.get('total_ultimate', 0):,.2f}",
                        f"${excel_details.get('ibnr_estimate', 0):,.2f}",
                        f"{excel_details.get('ibnr_ratio', 0):.1f}%"
                    ]
                }
                st.table(pd.DataFrame(summary_data))

            # Method Comparison Chart
            if ibnr_results:
                st.markdown("### 🔍 Method Comparison")

                fig = create_ibnr_comparison_chart(ibnr_results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Display method details
                st.markdown("**Method Details:**")
                for method, data in ibnr_results.items():
                    with st.expander(f"{method}: ${data['ibnr']:,.0f}", expanded=False):
                        if data['details']:
                            details_df = pd.DataFrame([
                                {'Parameter': k, 'Value': v}
                                for k, v in data['details'].items()
                                if not isinstance(v, (pd.DataFrame, dict, list)) or k in ['method', 'tail_factor']
                            ])
                            st.table(details_df)

            # Development Triangle Visualization
            if 'Excel Chain Ladder' in ibnr_results:
                excel_details = ibnr_results['Excel Chain Ladder']['details']
                if 'cumulative_triangle' in excel_details:
                    fig = create_excel_chain_ladder_chart(excel_details)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            # Bootstrap Confidence Intervals (if available)
            if (ibnr_results and
                    'Bootstrap' in ibnr_results and
                    'details' in ibnr_results['Bootstrap'] and
                    'simulations' in ibnr_results['Bootstrap']['details']):

                st.markdown("### 📊 Bootstrap Analysis")
                fig = create_ibnr_confidence_chart(ibnr_results['Bootstrap']['details'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Display confidence intervals
                bootstrap_details = ibnr_results['Bootstrap']['details']
                if 'confidence_95' in bootstrap_details and 'confidence_90' in bootstrap_details:
                    col7, col8, col9 = st.columns(3)

                    with col7:
                        st.metric(
                            "95% Confidence Interval",
                            f"${bootstrap_details['confidence_95'][0]:,.0f} - ${bootstrap_details['confidence_95'][1]:,.0f}",
                            f"Range: ${bootstrap_details['confidence_95'][1] - bootstrap_details['confidence_95'][0]:,.0f}"
                        )

                    with col8:
                        st.metric(
                            "90% Confidence Interval",
                            f"${bootstrap_details['confidence_90'][0]:,.0f} - ${bootstrap_details['confidence_90'][1]:,.0f}",
                            f"Range: ${bootstrap_details['confidence_90'][1] - bootstrap_details['confidence_90'][0]:,.0f}"
                        )

                    with col9:
                        st.metric(
                            "Standard Deviation",
                            f"${bootstrap_details.get('std_dev', 0):,.0f}",
                            f"Coefficient of Variation: {(bootstrap_details.get('std_dev', 0) / bootstrap_details.get('mean', 1)) * 100:.1f}%" if bootstrap_details.get(
                                'mean', 0) > 0 else "N/A"
                        )

            # Actuarial Notes and Assumptions
            with st.expander("📝 Actuarial Notes & Assumptions", expanded=False):
                st.markdown("""
                **IBNR Calculation Methods:**

                1. **Excel Chain Ladder Method:**
                   - Replicates the Excel model methodology
                   - Uses development factors similar to Excel formulas
                   - Projects cumulative claims to ultimate using chain ladder

                2. **Chain Ladder Method:**
                   - Uses historical development patterns to project future claims
                   - Assumes past development patterns will continue
                   - Most widely used method for short-tailed lines

                3. **Bornhuetter-Ferguson Method:**
                   - Combines actual reported claims with expected loss ratio
                   - Less sensitive to recent volatility
                   - Good for long-tailed or volatile lines

                4. **Frequency-Severity Method:**
                   - Separates claim frequency and severity analysis
                   - Uses statistical distributions (Poisson for frequency)
                   - Good when claim counts are stable

                5. **Bootstrap Method:**
                   - Uses resampling to estimate uncertainty
                   - Provides confidence intervals
                   - Computationally intensive but robust

                **Key Assumptions:**
                - Claim reporting patterns are consistent
                - No systemic changes in claims handling
                - Adequate historical data for development patterns
                - Premium information available for loss ratio methods

                **Limitations:**
                - Results are estimates with inherent uncertainty
                - Quality depends on data completeness
                - May not capture sudden changes or "shocks"
                - Professional judgment required for final selection
                """)

    # Premium Analysis Tab
    if "💎 Premium Analysis" in tab_names:
        premium_tab_index = tab_names.index("💎 Premium Analysis")
        with tabs[premium_tab_index]:
            st.markdown('<div class="main-header">Premium Analysis</div>', unsafe_allow_html=True)
            metrics = st.session_state.premium_metrics

            # Premium metrics row 1
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label" style="color: white;">Total Premium</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="metric-value" style="color: white;">${metrics.get("total_premium", 0):,.2f}</div>',
                    unsafe_allow_html=True)
                st.caption("Total premium collected")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Active Members</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{metrics.get("active_members", 0):,}</div>',
                            unsafe_allow_html=True)
                st.caption(f"of {metrics.get('total_members', 0):,} total")
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Avg Premium</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${metrics.get("avg_premium", 0):,.2f}</div>',
                            unsafe_allow_html=True)
                st.caption("Average premium per member")
                st.markdown('</div>', unsafe_allow_html=True)

            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                termination_rate = metrics.get('termination_rate', 0) * 100
                st.markdown('<div class="metric-label">Termination Rate</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{termination_rate:.1f}%</div>', unsafe_allow_html=True)
                st.caption(f"{metrics.get('terminated_members', 0):,} terminated")
                st.markdown('</div>', unsafe_allow_html=True)

            # Premium charts row 1
            col1, col2 = st.columns(2)

            with col1:
                if 'product_analysis' in metrics and not metrics['product_analysis'].empty:
                    fig = create_premium_product_chart(metrics['product_analysis'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'payer_analysis' in metrics and not metrics['payer_analysis'].empty:
                    fig = create_premium_payer_chart(metrics['payer_analysis'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            # Premium charts row 2
            col3, col4 = st.columns(2)

            with col3:
                if 'monthly_trends' in metrics and not metrics['monthly_trends'].empty:
                    fig = create_premium_trend_chart(metrics['monthly_trends'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            with col4:
                if 'age_distribution' in metrics and not metrics['age_distribution'].empty:
                    fig = create_premium_age_chart(metrics['age_distribution'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

    # Premium Details Tab
    if "📊 Premium Details" in tab_names:
        premium_details_index = tab_names.index("📊 Premium Details")
        with tabs[premium_details_index]:
            st.markdown('<div class="main-header">Premium Details</div>', unsafe_allow_html=True)

            # Product analysis table
            if 'product_analysis' in st.session_state.premium_metrics:
                product_data = st.session_state.premium_metrics['product_analysis'].copy()
                product_data['avg_premium'] = product_data['PREMIUM_AMOUNT'] / product_data['MEMBER_NO']
                product_data = product_data.rename(columns={
                    'PRODUCT_NAME': 'Product',
                    'PREMIUM_AMOUNT': 'Total Premium',
                    'MEMBER_NO': 'Members',
                    'avg_premium': 'Avg Premium'
                })

                st.markdown('<div class="sub-header">Product Performance</div>', unsafe_allow_html=True)
                st.dataframe(product_data, use_container_width=True)

            # Payer analysis table
            if 'payer_analysis' in st.session_state.premium_metrics:
                payer_data = st.session_state.premium_metrics['payer_analysis'].copy()
                payer_data['avg_premium'] = payer_data['PREMIUM_AMOUNT'] / payer_data['MEMBER_NO']
                payer_data = payer_data.rename(columns={
                    'PAYER_NAME': 'Payer',
                    'PREMIUM_AMOUNT': 'Total Premium',
                    'MEMBER_NO': 'Members',
                    'avg_premium': 'Avg Premium'
                })

                st.markdown('<div class="sub-header">Payer Analysis</div>', unsafe_allow_html=True)
                st.dataframe(payer_data, use_container_width=True)

            # Raw premium data
            st.markdown('<div class="sub-header">Premium Data</div>', unsafe_allow_html=True)
            st.dataframe(st.session_state.premium_data, use_container_width=True, height=400)

    # Combined View Tab
    if "🔗 Combined View" in tab_names:
        combined_index = tab_names.index("🔗 Combined View")
        with tabs[combined_index]:
            st.markdown('<div class="main-header">Combined Analysis</div>', unsafe_allow_html=True)

            # Calculate key combined metrics
            total_claims = st.session_state.claims_metrics.get('total_claimed', 0)
            total_paid = st.session_state.claims_metrics.get('total_paid', 0)
            total_premium = st.session_state.premium_metrics.get('total_premium', 0)
            ibnr_value = st.session_state.ibnr_estimate or 0

            # Loss ratio analysis
            col1, col2, col3 = st.columns(3)

            with col1:
                if total_premium > 0:
                    loss_ratio = (total_claims / total_premium) * 100
                    paid_loss_ratio = (total_paid / total_premium) * 100

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=loss_ratio,
                        title={'text': "Loss Ratio (%)"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 200]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 80], 'color': "lightgreen"},
                                {'range': [80, 120], 'color': "yellow"},
                                {'range': [120, 200], 'color': "red"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Member overlap
                if 'MEMBER_NO' in st.session_state.claims_data.columns and 'MEMBER_NO' in st.session_state.premium_data.columns:
                    claims_members = set(st.session_state.claims_data['MEMBER_NO'].unique())
                    premium_members = set(st.session_state.premium_data['MEMBER_NO'].unique())
                    common_members = claims_members.intersection(premium_members)

                    labels = ['Claims Only', 'Both', 'Premium Only']
                    values = [
                        len(claims_members - premium_members),
                        len(common_members),
                        len(premium_members - claims_members)
                    ]

                    fig = px.pie(
                        values=values,
                        names=labels,
                        title="Member Overlap",
                        color_discrete_sequence=['#ff9999', '#66b3ff', '#99ff99']
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

            with col3:
                # Financial summary with IBNR
                st.markdown("### Financial Summary")
                summary_data = {
                    'Metric': ['Total Premium', 'Total Claims', 'Total Paid', 'Estimated IBNR', 'Estimated Ultimate',
                               'Net Position'],
                    'Amount': [
                        total_premium,
                        total_claims,
                        total_paid,
                        ibnr_value,
                        total_claims + ibnr_value,
                        total_premium - (total_paid + ibnr_value)
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df['Amount'] = summary_df['Amount'].apply(lambda x: f"${x:,.2f}")
                st.table(summary_df)

            # Combined data table
            st.markdown('<div class="sub-header">Member-Level Combined View</div>', unsafe_allow_html=True)

            # Create combined summary
            if 'MEMBER_NO' in st.session_state.premium_data.columns:
                # Get premium by member
                premium_by_member = st.session_state.premium_data.groupby('MEMBER_NO').agg({
                    'PREMIUM_AMOUNT': 'sum',
                    'PRODUCT_NAME': 'first',
                    'PAYER_NAME': 'first'
                }).reset_index()

                # Get claims by member
                if 'MEMBER_NO' in st.session_state.claims_data.columns:
                    claims_by_member = st.session_state.claims_data.groupby('MEMBER_NO').agg({
                        'AMOUNT_CLAIMED': 'sum',
                        'TOTAL_PAID': 'sum'
                    }).reset_index()

                    # Merge data
                    combined = pd.merge(premium_by_member, claims_by_member,
                                        on='MEMBER_NO', how='left').fillna(0)

                    # Calculate metrics
                    combined['Loss_Ratio'] = (combined['TOTAL_PAID'] / combined['PREMIUM_AMOUNT']) * 100
                    combined['Loss_Ratio'] = combined['Loss_Ratio'].replace([np.inf, -np.inf], 0)

                    st.dataframe(combined, use_container_width=True, height=400)

    # Data Export Tab
    export_index = tab_names.index("📁 Data Export")
    with tabs[export_index]:
        st.markdown('<div class="main-header">Data Export</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="sub-header">Export Claims Data</div>', unsafe_allow_html=True)
            if st.session_state.claims_data is not None:
                csv = st.session_state.claims_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Claims as CSV",
                    data=csv,
                    file_name=f"claims_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col2:
            st.markdown('<div class="sub-header">Export Premium Data</div>', unsafe_allow_html=True)
            if st.session_state.premium_data is not None:
                csv = st.session_state.premium_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Premium as CSV",
                    data=csv,
                    file_name=f"premium_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        # Combined report
        if st.session_state.claims_data is not None and st.session_state.premium_data is not None:
            st.markdown('<div class="sub-header">Combined Report</div>', unsafe_allow_html=True)

            # Create summary report
            ibnr_value = st.session_state.ibnr_estimate or 0
            report_data = {
                'Metric': [
                    'Total Premium Collected',
                    'Total Claims Submitted',
                    'Total Claims Paid',
                    'Estimated IBNR',
                    'Estimated Ultimate Claims',
                    'Loss Ratio (Reported)',
                    'Loss Ratio (Ultimate)',
                    'Active Members',
                    'Average Premium',
                    'Average Claim'
                ],
                'Value': [
                    f"${total_premium:,.2f}",
                    f"${total_claims:,.2f}",
                    f"${total_paid:,.2f}",
                    f"${ibnr_value:,.2f}",
                    f"${total_claims + ibnr_value:,.2f}",
                    f"{(total_paid / total_premium * 100):.1f}%" if total_premium > 0 else "N/A",
                    f"{((total_paid + ibnr_value) / total_premium * 100):.1f}%" if total_premium > 0 else "N/A",
                    f"{st.session_state.premium_metrics.get('active_members', 0):,}",
                    f"${st.session_state.premium_metrics.get('avg_premium', 0):,.2f}",
                    f"${st.session_state.claims_metrics.get('avg_claim_amount', 0):,.2f}"
                ]
            }

            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("📊 **Actuarial Dashboard v2.0**")
with col2:
    st.caption("📍 Operations Department")
with col3:
    st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  # Dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
from scipy import stats
from scipy.optimize import curve_fit
import math

# Page configuration
st.set_page_config(
    page_title="Actuarial Dashboard - Operations",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1a237e;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #283593;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1a237e;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a237e;
        color: white;
    }
    .upload-section {
        border: 3px dashed #ddd;
        border-radius: 10px;
        padding: 3rem;
        text-align: center;
        background: #fafafa;
        margin-bottom: 2rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
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

# Initialize session state
if 'claims_data' not in st.session_state:
    st.session_state.claims_data = None
if 'premium_data' not in st.session_state:
    st.session_state.premium_data = None
if 'claims_metrics' not in st.session_state:
    st.session_state.claims_metrics = None
if 'premium_metrics' not in st.session_state:
    st.session_state.premium_metrics = None
if 'claims_file_name' not in st.session_state:
    st.session_state.claims_file_name = None
if 'premium_file_name' not in st.session_state:
    st.session_state.premium_file_name = None
if 'ibnr_estimate' not in st.session_state:
    st.session_state.ibnr_estimate = None
if 'ibnr_method' not in st.session_state:
    st.session_state.ibnr_method = None
if 'development_triangle' not in st.session_state:
    st.session_state.development_triangle = None
if 'ibnr_results' not in st.session_state:
    st.session_state.ibnr_results = None


# Helper functions for claims data (existing)
def clean_column_names(df):
    """Clean and standardize column names"""
    cleaned_columns = []
    for col in df.columns:
        col_str = str(col)
        cleaned = col_str.strip().upper().replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')
        cleaned_columns.append(cleaned)
    df.columns = cleaned_columns
    return df


def parse_claims_file(file):
    """Parse uploaded claims Excel file"""
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


# Helper functions for premium data
def parse_premium_file(file):
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


def calculate_premium_metrics(df):
    """Calculate premium metrics"""
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


def calculate_claims_metrics(df):
    """Calculate claims metrics"""
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


# EXCEL CHAIN LADDER MODEL FUNCTIONS
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


# IBNR Calculation Functions - UPDATED to fix the date error
def calculate_development_triangle(df, accident_period='M', development_period='M'):
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


def calculate_ibnr_chain_ladder(cum_triangle):
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


def calculate_ibnr_bornhuetter_ferguson(df, expected_loss_ratio=0.75):
    """Calculate IBNR using Bornhuetter-Ferguson method"""
    if df is None or df.empty or 'AMOUNT_CLAIMED' not in df.columns:
        return 0, {}

    try:
        # Calculate earned premium (simplified - use average premium per member * number of claims)
        if st.session_state.premium_metrics and 'avg_premium' in st.session_state.premium_metrics:
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


def calculate_ibnr_frequency_severity(df):
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
            lambda_freq = monthly_counts.mean()

            # Calculate severity distribution
            severities = df['AMOUNT_CLAIMED'].values
            avg_severity = np.mean(severities)
            std_severity = np.std(severities)

            # Estimate IBNR
            months_unreported = 3  # Assume 3 months of unreported claims
            expected_unreported_claims = lambda_freq * months_unreported
            ibnr = expected_unreported_claims * avg_severity

            return ibnr, {
                'method': 'Frequency-Severity',
                'lambda_frequency': lambda_freq,
                'avg_severity': avg_severity,
                'std_severity': std_severity,
                'expected_unreported_claims': expected_unreported_claims,
                'months_unreported': months_unreported
            }
        else:
            return 0, {}

    except Exception as e:
        st.error(f"Error in Frequency-Severity calculation: {str(e)}")
        return 0, {}


def calculate_ibnr_bootstrap(df, n_simulations=1000):
    """Calculate IBNR using Bootstrap method with confidence intervals"""
    if df is None or df.empty:
        return 0, {}

    try:
        # Get development triangle
        inc_triangle, cum_triangle, _ = calculate_development_triangle(df)

        if cum_triangle is None:
            return 0, {}

        # Simple bootstrap simulation
        ibnr_simulations = []
        n_accident = len(cum_triangle.index)
        n_development = len(cum_triangle.columns)

        for _ in range(n_simulations):
            # Resample development factors
            sampled_factors = []
            for i in range(n_development - 1):
                current_col = cum_triangle.iloc[:, i]
                next_col = cum_triangle.iloc[:, i + 1]

                # Bootstrap ratio
                ratios = []
                for j in range(len(current_col)):
                    if current_col.iloc[j] > 0:
                        ratio = next_col.iloc[j] / current_col.iloc[j]
                        ratios.append(ratio)

                if ratios:
                    sampled_factor = np.random.choice(ratios)
                    sampled_factors.append(sampled_factor)
                else:
                    sampled_factors.append(1.0)

            # Calculate ultimate for each accident period
            ultimate_total = 0
            for i in range(n_accident):
                current_value = cum_triangle.iloc[i, -1]
                remaining_factors = 1.0

                # Apply remaining development factors
                for j in range(n_accident - i - 1):
                    if j < len(sampled_factors):
                        remaining_factors *= sampled_factors[j]

                ultimate_total += current_value * remaining_factors

            reported_total = cum_triangle.iloc[:, -1].sum()
            ibnr_sim = ultimate_total - reported_total
            ibnr_simulations.append(ibnr_sim)

        # Calculate statistics
        ibnr_mean = np.mean(ibnr_simulations)
        ibnr_std = np.std(ibnr_simulations)
        ibnr_median = np.median(ibnr_simulations)

        # Calculate confidence intervals
        confidence_95 = np.percentile(ibnr_simulations, [2.5, 97.5])
        confidence_90 = np.percentile(ibnr_simulations, [5, 95])

        return ibnr_mean, {
            'method': 'Bootstrap Simulation',
            'n_simulations': n_simulations,
            'mean': ibnr_mean,
            'median': ibnr_median,
            'std_dev': ibnr_std,
            'confidence_95': confidence_95,
            'confidence_90': confidence_90,
            'simulations': ibnr_simulations
        }

    except Exception as e:
        st.error(f"Error in Bootstrap calculation: {str(e)}")
        return 0, {}


def calculate_composite_ibnr(df, methods=['chain_ladder', 'bornhuetter', 'frequency', 'excel']):
    """Calculate composite IBNR using multiple methods including Excel model"""
    results = {}

    # Chain Ladder
    if 'chain_ladder' in methods:
        inc_triangle, cum_triangle, _ = calculate_development_triangle(df)
        if cum_triangle is not None and not cum_triangle.empty:
            ibnr_cl, details_cl = calculate_ibnr_chain_ladder(cum_triangle)
            results['Chain Ladder'] = {
                'ibnr': ibnr_cl,
                'details': details_cl
            }

    # Bornhuetter-Ferguson
    if 'bornhuetter' in methods:
        ibnr_bf, details_bf = calculate_ibnr_bornhuetter_ferguson(df)
        results['Bornhuetter-Ferguson'] = {
            'ibnr': ibnr_bf,
            'details': details_bf
        }

    # Frequency-Severity
    if 'frequency' in methods:
        ibnr_fs, details_fs = calculate_ibnr_frequency_severity(df)
        results['Frequency-Severity'] = {
            'ibnr': ibnr_fs,
            'details': details_fs
        }

    # Bootstrap
    if 'bootstrap' in methods:
        ibnr_bs, details_bs = calculate_ibnr_bootstrap(df)
        results['Bootstrap'] = {
            'ibnr': ibnr_bs,
            'details': details_bs
        }

    # Excel Chain Ladder Model
    if 'excel' in methods:
        ibnr_excel, details_excel = calculate_ibnr_excel_chain_ladder(df)
        results['Excel Chain Ladder'] = {
            'ibnr': ibnr_excel,
            'details': details_excel
        }

    if not results:
        return 0, {}

    # Calculate weighted average
    weights = {
        'Excel Chain Ladder': 0.40,  # Give more weight to Excel model
        'Chain Ladder': 0.25,
        'Bornhuetter-Ferguson': 0.20,
        'Frequency-Severity': 0.10,
        'Bootstrap': 0.05
    }

    composite_ibnr = 0
    valid_methods = 0

    for method, data in results.items():
        if data['ibnr'] > 0:
            composite_ibnr += data['ibnr'] * weights.get(method, 0.20)
            valid_methods += 1

    if valid_methods > 0:
        # Adjust for number of valid methods
        composite_ibnr = composite_ibnr * (valid_methods / len(methods))
    else:
        composite_ibnr = 0

    return composite_ibnr, {
        'method': 'Composite (Weighted Average)',
        'component_results': results,
        'weights': weights
    }


# Chart functions for premium data
def create_premium_product_chart(product_data):
    """Create premium by product chart"""
    if product_data.empty or 'PRODUCT_NAME' not in product_data.columns:
        return None

    fig = px.bar(
        product_data.head(10),
        x='PRODUCT_NAME',
        y='PREMIUM_AMOUNT',
        title="Premium by Product",
        labels={'PRODUCT_NAME': 'Product', 'PREMIUM_AMOUNT': 'Premium Amount ($)'},
        color='PREMIUM_AMOUNT',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(height=400, template="plotly_white")
    return fig


def create_premium_payer_chart(payer_data):
    """Create premium by payer chart"""
    if payer_data.empty or 'PAYER_NAME' not in payer_data.columns:
        return None

    fig = px.pie(
        payer_data.head(10),
        values='PREMIUM_AMOUNT',
        names='PAYER_NAME',
        title="Premium Distribution by Payer",
        hole=0.4
    )

    fig.update_layout(height=400, template="plotly_white")
    return fig


def create_premium_trend_chart(monthly_data):
    """Create premium trend chart"""
    if monthly_data.empty or 'join_month' not in monthly_data.columns:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_data['join_month'],
        y=monthly_data['PREMIUM_AMOUNT'],
        name='Monthly Premium',
        line=dict(color='#764ba2', width=3),
        fill='tozeroy',
        fillcolor='rgba(118, 75, 162, 0.1)'
    ))

    fig.update_layout(
        title="Premium Enrollment Trends",
        xaxis_title="Month",
        yaxis_title="Premium Amount ($)",
        template="plotly_white",
        hovermode="x unified",
        height=400
    )

    return fig


def create_premium_age_chart(age_data):
    """Create premium age distribution chart"""
    if age_data.empty or 'age_group' not in age_data.columns:
        return None

    fig = px.bar(
        age_data,
        x='age_group',
        y='PREMIUM_AMOUNT',
        title="Premium by Age Group",
        labels={'age_group': 'Age Group', 'PREMIUM_AMOUNT': 'Total Premium ($)'},
        color='member_count',
        color_continuous_scale='Blues'
    )

    fig.update_layout(height=400, template="plotly_white")
    return fig


# Chart functions for claims data
def create_monthly_trend_chart(monthly_data):
    """Create monthly trend chart for claims"""
    if monthly_data.empty or 'month_year' not in monthly_data.columns:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_data['month_year'],
        y=monthly_data['AMOUNT_CLAIMED'],
        name='Amount Claimed',
        line=dict(color='#1a237e', width=3),
        fill='tozeroy',
        fillcolor='rgba(26, 35, 126, 0.1)'
    ))

    if 'TOTAL_PAID' in monthly_data.columns:
        fig.add_trace(go.Scatter(
            x=monthly_data['month_year'],
            y=monthly_data['TOTAL_PAID'],
            name='Amount Paid',
            line=dict(color='#4caf50', width=3),
            fill='tozeroy',
            fillcolor='rgba(76, 175, 80, 0.1)'
        ))

    fig.update_layout(
        title="Monthly Claims Trend",
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        template="plotly_white",
        hovermode="x unified",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_provider_chart(provider_data):
    """Create provider analysis chart for claims"""
    if provider_data.empty or 'PROVIDER_NAME' not in provider_data.columns:
        return None

    # Truncate long provider names for better display
    provider_data = provider_data.copy()
    provider_data['PROVIDER_NAME_SHORT'] = provider_data['PROVIDER_NAME'].apply(
        lambda x: x[:30] + '...' if len(x) > 30 else x
    )

    fig = px.bar(
        provider_data.head(10),
        y='PROVIDER_NAME_SHORT',
        x='AMOUNT_CLAIMED',
        orientation='h',
        title="Top Providers by Claims Amount",
        labels={'PROVIDER_NAME_SHORT': 'Provider', 'AMOUNT_CLAIMED': 'Amount Claimed ($)'},
        color='AMOUNT_CLAIMED',
        color_continuous_scale='Blues',
        hover_data=['PROVIDER_NAME']
    )

    fig.update_layout(
        height=500,
        template="plotly_white",
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


def create_age_distribution_chart(age_data):
    """Create age distribution chart for claims"""
    if age_data.empty or 'age_group' not in age_data.columns:
        return None

    fig = px.pie(
        age_data,
        values='count',
        names='age_group',
        title="Age Distribution of Members",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Blues_r
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400, template="plotly_white")

    return fig


def create_service_type_chart(service_data):
    """Create service type chart for claims"""
    if service_data.empty or 'BASE_BENEFIT_DESCRIPTION' not in service_data.columns:
        return None

    # Truncate long descriptions
    service_data = service_data.copy()
    service_data['SERVICE_SHORT'] = service_data['BASE_BENEFIT_DESCRIPTION'].apply(
        lambda x: x[:40] + '...' if len(x) > 40 else x
    )

    fig = px.treemap(
        service_data.head(10),
        path=['SERVICE_SHORT'],
        values='AMOUNT_CLAIMED',
        title="Service Type Breakdown",
        color='AMOUNT_CLAIMED',
        color_continuous_scale='Blues',
        hover_data=['BASE_BENEFIT_DESCRIPTION']
    )

    fig.update_layout(height=500, template="plotly_white")

    return fig


def create_financial_kpi_chart(metrics):
    """Create financial KPI chart for claims"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Paid Ratio', 'Average Claim', 'Rejection Rate'),
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
    )

    # Paid Ratio
    paid_ratio = metrics.get('paid_ratio', 0) * 100
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=paid_ratio,
            title={'text': "Paid Ratio (%)"},
            domain={'row': 0, 'column': 0},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1a237e"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "darkgray"}
                ]
            }
        ),
        row=1, col=1
    )

    # Average Claim
    avg_claim = metrics.get('avg_claim_amount', 0)
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=avg_claim,
            title={'text': "Average Claim ($)"},
            domain={'row': 0, 'column': 1},
            number={'prefix': "$", 'valueformat': '.2f'}
        ),
        row=1, col=2
    )

    # Rejection Rate
    rejection_rate = metrics.get('rejection_rate', 0) * 100
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=rejection_rate,
            title={'text': "Rejection Rate (%)"},
            domain={'row': 0, 'column': 2},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#f44336"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 40], 'color': "yellow"},
                    {'range': [40, 100], 'color': "lightcoral"}
                ]
            }
        ),
        row=1, col=3
    )

    fig.update_layout(height=300, template="plotly_white")

    return fig


# IBNR Chart Functions
def create_ibnr_comparison_chart(ibnr_results):
    """Create comparison chart of different IBNR methods"""
    if not ibnr_results or ibnr_results is None:
        return None

    methods = []
    values = []

    for method, data in ibnr_results.items():
        if data and 'ibnr' in data:
            methods.append(method)
            values.append(data['ibnr'])

    if not methods:
        return None

    fig = px.bar(
        x=methods,
        y=values,
        title="IBNR Estimates by Method",
        labels={'x': 'Method', 'y': 'IBNR Amount ($)'},
        color=values,
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        height=400,
        template="plotly_white",
        xaxis_tickangle=-45
    )

    return fig


def create_development_triangle_chart(cum_triangle):
    """Create heatmap of development triangle"""
    if cum_triangle is None or cum_triangle.empty:
        return None

    # Convert to dataframe for plotting
    plot_data = cum_triangle.copy()
    plot_data.index = plot_data.index.astype(str)

    fig = px.imshow(
        plot_data,
        title="Claims Development Triangle",
        labels=dict(x="Development Period", y="Accident Period", color="Cumulative Claims"),
        aspect="auto",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        height=500,
        template="plotly_white",
        xaxis_title="Development Lag (Months)",
        yaxis_title="Accident Period"
    )

    return fig


def create_ibnr_confidence_chart(bootstrap_details):
    """Create confidence interval chart for bootstrap method"""
    if not bootstrap_details or 'simulations' not in bootstrap_details:
        return None

    simulations = bootstrap_details['simulations']

    fig = go.Figure()

    # Histogram of simulations
    fig.add_trace(go.Histogram(
        x=simulations,
        nbinsx=50,
        name='IBNR Distribution',
        marker_color='#1a237e',
        opacity=0.7
    ))

    # Add mean line
    mean_val = bootstrap_details.get('mean', 0)
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: ${mean_val:,.0f}",
        annotation_position="top right"
    )

    # Add confidence intervals
    conf_95 = bootstrap_details.get('confidence_95', [0, 0])
    conf_90 = bootstrap_details.get('confidence_90', [0, 0])

    fig.add_vrect(
        x0=conf_95[0], x1=conf_95[1],
        fillcolor="rgba(255, 0, 0, 0.1)",
        line_width=0,
        annotation_text="95% CI",
        annotation_position="top left"
    )

    fig.add_vrect(
        x0=conf_90[0], x1=conf_90[1],
        fillcolor="rgba(0, 255, 0, 0.1)",
        line_width=0,
        annotation_text="90% CI",
        annotation_position="top right"
    )

    fig.update_layout(
        title="Bootstrap IBNR Distribution with Confidence Intervals",
        xaxis_title="IBNR Amount ($)",
        yaxis_title="Frequency",
        height=400,
        template="plotly_white",
        showlegend=False
    )

    return fig


# Excel Chain Ladder specific charts
def create_excel_chain_ladder_chart(ibnr_details):
    """Create visualization for Excel Chain Ladder model"""
    if not ibnr_details or 'cumulative_triangle' not in ibnr_details:
        return None

    cum_triangle = ibnr_details['cumulative_triangle']

    # Create heatmap of cumulative triangle
    fig = px.imshow(
        cum_triangle,
        title="Excel Chain Ladder - Cumulative Claims Triangle",
        labels=dict(x="Development Lag (Months)", y="Accident Period", color="Cumulative Claims"),
        aspect="auto",
        color_continuous_scale="Blues",
        text_auto=False
    )

    fig.update_layout(
        height=500,
        template="plotly_white",
        xaxis_title="Development Lag (Months)",
        yaxis_title="Accident Period (Year-Month)"
    )

    return fig


def create_development_factors_chart(development_factors):
    """Create bar chart of development factors"""
    if not development_factors:
        return None

    # Convert to DataFrame for plotting
    factors_df = pd.DataFrame({
        'Transition': list(development_factors.keys()),
        'Factor': list(development_factors.values())
    })

    fig = px.bar(
        factors_df,
        x='Transition',
        y='Factor',
        title="Development Factors (Age-to-Age)",
        labels={'Transition': 'Lag Transition', 'Factor': 'Development Factor'},
        color='Factor',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        height=400,
        template="plotly_white",
        xaxis_tickangle=-45
    )

    # Add horizontal line at 1.0
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="No Development")

    return fig


# Sidebar
with st.sidebar:
    st.markdown("## 📊 Actuarial Dashboard")
    st.markdown("### Operations Department")
    st.markdown("---")

    # Claims file uploader
    st.markdown("### 📁 Upload Claims Data")
    claims_file = st.file_uploader(
        "Choose claims Excel file",
        type=['xlsx', 'xls', 'csv'],
        key="claims_uploader",
        help="Upload Excel file containing claims data"
    )

    if claims_file is not None:
        if st.button("Process Claims", type="primary", use_container_width=True):
            with st.spinner("Processing claims file..."):
                st.session_state.claims_data = None
                st.session_state.claims_metrics = None
                st.session_state.ibnr_estimate = None
                st.session_state.ibnr_results = None

                if claims_file.name.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(claims_file)
                        df = clean_column_names(df)
                        st.session_state.claims_data = df
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                else:
                    df = parse_claims_file(claims_file)
                    if df is not None:
                        st.session_state.claims_data = df

                if st.session_state.claims_data is not None:
                    st.session_state.claims_metrics = calculate_claims_metrics(st.session_state.claims_data)
                    st.session_state.claims_file_name = claims_file.name
                    st.success("Claims file processed!")

    # Premium file uploader
    st.markdown("### 💰 Upload Premium Data")
    premium_file = st.file_uploader(
        "Choose premium Excel file",
        type=['xlsx', 'xls', 'csv'],
        key="premium_uploader",
        help="Upload Excel file containing premium data"
    )

    if premium_file is not None:
        if st.button("Process Premium", type="secondary", use_container_width=True):
            with st.spinner("Processing premium file..."):
                st.session_state.premium_data = None
                st.session_state.premium_metrics = None

                if premium_file.name.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(premium_file)
                        df = clean_column_names(df)
                        st.session_state.premium_data = df
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                else:
                    df = parse_premium_file(premium_file)
                    if df is not None:
                        st.session_state.premium_data = df

                if st.session_state.premium_data is not None:
                    st.session_state.premium_metrics = calculate_premium_metrics(st.session_state.premium_data)
                    st.session_state.premium_file_name = premium_file.name
                    st.success("Premium file processed!")

    st.markdown("---")

    # Data info
    data_sections = []

    if st.session_state.claims_data is not None:
        data_sections.append(f"""
        **Claims File:** {st.session_state.claims_file_name}
        **Claims Records:** {len(st.session_state.claims_data):,}
        **Unique Members (Claims):** {st.session_state.claims_metrics.get('unique_members', 0):,}
        """)

    if st.session_state.premium_data is not None:
        data_sections.append(f"""
        **Premium File:** {st.session_state.premium_file_name}
        **Premium Records:** {len(st.session_state.premium_data):,}
        **Unique Members (Premium):** {st.session_state.premium_metrics.get('unique_members', 0):,}
        **Total Premium:** ${st.session_state.premium_metrics.get('total_premium', 0):,.2f}
        """)

    if data_sections:
        st.markdown("### 📋 Data Summary")
        st.info("\n".join(data_sections))

        # Combined analysis if both datasets available
        if st.session_state.claims_data is not None and st.session_state.premium_data is not None:
            st.markdown("---")
            st.markdown("### 🔗 Combined Analysis")

            # Calculate combined metrics
            total_claims = st.session_state.claims_metrics.get('total_claimed', 0)
            total_premium = st.session_state.premium_metrics.get('total_premium', 0)

            if total_premium > 0:
                loss_ratio = (total_claims / total_premium) * 100
                st.metric("Loss Ratio", f"{loss_ratio:.1f}%")

            # Common members
            if 'MEMBER_NO' in st.session_state.claims_data.columns and 'MEMBER_NO' in st.session_state.premium_data.columns:
                claims_members = set(st.session_state.claims_data['MEMBER_NO'].unique())
                premium_members = set(st.session_state.premium_data['MEMBER_NO'].unique())
                common_members = len(claims_members.intersection(premium_members))

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Common Members", common_members)
                with col2:
                    coverage = (common_members / len(premium_members) * 100) if len(premium_members) > 0 else 0
                    st.metric("Coverage", f"{coverage:.1f}%")

        # IBNR estimate if available
        if st.session_state.ibnr_estimate is not None:
            st.markdown("---")
            st.markdown("### 📈 IBNR Estimate")
            ibnr_value = st.session_state.ibnr_estimate
            st.metric("Estimated IBNR", f"${ibnr_value:,.0f}")

    st.markdown("---")
    st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Main content
if st.session_state.claims_data is None and st.session_state.premium_data is None:
    # Upload page
    st.markdown('<div class="main-header">Actuarial Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Health Insurance Analysis Platform</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Claims Data")
        st.markdown("Analyze claims patterns and utilization")
        st.markdown("")
        st.markdown("**Expected columns:**")
        st.markdown("- Member Number")
        st.markdown("- Claim Amount")
        st.markdown("- Paid Amount")
        st.markdown("- Service Date")
        st.markdown("- Provider Name")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="upload-section" style="background: #f0f7ff;">', unsafe_allow_html=True)
        st.markdown("### 💰 Upload Premium Data")
        st.markdown("Analyze premium collection and membership")
        st.markdown("")
        st.markdown("**Expected columns:**")
        st.markdown("- Member Number")
        st.markdown("- Premium Amount")
        st.markdown("- Product Name")
        st.markdown("- Payer Name")
        st.markdown("- Join Date")
        st.markdown('</div>', unsafe_allow_html=True)

    # Sample data preview
    with st.expander("📋 Sample Premium Data Format", expanded=True):
        sample_data = {
            'Member': [13574465, 13574474, 13574492],
            'Name': ['MR CHAMUNORWA NYAMAKURA', 'MRS RACHAEL MUTENDI', 'MR MUNYARADZI MOYO'],
            'BirthDate': ['1953/07/09', '1963/03/11', '1998/07/07'],
            'Payer Name': ['SELF PAYING', 'SELF PAYING', 'SELF PAYING'],
            'Join Date': ['2025/02/01', '2025/02/01', '2025/02/01'],
            'Product Name': ['GENLINK', 'GENCORE', 'GENCARE'],
            'Premium': [121, 16, 69]
        }
        st.dataframe(pd.DataFrame(sample_data))

else:
    # Create tabs based on available data
    tab_names = []

    if st.session_state.claims_data is not None:
        tab_names.extend(
            ["📈 Claims Overview", "💰 Claims Financial", "👥 Claims Members", "🏥 Providers", "📊 IBNR Analysis"])

    if st.session_state.premium_data is not None:
        tab_names.extend(["💎 Premium Analysis", "📊 Premium Details"])

    if st.session_state.claims_data is not None and st.session_state.premium_data is not None:
        tab_names.append("🔗 Combined View")

    tab_names.append("📁 Data Export")

    tabs = st.tabs(tab_names)

    # Claims Overview Tab
    if "📈 Claims Overview" in tab_names:
        claims_overview_index = tab_names.index("📈 Claims Overview")
        with tabs[claims_overview_index]:
            st.markdown('<div class="main-header">Claims Overview</div>', unsafe_allow_html=True)
            metrics = st.session_state.claims_metrics

            # Key metrics row 1
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total Claims</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{metrics.get("total_claims", 0):,}</div>',
                            unsafe_allow_html=True)
                st.caption("Number of claim records")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total Claimed</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${metrics.get("total_claimed", 0):,.2f}</div>',
                            unsafe_allow_html=True)
                st.caption("Total amount claimed")
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total Paid</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${metrics.get("total_paid", 0):,.2f}</div>',
                            unsafe_allow_html=True)
                st.caption("Total amount paid")
                st.markdown('</div>', unsafe_allow_html=True)

            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Paid Ratio</div>', unsafe_allow_html=True)
                paid_ratio = metrics.get("paid_ratio", 0) * 100
                st.markdown(f'<div class="metric-value">{paid_ratio:.1f}%</div>', unsafe_allow_html=True)
                st.caption("Paid / Claimed ratio")
                st.markdown('</div>', unsafe_allow_html=True)

            # Key metrics row 2 (with IBNR)
            col5, col6, col7, col8 = st.columns(4)

            with col5:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Unique Members</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{metrics.get("unique_members", 0):,}</div>',
                            unsafe_allow_html=True)
                st.caption("Number of unique members")
                st.markdown('</div>', unsafe_allow_html=True)

            with col6:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Avg Claim</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${metrics.get("avg_claim_amount", 0):,.2f}</div>',
                            unsafe_allow_html=True)
                st.caption("Average claim amount")
                st.markdown('</div>', unsafe_allow_html=True)

            with col7:
                ibnr_value = st.session_state.ibnr_estimate or 0
                st.markdown('<div class="ibnr-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label" style="color: white;">Estimated IBNR</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color: white;">${ibnr_value:,.0f}</div>',
                            unsafe_allow_html=True)
                st.caption("Incurred But Not Reported")
                st.markdown('</div>', unsafe_allow_html=True)

            with col8:
                total_claims = metrics.get("total_claimed", 0)
                ibnr_ratio = (ibnr_value / total_claims * 100) if total_claims > 0 else 0
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">IBNR Ratio</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{ibnr_ratio:.1f}%</div>', unsafe_allow_html=True)
                st.caption("IBNR/Reported Claims")
                st.markdown('</div>', unsafe_allow_html=True)

            # Charts
            if 'monthly_trends' in metrics and not metrics['monthly_trends'].empty:
                fig = create_monthly_trend_chart(metrics['monthly_trends'])
                st.plotly_chart(fig, use_container_width=True)

    # Claims Financial Tab
    if "💰 Claims Financial" in tab_names:
        claims_financial_index = tab_names.index("💰 Claims Financial")
        with tabs[claims_financial_index]:
            st.markdown('<div class="main-header">Claims Financial Analysis</div>', unsafe_allow_html=True)
            metrics = st.session_state.claims_metrics

            # Financial KPIs
            st.markdown('<div class="sub-header">Key Financial Indicators</div>', unsafe_allow_html=True)
            fig = create_financial_kpi_chart(metrics)
            st.plotly_chart(fig, use_container_width=True)

            # Financial details
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="sub-header">Payment Sources Analysis</div>', unsafe_allow_html=True)

                # Check for payment source columns
                payment_sources_data = []

                # Check various payment source columns
                df = st.session_state.claims_data

                # Look for common payment source column names
                payment_columns = {
                    'PAID_FROM_RISK_AMT': 'Risk Pool',
                    'PAID_FROM_THRESHHOLD': 'Threshold',
                    'PAID_FROM_SAVINGS': 'Savings',
                    'PAID_FROM_ACCUMULATOR': 'Accumulator',
                    'COPAY': 'Co-payment',
                    'DEDUCTIBLE': 'Deductible'
                }

                for col, source_name in payment_columns.items():
                    if col in df.columns:
                        try:
                            amount = float(df[col].sum())
                            if amount > 0:
                                payment_sources_data.append({
                                    'Source': source_name,
                                    'Amount': amount,
                                    'Percentage': (amount / metrics.get('total_paid', 1) * 100) if metrics.get(
                                        'total_paid', 0) > 0 else 0
                                })
                        except:
                            pass

                if payment_sources_data:
                    payment_df = pd.DataFrame(payment_sources_data)
                    fig = px.pie(
                        payment_df,
                        values='Amount',
                        names='Source',
                        title="Payment Sources Distribution",
                        color_discrete_sequence=px.colors.sequential.Blues,
                        hover_data=['Percentage']
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)

                    # Payment source table
                    st.markdown("**Payment Source Details:**")
                    payment_df['Amount'] = payment_df['Amount'].apply(lambda x: f"${x:,.2f}")
                    payment_df['Percentage'] = payment_df['Percentage'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(payment_df[['Source', 'Amount', 'Percentage']], use_container_width=True)
                else:
                    st.info(
                        "Payment source data not available. Add columns like PAID_FROM_RISK_AMT, PAID_FROM_THRESHHOLD, etc.")

            with col2:
                st.markdown('<div class="sub-header">Top Claims by Amount</div>', unsafe_allow_html=True)

                if 'AMOUNT_CLAIMED' in df.columns and 'TOTAL_PAID' in df.columns:
                    try:
                        # Get top 10 claims
                        top_claims = df.nlargest(10, 'AMOUNT_CLAIMED')

                        # Select available columns for display
                        display_columns = []
                        column_mapping = {
                            'MEMBER_NO': 'Member ID',
                            'SERVICE_DATE': 'Service Date',
                            'AMOUNT_CLAIMED': 'Amount Claimed ($)',
                            'TOTAL_PAID': 'Amount Paid ($)',
                            'PROVIDER_NAME': 'Provider',
                            'BASE_BENEFIT_DESCRIPTION': 'Service Type'
                        }

                        for col, display_name in column_mapping.items():
                            if col in top_claims.columns:
                                display_columns.append(col)

                        if display_columns:
                            # Create display dataframe with unique column names
                            top_claims_display = top_claims[display_columns].copy()

                            # Apply unique display names
                            rename_dict = {}
                            for col in display_columns:
                                if col in column_mapping:
                                    rename_dict[col] = column_mapping[col]

                            # Rename columns
                            top_claims_display = top_claims_display.rename(columns=rename_dict)

                            # Calculate Paid % as a separate column
                            if 'Amount Claimed ($)' in top_claims_display.columns and 'Amount Paid ($)' in top_claims_display.columns:
                                # Get original values for calculation
                                original_claimed = top_claims['AMOUNT_CLAIMED'].values
                                original_paid = top_claims['TOTAL_PAID'].values

                                # Calculate paid percentage
                                paid_percentages = []
                                for i in range(len(original_claimed)):
                                    if original_claimed[i] > 0:
                                        paid_percentages.append(
                                            f"{(original_paid[i] / original_claimed[i] * 100):.1f}%")
                                    else:
                                        paid_percentages.append("0.0%")

                                # Add Paid % column
                                top_claims_display['Paid %'] = paid_percentages

                            # Format currency columns
                            if 'Amount Claimed ($)' in top_claims_display.columns:
                                top_claims_display['Amount Claimed ($)'] = top_claims_display[
                                    'Amount Claimed ($)'].apply(
                                    lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x
                                )

                            if 'Amount Paid ($)' in top_claims_display.columns:
                                top_claims_display['Amount Paid ($)'] = top_claims_display['Amount Paid ($)'].apply(
                                    lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x
                                )

                            # Format date column if present
                            if 'Service Date' in top_claims_display.columns:
                                top_claims_display['Service Date'] = top_claims_display['Service Date'].apply(
                                    lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else ''
                                )

                            st.dataframe(
                                top_claims_display,
                                use_container_width=True,
                                height=400
                            )
                        else:
                            st.info("No claim data available for display")
                    except Exception as e:
                        st.error(f"Error displaying top claims: {str(e)}")
                        st.info("Try checking the data format and column names")
                else:
                    st.info("Claim amount data not available")

            # Monthly financial analysis
            st.markdown('<div class="sub-header">Monthly Financial Summary</div>', unsafe_allow_html=True)

            if 'monthly_trends' in metrics and not metrics['monthly_trends'].empty:
                monthly_financial = metrics['monthly_trends'].copy()

                # Format for display
                monthly_financial_display = monthly_financial.copy()
                monthly_financial_display['AMOUNT_CLAIMED'] = monthly_financial_display['AMOUNT_CLAIMED'].apply(
                    lambda x: f"${x:,.2f}")
                monthly_financial_display['TOTAL_PAID'] = monthly_financial_display['TOTAL_PAID'].apply(
                    lambda x: f"${x:,.2f}")

                if 'AMOUNT_CLAIMED' in monthly_financial.columns and 'TOTAL_PAID' in monthly_financial.columns:
                    monthly_financial_display['Paid %'] = (
                                (monthly_financial['TOTAL_PAID'] / monthly_financial['AMOUNT_CLAIMED']) * 100).round(
                        1).apply(lambda x: f"{x:.1f}%")
                    monthly_financial_display['Variance'] = (
                                monthly_financial['AMOUNT_CLAIMED'] - monthly_financial['TOTAL_PAID']).apply(
                        lambda x: f"${x:,.2f}")

                monthly_financial_display = monthly_financial_display.rename(columns={
                    'month_year': 'Month',
                    'AMOUNT_CLAIMED': 'Amount Claimed',
                    'TOTAL_PAID': 'Amount Paid',
                    'MEMBER_NO': 'Unique Members'
                })

                st.dataframe(
                    monthly_financial_display,
                    use_container_width=True
                )

                # Monthly financial chart
                if len(monthly_financial) > 1:
                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        x=monthly_financial['month_year'],
                        y=monthly_financial['AMOUNT_CLAIMED'],
                        name='Amount Claimed',
                        marker_color='#1a237e'
                    ))

                    fig.add_trace(go.Bar(
                        x=monthly_financial['month_year'],
                        y=monthly_financial['TOTAL_PAID'],
                        name='Amount Paid',
                        marker_color='#4caf50'
                    ))

                    fig.update_layout(
                        title="Monthly Claims vs Paid Amounts",
                        xaxis_title="Month",
                        yaxis_title="Amount ($)",
                        barmode='group',
                        template="plotly_white",
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Monthly financial data not available")

    # Claims Members Tab
    if "👥 Claims Members" in tab_names:
        claims_members_index = tab_names.index("👥 Claims Members")
        with tabs[claims_members_index]:
            st.markdown('<div class="main-header">Member Analytics</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="sub-header">Member Demographics</div>', unsafe_allow_html=True)

                # Check for age data
                df = st.session_state.claims_data
                if 'CURRENT_AGE' in df.columns:
                    try:
                        # Create age groups
                        df_temp = df.copy()
                        df_temp['AGE_GROUP'] = pd.cut(
                            df_temp['CURRENT_AGE'],
                            bins=[0, 18, 30, 40, 50, 60, 70, 100],
                            labels=['0-18', '19-30', '31-40', '41-50', '51-60', '61-70', '71+'],
                            include_lowest=True
                        )
                        age_dist = df_temp.groupby('AGE_GROUP').size().reset_index(name='count')

                        fig = px.bar(
                            age_dist,
                            x='AGE_GROUP',
                            y='count',
                            title="Age Distribution of Claimants",
                            labels={'AGE_GROUP': 'Age Group', 'count': 'Number of Claims'},
                            color='count',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.info("Could not create age distribution chart")
                else:
                    st.info("Age data not available for members")

                # Gender distribution
                if 'GENDER' in df.columns:
                    gender_dist = df['GENDER'].value_counts().reset_index()
                    gender_dist.columns = ['gender', 'count']

                    if not gender_dist.empty:
                        fig = px.pie(
                            gender_dist,
                            values='count',
                            names='gender',
                            title="Gender Distribution",
                            color_discrete_sequence=px.colors.sequential.Blues,
                            hole=0.4
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No gender data available")
                else:
                    st.info("Gender data not available")

            with col2:
                st.markdown('<div class="sub-header">Top Members by Claims</div>', unsafe_allow_html=True)

                if 'MEMBER_NO' in df.columns and 'AMOUNT_CLAIMED' in df.columns:
                    try:
                        # Calculate member-level statistics
                        member_stats = df.groupby('MEMBER_NO').agg({
                            'AMOUNT_CLAIMED': ['sum', 'count'],
                            'TOTAL_PAID': 'sum'
                        }).reset_index()

                        # Flatten column names
                        member_stats.columns = ['MEMBER_NO', 'TOTAL_CLAIMED', 'CLAIM_COUNT', 'TOTAL_PAID']

                        # Get top members by claim count
                        top_members = member_stats.nlargest(10, 'CLAIM_COUNT')

                        # Create bar chart
                        fig = px.bar(
                            top_members,
                            x='MEMBER_NO',
                            y='CLAIM_COUNT',
                            title="Top Members by Number of Claims",
                            labels={'MEMBER_NO': 'Member ID', 'CLAIM_COUNT': 'Number of Claims'},
                            color='TOTAL_CLAIMED',
                            color_continuous_scale='Viridis',
                            hover_data=['TOTAL_CLAIMED', 'TOTAL_PAID']
                        )
                        fig.update_layout(xaxis={'type': 'category'})
                        st.plotly_chart(fig, use_container_width=True)

                        # Member details table
                        st.markdown("**Top Members Details:**")
                        member_details = top_members.copy()
                        member_details['TOTAL_CLAIMED'] = member_details['TOTAL_CLAIMED'].apply(lambda x: f"${x:,.2f}")
                        member_details['TOTAL_PAID'] = member_details['TOTAL_PAID'].apply(lambda x: f"${x:,.2f}")
                        member_details['Paid %'] = ((member_details['TOTAL_PAID'].str.replace('$', '').str.replace(',',
                                                                                                                   '').astype(
                            float) /
                                                     member_details['TOTAL_CLAIMED'].str.replace('$', '').str.replace(
                                                         ',', '').astype(float)) * 100).round(1).apply(
                            lambda x: f"{x:.1f}%")

                        st.dataframe(
                            member_details[['MEMBER_NO', 'CLAIM_COUNT', 'TOTAL_CLAIMED', 'TOTAL_PAID', 'Paid %']],
                            use_container_width=True
                        )
                    except Exception as e:
                        st.info(f"Error analyzing member data: {str(e)}")
                else:
                    st.info("Member ID or claim amount data not available")

            # Member segmentation analysis
            st.markdown('<div class="sub-header">Member Segmentation Analysis</div>', unsafe_allow_html=True)

            if 'MEMBER_NO' in df.columns and 'AMOUNT_CLAIMED' in df.columns:
                try:
                    # Create member segments based on claim frequency
                    member_claims = df.groupby('MEMBER_NO').agg({
                        'AMOUNT_CLAIMED': 'sum',
                        'TOTAL_PAID': 'sum'
                    }).reset_index()
                    member_claims['claim_count'] = df.groupby('MEMBER_NO').size().values

                    # Segment members
                    member_claims['Segment'] = pd.cut(
                        member_claims['claim_count'],
                        bins=[0, 1, 3, 5, float('inf')],
                        labels=['One-time (1 claim)', 'Occasional (2-3 claims)', 'Regular (4-5 claims)',
                                'Frequent (6+ claims)']
                    )

                    # Calculate segment summary
                    segment_summary = member_claims.groupby('Segment').agg({
                        'MEMBER_NO': 'count',
                        'AMOUNT_CLAIMED': 'sum',
                        'TOTAL_PAID': 'sum',
                        'claim_count': 'mean'
                    }).reset_index()

                    segment_summary = segment_summary.rename(columns={
                        'MEMBER_NO': 'Member Count',
                        'AMOUNT_CLAIMED': 'Total Claimed',
                        'TOTAL_PAID': 'Total Paid',
                        'claim_count': 'Avg Claims per Member'
                    })

                    # Create visualization
                    col3, col4 = st.columns(2)

                    with col3:
                        fig = px.sunburst(
                            segment_summary,
                            path=['Segment'],
                            values='Total Claimed',
                            title="Member Segmentation by Claim Frequency",
                            color='Member Count',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col4:
                        # Format segment summary for display
                        display_summary = segment_summary.copy()
                        display_summary['Total Claimed'] = display_summary['Total Claimed'].apply(
                            lambda x: f"${x:,.2f}")
                        display_summary['Total Paid'] = display_summary['Total Paid'].apply(lambda x: f"${x:,.2f}")
                        display_summary['Avg Claims per Member'] = display_summary['Avg Claims per Member'].apply(
                            lambda x: f"{x:.1f}")

                        st.dataframe(
                            display_summary,
                            use_container_width=True
                        )
                except Exception as e:
                    st.info(f"Member segmentation analysis not available: {str(e)}")
            else:
                st.info("Member segmentation data not available")

    # Providers Tab
    if "🏥 Providers" in tab_names:
        providers_index = tab_names.index("🏥 Providers")
        with tabs[providers_index]:
            st.markdown('<div class="main-header">Provider Analysis</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="sub-header">Top 10 Providers by Claims Volume</div>', unsafe_allow_html=True)

                df = st.session_state.claims_data
                if 'PROVIDER_NAME' in df.columns and 'AMOUNT_CLAIMED' in df.columns:
                    try:
                        # Calculate provider statistics
                        provider_stats = df.groupby('PROVIDER_NAME').agg({
                            'AMOUNT_CLAIMED': 'sum',
                            'TOTAL_PAID': 'sum',
                            'MEMBER_NO': 'nunique'
                        }).reset_index()

                        provider_stats = provider_stats.sort_values('AMOUNT_CLAIMED', ascending=False).head(10)
                        provider_stats['Paid %'] = (
                                    provider_stats['TOTAL_PAID'] / provider_stats['AMOUNT_CLAIMED'] * 100).round(1)

                        # Create grouped bar chart
                        fig = px.bar(
                            provider_stats,
                            x='PROVIDER_NAME',
                            y=['AMOUNT_CLAIMED', 'TOTAL_PAID'],
                            title="Top Providers - Claimed vs Paid Amounts",
                            barmode='group',
                            labels={'value': 'Amount ($)', 'variable': 'Type', 'PROVIDER_NAME': 'Provider'},
                            color_discrete_map={'AMOUNT_CLAIMED': '#1a237e', 'TOTAL_PAID': '#4caf50'}
                        )

                        # Truncate long provider names for better display
                        fig.update_layout(
                            xaxis={'tickangle': -45, 'tickmode': 'array'},
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.info(f"Error creating provider chart: {str(e)}")
                else:
                    st.info("Provider name or claim amount data not available")

            with col2:
                st.markdown('<div class="sub-header">Provider Efficiency Analysis</div>', unsafe_allow_html=True)

                if 'PROVIDER_NAME' in df.columns and 'AMOUNT_CLAIMED' in df.columns and 'TOTAL_PAID' in df.columns:
                    try:
                        provider_stats = df.groupby('PROVIDER_NAME').agg({
                            'AMOUNT_CLAIMED': 'sum',
                            'TOTAL_PAID': 'sum',
                            'MEMBER_NO': 'nunique'
                        }).reset_index()

                        # Calculate efficiency metrics
                        provider_stats['Efficiency'] = (
                                    provider_stats['TOTAL_PAID'] / provider_stats['AMOUNT_CLAIMED'] * 100).round(1)
                        provider_stats['Avg Claim per Member'] = (
                                    provider_stats['AMOUNT_CLAIMED'] / provider_stats['MEMBER_NO']).round(2)

                        # Filter for providers with sufficient data
                        filtered_providers = provider_stats[provider_stats['AMOUNT_CLAIMED'] > 0].nlargest(15,
                                                                                                           'AMOUNT_CLAIMED')

                        if not filtered_providers.empty:
                            fig = px.scatter(
                                filtered_providers,
                                x='AMOUNT_CLAIMED',
                                y='Efficiency',
                                size='MEMBER_NO',
                                color='Avg Claim per Member',
                                hover_name='PROVIDER_NAME',
                                title="Provider Efficiency Analysis",
                                labels={
                                    'AMOUNT_CLAIMED': 'Total Claimed ($)',
                                    'Efficiency': 'Paid %',
                                    'MEMBER_NO': 'Unique Members',
                                    'Avg Claim per Member': 'Avg Claim/Member ($)'
                                },
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Insufficient data for provider efficiency analysis")
                    except Exception as e:
                        st.info(f"Error analyzing provider efficiency: {str(e)}")
                else:
                    st.info("Provider efficiency data not available")

            # Provider details table
            st.markdown('<div class="sub-header">Provider Performance Details</div>', unsafe_allow_html=True)

            if 'PROVIDER_NAME' in df.columns and 'AMOUNT_CLAIMED' in df.columns:
                try:
                    provider_details = df.groupby('PROVIDER_NAME').agg({
                        'AMOUNT_CLAIMED': ['sum', 'mean', 'count'],
                        'TOTAL_PAID': 'sum',
                        'MEMBER_NO': 'nunique'
                    }).reset_index()

                    # Flatten column names
                    provider_details.columns = ['PROVIDER_NAME', 'Total_Claimed', 'Avg_Claim', 'Claim_Count',
                                                'Total_Paid', 'Unique_Members']

                    # Calculate additional metrics
                    provider_details['Paid_%'] = (
                                provider_details['Total_Paid'] / provider_details['Total_Claimed'] * 100).round(1)
                    provider_details['Claims_per_Member'] = (
                                provider_details['Claim_Count'] / provider_details['Unique_Members']).round(2)

                    # Sort by total claimed
                    provider_details = provider_details.sort_values('Total_Claimed', ascending=False).head(20)

                    # Format for display
                    display_details = provider_details.copy()
                    display_details['Total_Claimed'] = display_details['Total_Claimed'].apply(lambda x: f"${x:,.2f}")
                    display_details['Total_Paid'] = display_details['Total_Paid'].apply(lambda x: f"${x:,.2f}")
                    display_details['Avg_Claim'] = display_details['Avg_Claim'].apply(lambda x: f"${x:,.2f}")
                    display_details['Paid_%'] = display_details['Paid_%'].apply(lambda x: f"{x:.1f}%")

                    display_details = display_details.rename(columns={
                        'PROVIDER_NAME': 'Provider',
                        'Total_Claimed': 'Total Claimed',
                        'Total_Paid': 'Total Paid',
                        'Avg_Claim': 'Average Claim',
                        'Claim_Count': 'Claim Count',
                        'Unique_Members': 'Unique Members',
                        'Paid_%': 'Paid %',
                        'Claims_per_Member': 'Claims/Member'
                    })

                    st.dataframe(
                        display_details,
                        use_container_width=True,
                        height=500
                    )
                except Exception as e:
                    st.info(f"Provider details not available: {str(e)}")
            else:
                st.info("Provider performance data not available")

            # Provider type analysis (if available)
            if 'PROVIDER_TYPE' in df.columns or 'PROVIDER_CATEGORY' in df.columns:
                st.markdown('<div class="sub-header">Provider Type Analysis</div>', unsafe_allow_html=True)

                provider_type_col = 'PROVIDER_TYPE' if 'PROVIDER_TYPE' in df.columns else 'PROVIDER_CATEGORY'

                try:
                    type_analysis = df.groupby(provider_type_col).agg({
                        'AMOUNT_CLAIMED': 'sum',
                        'TOTAL_PAID': 'sum',
                        'MEMBER_NO': 'nunique',
                        'PROVIDER_NAME': 'nunique'
                    }).reset_index()

                    type_analysis = type_analysis.sort_values('AMOUNT_CLAIMED', ascending=False)

                    col5, col6 = st.columns(2)

                    with col5:
                        fig = px.bar(
                            type_analysis.head(10),
                            x=provider_type_col,
                            y='AMOUNT_CLAIMED',
                            title="Claims by Provider Type",
                            labels={provider_type_col: 'Provider Type', 'AMOUNT_CLAIMED': 'Total Claimed ($)'},
                            color='AMOUNT_CLAIMED',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col6:
                        # Format for display
                        display_type = type_analysis.copy()
                        display_type['AMOUNT_CLAIMED'] = display_type['AMOUNT_CLAIMED'].apply(lambda x: f"${x:,.2f}")
                        display_type['TOTAL_PAID'] = display_type['TOTAL_PAID'].apply(lambda x: f"${x:,.2f}")
                        display_type = display_type.rename(columns={
                            provider_type_col: 'Provider Type',
                            'AMOUNT_CLAIMED': 'Total Claimed',
                            'TOTAL_PAID': 'Total Paid',
                            'MEMBER_NO': 'Unique Members',
                            'PROVIDER_NAME': 'Unique Providers'
                        })

                        st.dataframe(
                            display_type,
                            use_container_width=True
                        )
                except:
                    st.info("Provider type analysis not available")

    # IBNR Analysis Tab - UPDATED with Excel Chain Ladder Model
    if "📊 IBNR Analysis" in tab_names:
        ibnr_tab_index = tab_names.index("📊 IBNR Analysis")
        with tabs[ibnr_tab_index]:
            st.markdown('<div class="main-header">IBNR Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">Incurred But Not Reported Claims Estimation</div>',
                        unsafe_allow_html=True)

            # Excel Model Upload Section
            st.markdown("### 📋 Excel Chain Ladder Model")
            col_upload, col_info = st.columns([2, 1])

            with col_upload:
                excel_model_file = st.file_uploader(
                    "Upload Excel Chain Ladder Model (Optional)",
                    type=['xlsx'],
                    help="Upload your IBNR Model.xlsx file for comparison"
                )

            with col_info:
                st.info("""
                **Excel Model Info:**
                - Based on Chain Ladder method
                - Uses development factors
                - Projects cumulative claims to ultimate
                """)

            # Main IBNR Calculation
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🎯 IBNR Calculation Methods")

                # Method selection - include Excel model
                selected_methods = st.multiselect(
                    "Select IBNR calculation methods:",
                    options=['Excel Chain Ladder', 'Chain Ladder', 'Bornhuetter-Ferguson', 'Frequency-Severity',
                             'Bootstrap'],
                    default=['Excel Chain Ladder', 'Chain Ladder', 'Bornhuetter-Ferguson']
                )

                # Parameters
                with st.expander("⚙️ Method Parameters", expanded=False):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        loss_ratio = st.slider("Expected Loss Ratio", 0.1, 1.0, 0.75, 0.05)
                        months_unreported = st.slider("Months of Unreported Claims", 1, 12, 3)

                    with col_b:
                        n_simulations = st.number_input("Bootstrap Simulations", 100, 10000, 1000, step=100)
                        tail_factor = st.number_input("Tail Factor", 1.0, 2.0, 1.05, 0.01)

                # Calculate button
                if st.button("Calculate IBNR", type="primary", use_container_width=True):
                    with st.spinner("Calculating IBNR using selected methods..."):
                        # Convert selected methods to format for composite function
                        method_map = {
                            'Excel Chain Ladder': 'excel',
                            'Chain Ladder': 'chain_ladder',
                            'Bornhuetter-Ferguson': 'bornhuetter',
                            'Frequency-Severity': 'frequency',
                            'Bootstrap': 'bootstrap'
                        }

                        selected_methods_formatted = [method_map[m] for m in selected_methods if m in method_map]

                        # Calculate IBNR
                        composite_ibnr, ibnr_details = calculate_composite_ibnr(
                            st.session_state.claims_data,
                            methods=selected_methods_formatted
                        )

                        st.session_state.ibnr_estimate = composite_ibnr
                        st.session_state.ibnr_method = ibnr_details.get('method', 'Composite')

                        if 'component_results' in ibnr_details:
                            st.session_state.ibnr_results = ibnr_details['component_results']
                        else:
                            st.session_state.ibnr_results = None

                        st.success(f"IBNR calculated successfully!")
                        st.rerun()

            with col2:
                st.markdown("### 📊 IBNR Estimate")

                if st.session_state.ibnr_estimate is not None:
                    # Display IBNR metrics
                    st.markdown('<div class="ibnr-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label" style="color: white;">Estimated IBNR</div>',
                                unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-value" style="color: white; font-size: 2.5rem;">${st.session_state.ibnr_estimate:,.0f}</div>',
                        unsafe_allow_html=True)
                    st.caption(f"Method: {st.session_state.ibnr_method}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Calculate IBNR ratio
                    reported_claims = st.session_state.claims_metrics.get('total_claimed', 0)
                    if reported_claims > 0:
                        ibnr_ratio = (st.session_state.ibnr_estimate / reported_claims) * 100

                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">IBNR Ratio</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-value">{ibnr_ratio:.1f}%</div>', unsafe_allow_html=True)
                        st.caption("IBNR as % of Reported Claims")
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">Estimated Ultimate</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="metric-value">${reported_claims + st.session_state.ibnr_estimate:,.0f}</div>',
                            unsafe_allow_html=True)
                        st.caption("Reported + IBNR")
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Click 'Calculate IBNR' to generate estimates")

            # Display Excel Model Results if available
            ibnr_results = st.session_state.ibnr_results
            if ibnr_results and 'Excel Chain Ladder' in ibnr_results:
                st.markdown("### 📐 Excel Chain Ladder Model Details")

                excel_details = ibnr_results['Excel Chain Ladder']['details']

                # Display development factors
                if 'development_factors' in excel_details:
                    st.markdown("**Development Factors:**")
                    factors_df = pd.DataFrame({
                        'Transition': list(excel_details['development_factors'].keys()),
                        'Factor': list(excel_details['development_factors'].values())
                    })
                    st.dataframe(factors_df, use_container_width=True)

                # Display triangles
                col3, col4 = st.columns(2)

                with col3:
                    if 'incremental_triangle' in excel_details:
                        inc_triangle = excel_details['incremental_triangle']
                        st.markdown("**Incremental Triangle:**")
                        st.dataframe(inc_triangle.style.format("${:,.0f}"), use_container_width=True, height=300)

                with col4:
                    if 'cumulative_triangle' in excel_details:
                        cum_triangle = excel_details['cumulative_triangle']
                        st.markdown("**Cumulative Triangle:**")
                        st.dataframe(cum_triangle.style.format("${:,.0f}"), use_container_width=True, height=300)

                # Display summary metrics
                st.markdown("**Model Summary:**")
                summary_data = {
                    'Metric': ['Total Reported Claims', 'Estimated Ultimate Claims', 'IBNR Reserve', 'IBNR Ratio'],
                    'Value': [
                        f"${excel_details.get('total_reported', 0):,.2f}",
                        f"${excel_details.get('total_ultimate', 0):,.2f}",
                        f"${excel_details.get('ibnr_estimate', 0):,.2f}",
                        f"{excel_details.get('ibnr_ratio', 0):.1f}%"
                    ]
                }
                st.table(pd.DataFrame(summary_data))

            # Method Comparison Chart
            if ibnr_results:
                st.markdown("### 🔍 Method Comparison")

                fig = create_ibnr_comparison_chart(ibnr_results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Display method details
                st.markdown("**Method Details:**")
                for method, data in ibnr_results.items():
                    with st.expander(f"{method}: ${data['ibnr']:,.0f}", expanded=False):
                        if data['details']:
                            details_df = pd.DataFrame([
                                {'Parameter': k, 'Value': v}
                                for k, v in data['details'].items()
                                if not isinstance(v, (pd.DataFrame, dict, list)) or k in ['method', 'tail_factor']
                            ])
                            st.table(details_df)

            # Development Triangle Visualization
            if 'Excel Chain Ladder' in ibnr_results:
                excel_details = ibnr_results['Excel Chain Ladder']['details']
                if 'cumulative_triangle' in excel_details:
                    fig = create_excel_chain_ladder_chart(excel_details)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            # Bootstrap Confidence Intervals (if available)
            if (ibnr_results and
                    'Bootstrap' in ibnr_results and
                    'details' in ibnr_results['Bootstrap'] and
                    'simulations' in ibnr_results['Bootstrap']['details']):

                st.markdown("### 📊 Bootstrap Analysis")
                fig = create_ibnr_confidence_chart(ibnr_results['Bootstrap']['details'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Display confidence intervals
                bootstrap_details = ibnr_results['Bootstrap']['details']
                if 'confidence_95' in bootstrap_details and 'confidence_90' in bootstrap_details:
                    col7, col8, col9 = st.columns(3)

                    with col7:
                        st.metric(
                            "95% Confidence Interval",
                            f"${bootstrap_details['confidence_95'][0]:,.0f} - ${bootstrap_details['confidence_95'][1]:,.0f}",
                            f"Range: ${bootstrap_details['confidence_95'][1] - bootstrap_details['confidence_95'][0]:,.0f}"
                        )

                    with col8:
                        st.metric(
                            "90% Confidence Interval",
                            f"${bootstrap_details['confidence_90'][0]:,.0f} - ${bootstrap_details['confidence_90'][1]:,.0f}",
                            f"Range: ${bootstrap_details['confidence_90'][1] - bootstrap_details['confidence_90'][0]:,.0f}"
                        )

                    with col9:
                        st.metric(
                            "Standard Deviation",
                            f"${bootstrap_details.get('std_dev', 0):,.0f}",
                            f"Coefficient of Variation: {(bootstrap_details.get('std_dev', 0) / bootstrap_details.get('mean', 1)) * 100:.1f}%" if bootstrap_details.get(
                                'mean', 0) > 0 else "N/A"
                        )

            # Actuarial Notes and Assumptions
            with st.expander("📝 Actuarial Notes & Assumptions", expanded=False):
                st.markdown("""
                **IBNR Calculation Methods:**

                1. **Excel Chain Ladder Method:**
                   - Replicates the Excel model methodology
                   - Uses development factors similar to Excel formulas
                   - Projects cumulative claims to ultimate using chain ladder

                2. **Chain Ladder Method:**
                   - Uses historical development patterns to project future claims
                   - Assumes past development patterns will continue
                   - Most widely used method for short-tailed lines

                3. **Bornhuetter-Ferguson Method:**
                   - Combines actual reported claims with expected loss ratio
                   - Less sensitive to recent volatility
                   - Good for long-tailed or volatile lines

                4. **Frequency-Severity Method:**
                   - Separates claim frequency and severity analysis
                   - Uses statistical distributions (Poisson for frequency)
                   - Good when claim counts are stable

                5. **Bootstrap Method:**
                   - Uses resampling to estimate uncertainty
                   - Provides confidence intervals
                   - Computationally intensive but robust

                **Key Assumptions:**
                - Claim reporting patterns are consistent
                - No systemic changes in claims handling
                - Adequate historical data for development patterns
                - Premium information available for loss ratio methods

                **Limitations:**
                - Results are estimates with inherent uncertainty
                - Quality depends on data completeness
                - May not capture sudden changes or "shocks"
                - Professional judgment required for final selection
                """)

    # Premium Analysis Tab
    if "💎 Premium Analysis" in tab_names:
        premium_tab_index = tab_names.index("💎 Premium Analysis")
        with tabs[premium_tab_index]:
            st.markdown('<div class="main-header">Premium Analysis</div>', unsafe_allow_html=True)
            metrics = st.session_state.premium_metrics

            # Premium metrics row 1
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label" style="color: white;">Total Premium</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="metric-value" style="color: white;">${metrics.get("total_premium", 0):,.2f}</div>',
                    unsafe_allow_html=True)
                st.caption("Total premium collected")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Active Members</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{metrics.get("active_members", 0):,}</div>',
                            unsafe_allow_html=True)
                st.caption(f"of {metrics.get('total_members', 0):,} total")
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Avg Premium</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${metrics.get("avg_premium", 0):,.2f}</div>',
                            unsafe_allow_html=True)
                st.caption("Average premium per member")
                st.markdown('</div>', unsafe_allow_html=True)

            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                termination_rate = metrics.get('termination_rate', 0) * 100
                st.markdown('<div class="metric-label">Termination Rate</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{termination_rate:.1f}%</div>', unsafe_allow_html=True)
                st.caption(f"{metrics.get('terminated_members', 0):,} terminated")
                st.markdown('</div>', unsafe_allow_html=True)

            # Premium charts row 1
            col1, col2 = st.columns(2)

            with col1:
                if 'product_analysis' in metrics and not metrics['product_analysis'].empty:
                    fig = create_premium_product_chart(metrics['product_analysis'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'payer_analysis' in metrics and not metrics['payer_analysis'].empty:
                    fig = create_premium_payer_chart(metrics['payer_analysis'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            # Premium charts row 2
            col3, col4 = st.columns(2)

            with col3:
                if 'monthly_trends' in metrics and not metrics['monthly_trends'].empty:
                    fig = create_premium_trend_chart(metrics['monthly_trends'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            with col4:
                if 'age_distribution' in metrics and not metrics['age_distribution'].empty:
                    fig = create_premium_age_chart(metrics['age_distribution'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

    # Premium Details Tab
    if "📊 Premium Details" in tab_names:
        premium_details_index = tab_names.index("📊 Premium Details")
        with tabs[premium_details_index]:
            st.markdown('<div class="main-header">Premium Details</div>', unsafe_allow_html=True)

            # Product analysis table
            if 'product_analysis' in st.session_state.premium_metrics:
                product_data = st.session_state.premium_metrics['product_analysis'].copy()
                product_data['avg_premium'] = product_data['PREMIUM_AMOUNT'] / product_data['MEMBER_NO']
                product_data = product_data.rename(columns={
                    'PRODUCT_NAME': 'Product',
                    'PREMIUM_AMOUNT': 'Total Premium',
                    'MEMBER_NO': 'Members',
                    'avg_premium': 'Avg Premium'
                })

                st.markdown('<div class="sub-header">Product Performance</div>', unsafe_allow_html=True)
                st.dataframe(product_data, use_container_width=True)

            # Payer analysis table
            if 'payer_analysis' in st.session_state.premium_metrics:
                payer_data = st.session_state.premium_metrics['payer_analysis'].copy()
                payer_data['avg_premium'] = payer_data['PREMIUM_AMOUNT'] / payer_data['MEMBER_NO']
                payer_data = payer_data.rename(columns={
                    'PAYER_NAME': 'Payer',
                    'PREMIUM_AMOUNT': 'Total Premium',
                    'MEMBER_NO': 'Members',
                    'avg_premium': 'Avg Premium'
                })

                st.markdown('<div class="sub-header">Payer Analysis</div>', unsafe_allow_html=True)
                st.dataframe(payer_data, use_container_width=True)

            # Raw premium data
            st.markdown('<div class="sub-header">Premium Data</div>', unsafe_allow_html=True)
            st.dataframe(st.session_state.premium_data, use_container_width=True, height=400)

    # Combined View Tab
    if "🔗 Combined View" in tab_names:
        combined_index = tab_names.index("🔗 Combined View")
        with tabs[combined_index]:
            st.markdown('<div class="main-header">Combined Analysis</div>', unsafe_allow_html=True)

            # Calculate key combined metrics
            total_claims = st.session_state.claims_metrics.get('total_claimed', 0)
            total_paid = st.session_state.claims_metrics.get('total_paid', 0)
            total_premium = st.session_state.premium_metrics.get('total_premium', 0)
            ibnr_value = st.session_state.ibnr_estimate or 0

            # Loss ratio analysis
            col1, col2, col3 = st.columns(3)

            with col1:
                if total_premium > 0:
                    loss_ratio = (total_claims / total_premium) * 100
                    paid_loss_ratio = (total_paid / total_premium) * 100

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=loss_ratio,
                        title={'text': "Loss Ratio (%)"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 200]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 80], 'color': "lightgreen"},
                                {'range': [80, 120], 'color': "yellow"},
                                {'range': [120, 200], 'color': "red"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Member overlap
                if 'MEMBER_NO' in st.session_state.claims_data.columns and 'MEMBER_NO' in st.session_state.premium_data.columns:
                    claims_members = set(st.session_state.claims_data['MEMBER_NO'].unique())
                    premium_members = set(st.session_state.premium_data['MEMBER_NO'].unique())
                    common_members = claims_members.intersection(premium_members)

                    labels = ['Claims Only', 'Both', 'Premium Only']
                    values = [
                        len(claims_members - premium_members),
                        len(common_members),
                        len(premium_members - claims_members)
                    ]

                    fig = px.pie(
                        values=values,
                        names=labels,
                        title="Member Overlap",
                        color_discrete_sequence=['#ff9999', '#66b3ff', '#99ff99']
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

            with col3:
                # Financial summary with IBNR
                st.markdown("### Financial Summary")
                summary_data = {
                    'Metric': ['Total Premium', 'Total Claims', 'Total Paid', 'Estimated IBNR', 'Estimated Ultimate',
                               'Net Position'],
                    'Amount': [
                        total_premium,
                        total_claims,
                        total_paid,
                        ibnr_value,
                        total_claims + ibnr_value,
                        total_premium - (total_paid + ibnr_value)
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df['Amount'] = summary_df['Amount'].apply(lambda x: f"${x:,.2f}")
                st.table(summary_df)

            # Combined data table
            st.markdown('<div class="sub-header">Member-Level Combined View</div>', unsafe_allow_html=True)

            # Create combined summary
            if 'MEMBER_NO' in st.session_state.premium_data.columns:
                # Get premium by member
                premium_by_member = st.session_state.premium_data.groupby('MEMBER_NO').agg({
                    'PREMIUM_AMOUNT': 'sum',
                    'PRODUCT_NAME': 'first',
                    'PAYER_NAME': 'first'
                }).reset_index()

                # Get claims by member
                if 'MEMBER_NO' in st.session_state.claims_data.columns:
                    claims_by_member = st.session_state.claims_data.groupby('MEMBER_NO').agg({
                        'AMOUNT_CLAIMED': 'sum',
                        'TOTAL_PAID': 'sum'
                    }).reset_index()

                    # Merge data
                    combined = pd.merge(premium_by_member, claims_by_member,
                                        on='MEMBER_NO', how='left').fillna(0)

                    # Calculate metrics
                    combined['Loss_Ratio'] = (combined['TOTAL_PAID'] / combined['PREMIUM_AMOUNT']) * 100
                    combined['Loss_Ratio'] = combined['Loss_Ratio'].replace([np.inf, -np.inf], 0)

                    st.dataframe(combined, use_container_width=True, height=400)

    # Data Export Tab
    export_index = tab_names.index("📁 Data Export")
    with tabs[export_index]:
        st.markdown('<div class="main-header">Data Export</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="sub-header">Export Claims Data</div>', unsafe_allow_html=True)
            if st.session_state.claims_data is not None:
                csv = st.session_state.claims_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Claims as CSV",
                    data=csv,
                    file_name=f"claims_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col2:
            st.markdown('<div class="sub-header">Export Premium Data</div>', unsafe_allow_html=True)
            if st.session_state.premium_data is not None:
                csv = st.session_state.premium_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Premium as CSV",
                    data=csv,
                    file_name=f"premium_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        # Combined report
        if st.session_state.claims_data is not None and st.session_state.premium_data is not None:
            st.markdown('<div class="sub-header">Combined Report</div>', unsafe_allow_html=True)

            # Create summary report
            ibnr_value = st.session_state.ibnr_estimate or 0
            report_data = {
                'Metric': [
                    'Total Premium Collected',
                    'Total Claims Submitted',
                    'Total Claims Paid',
                    'Estimated IBNR',
                    'Estimated Ultimate Claims',
                    'Loss Ratio (Reported)',
                    'Loss Ratio (Ultimate)',
                    'Active Members',
                    'Average Premium',
                    'Average Claim'
                ],
                'Value': [
                    f"${total_premium:,.2f}",
                    f"${total_claims:,.2f}",
                    f"${total_paid:,.2f}",
                    f"${ibnr_value:,.2f}",
                    f"${total_claims + ibnr_value:,.2f}",
                    f"{(total_paid / total_premium * 100):.1f}%" if total_premium > 0 else "N/A",
                    f"{((total_paid + ibnr_value) / total_premium * 100):.1f}%" if total_premium > 0 else "N/A",
                    f"{st.session_state.premium_metrics.get('active_members', 0):,}",
                    f"${st.session_state.premium_metrics.get('avg_premium', 0):,.2f}",
                    f"${st.session_state.claims_metrics.get('avg_claim_amount', 0):,.2f}"
                ]
            }

            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("📊 **Actuarial Dashboard v2.0**")
with col2:
    st.caption("📍 Operations Department")
with col3:
    st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")streamlit run com.py