import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="IBNR Calculator - Actuarial Tool",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 IBNR Calculator - Actuarial Reserve Estimation")
st.markdown("""
Upload your claims data to calculate IBNR reserves using multiple actuarial methods:
- **Chain Ladder Method**
- **Bornhuetter-Ferguson Method**
- **Expected Claims Ratio Method**
- **Payment Pattern Method**
""")

# Initialize session state for data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ibnr_results' not in st.session_state:
    st.session_state.ibnr_results = {}
if 'triangles' not in st.session_state:
    st.session_state.triangles = {}


def create_development_triangle(df, accident_date_col, development_date_col, value_col):
    """
    Create a development triangle from claims data
    """
    df = df.copy()

    # Convert date columns to datetime
    df[accident_date_col] = pd.to_datetime(df[accident_date_col], errors='coerce')
    df[development_date_col] = pd.to_datetime(df[development_date_col], errors='coerce')

    # Remove rows with invalid dates
    df = df.dropna(subset=[accident_date_col, development_date_col, value_col])

    # Calculate accident period (year-month)
    df['AccidentPeriod'] = df[accident_date_col].dt.to_period('M')

    # Calculate development lag in months
    df['DevelopmentLag'] = ((df[development_date_col] - df[accident_date_col])
                            .dt.days // 30).clip(lower=0)

    # Create triangle
    pivot = df.pivot_table(
        values=value_col,
        index='AccidentPeriod',
        columns='DevelopmentLag',
        aggfunc='sum',
        fill_value=0
    )

    # Sort periods
    pivot = pivot.sort_index()

    # Convert to cumulative triangle
    cum_triangle = pivot.cumsum(axis=1)

    return cum_triangle, df


def chain_ladder_method(triangle):
    """
    Implement Chain Ladder method
    """
    # Calculate age-to-age factors
    n_periods = triangle.shape[1]
    factors = []

    for i in range(n_periods - 1):
        numerator = triangle.iloc[:-(i + 1), i + 1].sum()
        denominator = triangle.iloc[:-(i + 1), i].sum()
        if denominator != 0:
            factors.append(numerator / denominator)
        else:
            factors.append(1.0)

    # Calculate tail factor (assume no further development after last period)
    factors.append(1.0)

    # Project ultimate claims
    ultimates = []
    projected_triangle = triangle.copy()

    for i in range(triangle.shape[0]):
        current_period = triangle.iloc[i]
        last_known = current_period.last_valid_index()

        if last_known is not None:
            last_value = current_period[last_known]
            # Project future development
            future_factors = np.prod(factors[last_known:])
            ultimate = last_value * future_factors
        else:
            ultimate = 0

        ultimates.append(ultimate)

        # Fill projected values in triangle
        for j in range(last_known + 1 if last_known is not None else 0, n_periods):
            if j == 0:
                projected_triangle.iat[i, j] = triangle.iat[i, j] if j < triangle.shape[1] else 0
            else:
                previous_val = projected_triangle.iat[i, j - 1]
                projected_triangle.iat[i, j] = previous_val * factors[j - 1]

    ultimates = pd.Series(ultimates, index=triangle.index)
    ibnr = ultimates - triangle.iloc[:, -1].fillna(0)

    return {
        'ultimates': ultimates,
        'ibnr': ibnr.sum(),
        'factors': factors,
        'projected_triangle': projected_triangle
    }


def bornhuetter_ferguson_method(triangle, expected_claims_ratio, earned_premium):
    """
    Implement Bornhuetter-Ferguson method
    """
    # Chain Ladder projection
    cl_result = chain_ladder_method(triangle)
    cl_ultimates = cl_result['ultimates']

    # Expected claims based on premium
    expected_claims = earned_premium * expected_claims_ratio

    # Calculate development pattern from Chain Ladder
    cl_paid = triangle.iloc[:, -1].fillna(0)
    cl_development = cl_paid / cl_ultimates.replace(0, 1)

    # BF Ultimate = Paid + (1 - % developed) * Expected Claims
    bf_ultimates = cl_paid + (1 - cl_development) * expected_claims

    bf_ibnr = bf_ultimates - cl_paid

    return {
        'ultimates': bf_ultimates,
        'ibnr': bf_ibnr.sum(),
        'expected_claims': expected_claims,
        'development_pattern': cl_development
    }


def expected_claims_ratio_method(earned_premium, expected_claims_ratio, paid_claims):
    """
    Implement Expected Claims Ratio method
    """
    expected_claims = earned_premium * expected_claims_ratio
    ibnr = expected_claims - paid_claims

    return {
        'ibnr': max(ibnr, 0),
        'expected_claims': expected_claims,
        'paid_claims': paid_claims
    }


def payment_pattern_method(df, accident_date_col, payment_date_col, amount_col, forecast_periods=12):
    """
    Implement Payment Pattern method
    """
    df = df.copy()

    # Convert dates
    df[accident_date_col] = pd.to_datetime(df[accident_date_col], errors='coerce')
    df[payment_date_col] = pd.to_datetime(df[payment_date_col], errors='coerce')

    # Remove rows with invalid dates or amounts
    df = df.dropna(subset=[accident_date_col, payment_date_col, amount_col])

    # Calculate payment pattern
    df['AccidentMonth'] = df[accident_date_col].dt.to_period('M')
    df['PaymentMonth'] = df[payment_date_col].dt.to_period('M')
    df['DevelopmentLag'] = (df['PaymentMonth'].dt.to_timestamp() -
                            df['AccidentMonth'].dt.to_timestamp()).dt.days // 30

    # Group by development lag
    pattern = df.groupby('DevelopmentLag')[amount_col].sum()
    pattern_pct = pattern / pattern.sum()

    # Recent accident months (last 6 months)
    recent_months = df['AccidentMonth'].unique()[-6:]
    recent_claims = df[df['AccidentMonth'].isin(recent_months)]

    # Estimate future payments
    if not recent_claims.empty:
        recent_by_month = recent_claims.groupby('AccidentMonth')[amount_col].sum()
        avg_recent_claims = recent_by_month.mean()
    else:
        avg_recent_claims = 0

    # Forecast future payments
    future_payments = 0
    if not pattern_pct.empty:
        max_lag = int(pattern_pct.index.max())
        for lag in range(max_lag + 1, max_lag + forecast_periods + 1):
            if lag <= max_lag:
                future_payments += avg_recent_claims * pattern_pct.get(lag, 0)
            else:
                # Extrapolate using last available pattern
                future_payments += avg_recent_claims * pattern_pct.iloc[-1] * (0.9 ** (lag - max_lag))

    return {
        'ibnr': future_payments,
        'payment_pattern': pattern_pct,
        'avg_recent_claims': avg_recent_claims
    }


def calculate_all_methods(df, config):
    """
    Calculate IBNR using all methods
    """
    results = {}

    try:
        # Create development triangle
        triangle, processed_df = create_development_triangle(
            df,
            config['accident_date_col'],
            config['development_date_col'],
            config['amount_col']
        )
        st.session_state.triangles = {'triangle': triangle, 'processed_df': processed_df}

        # Calculate metrics for methods
        total_paid = df[config['amount_col']].sum()
        earned_premium = config.get('earned_premium', total_paid * 1.5)  # Default if not provided
        expected_ratio = config.get('expected_claims_ratio', 0.75)  # Default 75% loss ratio

        # 1. Chain Ladder
        cl_result = chain_ladder_method(triangle)
        results['Chain Ladder'] = {
            'IBNR': cl_result['ibnr'],
            'Ultimate Claims': cl_result['ultimates'].sum(),
            'Paid Claims': triangle.iloc[:, -1].sum(),
            'Development Factors': cl_result['factors']
        }

        # 2. Bornhuetter-Ferguson
        bf_result = bornhuetter_ferguson_method(triangle, expected_ratio, earned_premium)
        results['Bornhuetter-Ferguson'] = {
            'IBNR': bf_result['ibnr'],
            'Ultimate Claims': bf_result['ultimates'].sum(),
            'Paid Claims': triangle.iloc[:, -1].sum(),
            'Expected Claims': bf_result['expected_claims']
        }

        # 3. Expected Claims Ratio
        ecr_result = expected_claims_ratio_method(earned_premium, expected_ratio, total_paid)
        results['Expected Claims Ratio'] = {
            'IBNR': ecr_result['ibnr'],
            'Expected Claims': ecr_result['expected_claims'],
            'Paid Claims': ecr_result['paid_claims']
        }

        # 4. Payment Pattern
        pp_result = payment_pattern_method(
            df,
            config['accident_date_col'],
            config['development_date_col'],
            config['amount_col']
        )
        results['Payment Pattern'] = {
            'IBNR': pp_result['ibnr'],
            'Average Recent Claims': pp_result['avg_recent_claims'],
            'Payment Pattern': pp_result['payment_pattern']
        }

        st.session_state.ibnr_results = results

    except Exception as e:
        st.error(f"Error in calculations: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

    return results


def visualize_results(results, triangle=None):
    """
    Create visualizations of IBNR results
    """
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["IBNR Comparison", "Development Triangle", "Payment Pattern", "Method Details"])

    with tab1:
        # IBNR comparison across methods
        methods = list(results.keys())
        ibnr_values = [results[m]['IBNR'] for m in methods]

        fig = go.Figure(data=[
            go.Bar(x=methods, y=ibnr_values,
                   marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ])
        fig.update_layout(
            title="IBNR Reserve by Method",
            xaxis_title="Method",
            yaxis_title="IBNR Reserve",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        summary_data = []
        for method, data in results.items():
            summary_data.append({
                'Method': method,
                'IBNR Reserve': f"${data['IBNR']:,.2f}",
                'Paid Claims': f"${data.get('Paid Claims', 0):,.2f}",
                'Ultimate Claims': f"${data.get('Ultimate Claims', data.get('Expected Claims', 0)):,.2f}"
            })
        st.dataframe(pd.DataFrame(summary_data))

    with tab2:
        if triangle is not None and not triangle.empty:
            # Display development triangle
            st.subheader("Development Triangle")

            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(triangle.fillna(0), annot=True, fmt=",.0f",
                        cmap="YlOrRd", ax=ax, cbar_kws={'label': 'Cumulative Amount'})
            ax.set_title("Development Triangle - Cumulative Claims")
            ax.set_xlabel("Development Period (Months)")
            ax.set_ylabel("Accident Period")
            st.pyplot(fig)

            # Display triangle as table
            with st.expander("View Triangle Data"):
                st.dataframe(triangle.style.format("{:,.0f}"))

    with tab3:
        if 'Payment Pattern' in results:
            st.subheader("Payment Pattern Analysis")

            # Plot payment pattern
            pattern = results['Payment Pattern']['Payment Pattern']
            if not pattern.empty:
                fig = go.Figure(data=[
                    go.Bar(x=pattern.index.astype(str), y=pattern.values,
                           marker_color='lightblue')
                ])
                fig.update_layout(
                    title="Payment Pattern by Development Lag",
                    xaxis_title="Development Lag (Months)",
                    yaxis_title="Percentage of Claims",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Display detailed results for each method
        for method, data in results.items():
            with st.expander(f"{method} Details"):
                if method == "Chain Ladder":
                    st.write("**Development Factors:**")
                    for i, factor in enumerate(data['Development Factors']):
                        st.write(f"  Age {i} to {i + 1}: {factor:.4f}")

                if method == "Bornhuetter-Ferguson":
                    st.write(f"**Expected Claims Ratio:** {config.get('expected_claims_ratio', 0.75):.2%}")
                    st.write(f"**Earned Premium:** ${config.get('earned_premium', 0):,.2f}")

                # Display all metrics
                for key, value in data.items():
                    if key != 'Development Factors' and key != 'Payment Pattern':
                        if isinstance(value, (int, float)):
                            st.write(f"**{key}:** ${value:,.2f}")
                        else:
                            st.write(f"**{key}:** {value}")


def get_column_mapping(df):
    """
    Get column mapping for common column names in the provided file format
    """
    mapping = {}

    # Date columns - look for common patterns
    date_keywords = ['date', 'Date', 'DATE']
    amount_keywords = ['paid', 'Paid', 'PAID', 'claim', 'Claim', 'CLAIM', 'amount', 'Amount', 'AMOUNT']

    for col in df.columns:
        col_lower = str(col).lower()

        # Check for date columns
        if any(keyword in col_lower for keyword in ['service', 'accident', 'occurrence']):
            if 'date' in col_lower:
                mapping['accident_date'] = col
        elif any(keyword in col_lower for keyword in ['received', 'report', 'assessment', 'assess']):
            if 'date' in col_lower:
                mapping['development_date'] = col
        elif 'date' in col_lower and 'accident_date' not in mapping:
            mapping['accident_date'] = mapping.get('accident_date', col)

        # Check for amount columns
        if 'total paid' in col_lower or col == 'TOTAL PAID':
            mapping['amount'] = col
        elif 'amount claimed' in col_lower or col == 'AMOUNT CLAIMED':
            mapping['amount'] = mapping.get('amount', col)
        elif any(keyword in col_lower for keyword in amount_keywords):
            mapping['amount'] = mapping.get('amount', col)

    return mapping


# Main app layout
st.sidebar.header("📤 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx', 'xls'])

# Global config
config = {}

if uploaded_file:
    try:
        # Read the file
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names

        # Let user select sheet if multiple sheets exist
        if len(sheet_names) > 1:
            selected_sheet = st.sidebar.selectbox("Select Sheet", sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        else:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])

        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()
        st.session_state.df = df

        st.sidebar.success(f"File loaded successfully! Shape: {df.shape}")

        # Show data preview
        with st.expander("📋 Preview Data", expanded=True):
            st.dataframe(df.head())

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Summary:**")
                st.write(f"- Rows: {df.shape[0]}")
                st.write(f"- Columns: {df.shape[1]}")

                # Check for date columns
                date_cols = []
                for col in df.columns:
                    try:
                        sample = pd.to_datetime(df[col].iloc[0], errors='raise')
                        date_cols.append(col)
                    except:
                        continue

                if date_cols:
                    min_date = None
                    max_date = None
                    for col in date_cols:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        col_min = df[col].min()
                        col_max = df[col].max()
                        if not pd.isna(col_min):
                            if min_date is None or col_min < min_date:
                                min_date = col_min
                        if not pd.isna(col_max):
                            if max_date is None or col_max > max_date:
                                max_date = col_max

                    if min_date and max_date:
                        st.write(f"- Date Range: {min_date.date()} to {max_date.date()}")

            with col2:
                st.write("**Column Types:**")
                for col in df.columns[:10]:  # Show first 10 columns
                    st.write(f"- {col}: {df[col].dtype}")
                if len(df.columns) > 10:
                    st.write(f"- ... and {len(df.columns) - 10} more columns")

        # Auto-detect column mapping
        column_mapping = get_column_mapping(df)

        # Configuration section
        st.sidebar.header("⚙️ Configuration")

        # Auto-detect date and amount columns
        date_cols = []
        for col in df.columns:
            try:
                # Try to convert first non-null value
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if sample:
                    pd.to_datetime(sample, errors='raise')
                    date_cols.append(col)
            except:
                continue

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # If no numeric cols found, try to convert object cols that look numeric
        if not numeric_cols:
            for col in df.columns:
                try:
                    # Try to convert to numeric
                    converted = pd.to_numeric(df[col], errors='coerce')
                    if converted.notna().any():
                        df[col] = converted
                        numeric_cols.append(col)
                except:
                    pass

        # Configuration inputs with smart defaults
        with st.sidebar.expander("Method Parameters", expanded=True):

            # Accident Date Column
            if date_cols:
                default_accident_idx = 0
                if 'accident_date' in column_mapping:
                    try:
                        default_accident_idx = date_cols.index(column_mapping['accident_date'])
                    except:
                        pass

                config['accident_date_col'] = st.selectbox(
                    "Accident Date Column",
                    date_cols,
                    index=default_accident_idx,
                    help="Typically 'SERVICE DATE' in your file"
                )
            else:
                st.warning("No date columns detected. Please ensure your data has date columns.")
                config['accident_date_col'] = None

            # Development Date Column
            if date_cols:
                default_development_idx = min(1, len(date_cols) - 1) if len(date_cols) > 1 else 0
                if 'development_date' in column_mapping:
                    try:
                        default_development_idx = date_cols.index(column_mapping['development_date'])
                    except:
                        pass

                config['development_date_col'] = st.selectbox(
                    "Development/Reporting Date Column",
                    date_cols,
                    index=default_development_idx,
                    help="Typically 'DATE RECEIVED' or 'ASSESS DATE' in your file"
                )
            else:
                config['development_date_col'] = None

            # Amount Column
            if numeric_cols:
                default_amount_idx = 0
                if 'amount' in column_mapping:
                    try:
                        default_amount_idx = numeric_cols.index(column_mapping['amount'])
                    except:
                        pass

                config['amount_col'] = st.selectbox(
                    "Claim Amount Column",
                    numeric_cols,
                    index=default_amount_idx,
                    help="Typically 'TOTAL PAID' or 'AMOUNT CLAIMED' in your file"
                )
            else:
                st.warning("No numeric columns detected for claim amounts.")
                config['amount_col'] = None

            # Additional parameters
            config['earned_premium'] = st.number_input(
                "Earned Premium (for BF & ECR methods)",
                min_value=0.0,
                value=float(df[numeric_cols[0]].sum() * 1.2) if numeric_cols else 1000000.0,
                step=10000.0,
                help="Total premium earned for the period"
            )

            config['expected_claims_ratio'] = st.slider(
                "Expected Claims Ratio (Loss Ratio)",
                min_value=0.0,
                max_value=1.5,
                value=0.75,
                step=0.05,
                help="Expected claims as percentage of earned premium"
            )

        # Show detected mapping
        with st.sidebar.expander("🔍 Auto-detected Columns"):
            if column_mapping:
                st.write("Auto-detected column mapping:")
                for key, value in column_mapping.items():
                    st.write(f"- **{key.replace('_', ' ').title()}**: `{value}`")
            else:
                st.write("No auto-detection available")

        # Calculate button
        if st.sidebar.button("🚀 Calculate IBNR", type="primary", use_container_width=True):
            if not all([config['accident_date_col'], config['development_date_col'], config['amount_col']]):
                st.error("Please select all required columns before calculating.")
            else:
                with st.spinner("Calculating IBNR using multiple methods..."):
                    results = calculate_all_methods(df, config)

                    if results:
                        # Display results
                        st.header("📈 IBNR Calculation Results")

                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)

                        avg_ibnr = np.mean([results[m]['IBNR'] for m in results])
                        min_ibnr = min([results[m]['IBNR'] for m in results])
                        max_ibnr = max([results[m]['IBNR'] for m in results])

                        with col1:
                            st.metric("Average IBNR", f"${avg_ibnr:,.2f}")
                        with col2:
                            st.metric("Minimum IBNR", f"${min_ibnr:,.2f}")
                        with col3:
                            st.metric("Maximum IBNR", f"${max_ibnr:,.2f}")
                        with col4:
                            st.metric("Range", f"${max_ibnr - min_ibnr:,.2f}")

                        # Visualizations
                        visualize_results(results, st.session_state.triangles.get('triangle', None))

                        # Export results
                        st.sidebar.header("📥 Export Results")

                        # Create downloadable report
                        report_data = {}
                        for method, data in results.items():
                            for key, value in data.items():
                                if key not in ['Development Factors', 'Payment Pattern']:
                                    report_data[f"{method}_{key}"] = value

                        report_df = pd.DataFrame([report_data])

                        # Convert to Excel
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            report_df.to_excel(writer, sheet_name='IBNR_Summary', index=False)
                            if 'triangle' in st.session_state.triangles and not st.session_state.triangles[
                                'triangle'].empty:
                                st.session_state.triangles['triangle'].to_excel(
                                    writer, sheet_name='Development_Triangle'
                                )

                        st.sidebar.download_button(
                            label="📥 Download Report",
                            data=output.getvalue(),
                            file_name=f"ibnr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback

        st.error(traceback.format_exc())

else:
    # Show instructions with specific guidance for the file format
    st.info("👈 Please upload an Excel file using the sidebar")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 For Your File Format")
        st.markdown("""
        Your Excel file should have these exact column names:

        **Required Columns:**
        - `SERVICE DATE` - Date of service/accident
        - `DATE RECEIVED` - Date claim was received
        - `TOTAL PAID` or `AMOUNT CLAIMED` - Claim amount

        **Additional Columns (optional):**
        - `ASSESS DATE` - Alternative for development date
        - `CLAIM NO` - Claim identifier
        - `MEMBER NO` - Member identifier
        - Any other relevant fields
        """)

        # Show exact column names expected
        with st.expander("🔤 Exact Column Names in Your File"):
            st.code("""
            MEMBER NO
            Gender
            INO
            PRACTICE NO
            DIS
            INV REF
            SERVICE DATE
            ASSESS DATE
            DATE RECEIVED
            CLM CODE
            CODE DESCRIPTION
            ICD10 CODE
            ICD10 DESCRIPTION
            UNITS
            SCRIPT CODE
            AMOUNT CLAIMED
            PAID FROM RISK AMT
            PAID FROM THRESHHOLD
            PAID FROM SAVINGS
            RECOVERY AMOUNT (FROM PAYER)
            TOTAL PAID
            TARIFF
            CO-PAY
            NET ASS FIN REQ AMT
            MEMBER LIABILITY (CLAIMED - PAID FROM RISK - PAID FROM SAVINGS - RECOVERY AMOUNT)
            AMOUNT REJECTED (CLAIMED - PAID FROM RISK - PAID FROM SAVINGS - COPAYMENT - RECOVERY AMOUNT)
            PAY TO
            REJ
            CLAIM BENEFIT INDICATOR
            CODE CATEGORY
            CODE CATEGORY B
            CLAIM MANUAL REASONS
            REV
            REASSESSED CLAIM LINE
            AUTH NO
            AUTH TYPE
            DIAG CLIN GROUP
            CLIN GROUP
            DL
            CLAIM NO
            CLAIM LINE NO
            DUPLICATE CLAIM
            DUPLICATE CLAIM LINE
            BASE BENEFIT CODE
            BASE BENEFIT DESCRIPTION
            PROVIDER NAME
            PAYER
            PAYER NAME
            OPTION NAME
            EXTERNAL PAY DATE
            EXTERNAL PAY INSTRUCTION DATE
            PAPER/ EDI
            ASSESSOR NAME
            ADDONS
            BIRTHDATE
            CURRENT AGE
            """)

    with col2:
        st.subheader("🎯 Recommended Configuration")
        st.markdown("""
        For optimal results with your file:

        **Column Selection:**
        1. **Accident Date:** `SERVICE DATE`
        2. **Development Date:** `DATE RECEIVED`
        3. **Claim Amount:** `TOTAL PAID` (preferred)

        **Method Parameters:**
        - **Earned Premium:** Total premium for the period
        - **Expected Claims Ratio:** Typically 70-80% for medical insurance

        **The tool will auto-detect these columns for you!**
        """)

        st.subheader("🔄 Processing Steps")
        st.markdown("""
        1. **Upload** your Excel file
        2. **Auto-detection** finds key columns
        3. **Verify** the suggested column mapping
        4. **Adjust parameters** if needed
        5. **Calculate** to see all IBNR methods
        6. **Export** comprehensive reports
        """)

# Add specific instructions for the user's file
st.markdown("---")
st.markdown("### 💡 Tips for Your Specific File")
st.markdown("""
1. **Column Names Matter:** The tool looks for exact column names like `SERVICE DATE`, `DATE RECEIVED`, and `TOTAL PAID`
2. **Date Format:** Ensure dates are in a recognizable format (YYYY-MM-DD)
3. **Data Quality:** Check for missing values in key columns
4. **Multiple Sheets:** If your file has multiple sheets, select the correct one from the dropdown
5. **Amount Column:** Use `TOTAL PAID` for paid amounts or `AMOUNT CLAIMED` for incurred amounts
""")

# Footer
st.markdown("---")
st.markdown("""
**IBNR Calculator** • Built for Actuarial Reserve Analysis • 
[Learn more about IBNR methods](https://www.soa.org/resources/research-reports/2024/reserving-methods-guide/)
""")