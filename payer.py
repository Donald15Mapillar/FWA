import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# -------------------------------------------------------------------
# Page configuration
st.set_page_config(
    page_title="Payer Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Payer Performance Report – Medical Aid Fund")
st.markdown("Upload your claims file (Excel format) to generate an interactive performance dashboard.")

# -------------------------------------------------------------------
# File upload
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Load data
    df = pd.read_excel(uploaded_file)

    # Clean column names (strip spaces)
    df.columns = df.columns.str.strip()

    st.sidebar.success("File loaded successfully!")
    st.sidebar.write(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

    # -------------------------------------------------------------------
    # Data preprocessing
    # Convert date columns (fix typo: to_datetime, not to_datime)
    date_cols = ['SERVICE DATE', 'ASSESS DATE', 'DATE RECEIVED', 'EXTERNAL PAY DATE', 'EXTERNAL PAY INSTRUCTION DATE', 'BIRTHDATE']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Create age groups for demographics
    if 'CURRENT AGE' in df.columns:
        df['AGE_GROUP'] = pd.cut(df['CURRENT AGE'], bins=[0, 18, 35, 50, 65, 120], labels=['0-18', '19-35', '36-50', '51-65', '65+'])

    # Derive a service category from 'BASE BENEFIT DESCRIPTION' or 'CODE CATEGORY'
    def categorize_service(desc):
        if pd.isna(desc):
            return 'Other'
        desc = str(desc).upper()
        if 'HOSPITAL' in desc or 'INPATIENT' in desc:
            return 'Inpatient'
        elif 'CONSULTATION' in desc or 'OUTPATIENT' in desc:
            return 'Outpatient'
        elif 'MEDICATION' in desc or 'CHRONIC' in desc or 'ACUTE' in desc:
            return 'Medication'
        elif 'PATHOLOGY' in desc or 'LAB' in desc:
            return 'Pathology'
        elif 'RADIOLOGY' in desc or 'X-RAY' in desc:
            return 'Radiology'
        elif 'DENTAL' in desc:
            return 'Dental'
        elif 'ONCOLOGY' in desc:
            return 'Oncology'
        elif 'PHYSIO' in desc or 'REHAB' in desc:
            return 'Rehab'
        else:
            return 'Other'

    if 'BASE BENEFIT DESCRIPTION' in df.columns:
        df['SERVICE_CATEGORY'] = df['BASE BENEFIT DESCRIPTION'].apply(categorize_service)
    else:
        df['SERVICE_CATEGORY'] = 'Unknown'

    # Processing time (days from received to assess)
    if 'DATE RECEIVED' in df.columns and 'ASSESS DATE' in df.columns:
        df['PROCESSING_DAYS'] = (df['ASSESS DATE'] - df['DATE RECEIVED']).dt.days

    # Member liability = AMOUNT CLAIMED - TOTAL PAID? We'll compute rejection amount
    if 'AMOUNT CLAIMED' in df.columns and 'TOTAL PAID' in df.columns:
        df['REJECTED_AMOUNT'] = df['AMOUNT CLAIMED'] - df['TOTAL PAID']

    # -------------------------------------------------------------------
    # Compute KPIs
    total_members = df['MEMBER NO'].nunique() if 'MEMBER NO' in df.columns else 0
    avg_age = df['CURRENT AGE'].mean() if 'CURRENT AGE' in df.columns else None
    total_claimed = df['AMOUNT CLAIMED'].sum() if 'AMOUNT CLAIMED' in df.columns else 0
    total_paid = df['TOTAL PAID'].sum() if 'TOTAL PAID' in df.columns else 0
    total_rejected = total_claimed - total_paid
    rejection_rate = (total_rejected / total_claimed * 100) if total_claimed > 0 else 0

    # -------------------------------------------------------------------
    # Dashboard layout with tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Summary", "Membership & Demographics", "Financial Performance",
        "Claims & Utilization", "Provider Network", "Member Experience"
    ])

    # ---------- Executive Summary ----------
    with tab1:
        st.header("Executive Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Members", f"{total_members:,}")
        if avg_age:
            col2.metric("Average Age", f"{avg_age:.1f}")
        col3.metric("Total Claimed", f"${total_claimed:,.0f}")
        col4.metric("Total Paid", f"${total_paid:,.0f}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rejected Amount", f"${total_rejected:,.0f}")
        col2.metric("Rejection Rate", f"{rejection_rate:.1f}%")
        # Dependency ratio (members under 18 vs adults)
        if 'CURRENT AGE' in df.columns:
            under_18 = df[df['CURRENT AGE'] < 18]['MEMBER NO'].nunique()
            adults = total_members - under_18
            dep_ratio = under_18 / adults if adults > 0 else 0
            col3.metric("Dependency Ratio", f"{dep_ratio:.2f}")
        # Average processing time
        if 'PROCESSING_DAYS' in df.columns:
            avg_proc = df['PROCESSING_DAYS'].mean()
            col4.metric("Avg Processing Days", f"{avg_proc:.1f}")

        st.markdown("**Key Highlights:**")
        st.info("• Top cost driver: check 'Claims & Utilization' tab.\n• Provider performance: see 'Provider Network' tab.")

    # ---------- Membership & Demographics ----------
    with tab2:
        st.header("Membership & Demographics")

        if 'CURRENT AGE' in df.columns:
            fig_age = px.histogram(df, x='CURRENT AGE', nbins=30, title='Age Distribution of Members',
                                   labels={'CURRENT AGE': 'Age'})
            st.plotly_chart(fig_age, use_container_width=True)

            # Age group pie
            age_group_counts = df.groupby('AGE_GROUP', observed=False)['MEMBER NO'].nunique().reset_index()
            age_group_counts.columns = ['Age Group', 'Member Count']
            fig_pie = px.pie(age_group_counts, values='Member Count', names='Age Group',
                             title='Members by Age Group')
            st.plotly_chart(fig_pie, use_container_width=True)

        # Geographic distribution if available
        if 'PROVINCE' in df.columns:  # not in sample, but placeholder
            geo = df['PROVINCE'].value_counts().reset_index()
            geo.columns = ['Province', 'Count']
            fig_geo = px.bar(geo, x='Province', y='Count', title='Geographic Distribution')
            st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.info("Geographic data not available in this file.")

    # ---------- Financial Performance ----------
    with tab3:
        st.header("Financial Performance")

        # Monthly aggregation
        if 'SERVICE DATE' in df.columns:
            df['YEAR_MONTH'] = df['SERVICE DATE'].dt.to_period('M').astype(str)
            monthly = df.groupby('YEAR_MONTH').agg({
                'AMOUNT CLAIMED': 'sum',
                'TOTAL PAID': 'sum',
                'REJECTED_AMOUNT': 'sum'
            }).reset_index()

            fig_fin = go.Figure()
            fig_fin.add_trace(go.Bar(x=monthly['YEAR_MONTH'], y=monthly['AMOUNT CLAIMED'], name='Claimed'))
            fig_fin.add_trace(go.Bar(x=monthly['YEAR_MONTH'], y=monthly['TOTAL PAID'], name='Paid'))
            fig_fin.add_trace(go.Scatter(x=monthly['YEAR_MONTH'], y=monthly['REJECTED_AMOUNT'], name='Rejected', mode='lines+markers', yaxis='y2'))
            fig_fin.update_layout(title='Monthly Claimed vs Paid',
                                   barmode='group',
                                   yaxis_title='Amount',
                                   yaxis2=dict(title='Rejected', overlaying='y', side='right'))
            st.plotly_chart(fig_fin, use_container_width=True)

            # Loss ratio if we had premium. We'll show paid/claimed ratio as proxy
            monthly['PAID_RATIO'] = monthly['TOTAL PAID'] / monthly['AMOUNT CLAIMED'] * 100
            fig_ratio = px.line(monthly, x='YEAR_MONTH', y='PAID_RATIO', title='Paid % of Claimed (proxy for loss ratio)',
                                labels={'PAID_RATIO': '% Paid', 'YEAR_MONTH': 'Month'})
            st.plotly_chart(fig_ratio, use_container_width=True)

        # Summary table
        st.subheader("Summary Totals")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Claimed", f"${total_claimed:,.0f}")
        col2.metric("Total Paid", f"${total_paid:,.0f}")
        col3.metric("Total Rejected", f"${total_rejected:,.0f}")

    # ---------- Claims & Utilization ----------
    with tab4:
        st.header("Claims & Utilization")

        # Breakdown by service category
        if 'SERVICE_CATEGORY' in df.columns:
            cat_stats = df.groupby('SERVICE_CATEGORY').agg(
                Frequency=('MEMBER NO', 'count'),
                Total_Cost=('AMOUNT CLAIMED', 'sum'),
                Avg_Cost=('AMOUNT CLAIMED', 'mean')
            ).reset_index().sort_values('Total_Cost', ascending=False)

            fig_cat = px.bar(cat_stats, x='SERVICE_CATEGORY', y='Total_Cost',
                             title='Total Claimed Cost by Service Category',
                             labels={'Total_Cost': 'Amount Claimed'})
            st.plotly_chart(fig_cat, use_container_width=True)

            # Frequency bar
            fig_freq = px.bar(cat_stats, x='SERVICE_CATEGORY', y='Frequency',
                              title='Claim Frequency by Service Category')
            st.plotly_chart(fig_freq, use_container_width=True)

        # Top 5 cost drivers (by ICD10 description or benefit description)
        if 'ICD10 DESCRIPTION' in df.columns:
            top_icd = df.groupby('ICD10 DESCRIPTION')['AMOUNT CLAIMED'].sum().nlargest(5).reset_index()
            fig_top = px.bar(top_icd, x='AMOUNT CLAIMED', y='ICD10 DESCRIPTION', orientation='h',
                             title='Top 5 Cost Drivers by ICD10 Description')
            st.plotly_chart(fig_top, use_container_width=True)
        elif 'BASE BENEFIT DESCRIPTION' in df.columns:
            top_benefit = df.groupby('BASE BENEFIT DESCRIPTION')['AMOUNT CLAIMED'].sum().nlargest(5).reset_index()
            fig_top = px.bar(top_benefit, x='AMOUNT CLAIMED', y='BASE BENEFIT DESCRIPTION', orientation='h',
                             title='Top 5 Cost Drivers by Benefit Description')
            st.plotly_chart(fig_top, use_container_width=True)

        # High-cost cases (> 95th percentile)
        if 'AMOUNT CLAIMED' in df.columns:
            high_cost_thresh = df['AMOUNT CLAIMED'].quantile(0.95)
            high_cost = df[df['AMOUNT CLAIMED'] > high_cost_thresh]
            st.metric("High-Cost Cases (>95th percentile)", f"{len(high_cost):,} claims", delta=f"Threshold: ${high_cost_thresh:,.0f}")

    # ---------- Provider Network ----------
    with tab5:
        st.header("Provider Network Performance")

        if 'PROVIDER NAME' in df.columns:
            # Group by provider
            provider_stats = df.groupby('PROVIDER NAME').agg(
                Claims_Volume=('MEMBER NO', 'count'),
                Total_Paid=('TOTAL PAID', 'sum'),
                Avg_Cost_per_Claim=('TOTAL PAID', 'mean')
            ).reset_index().sort_values('Total_Paid', ascending=False).head(15)

            fig_prov = px.bar(provider_stats, x='Total_Paid', y='PROVIDER NAME', orientation='h',
                              title='Top 15 Providers by Total Paid',
                              labels={'Total_Paid': 'Total Paid Amount'})
            st.plotly_chart(fig_prov, use_container_width=True)

            # Average cost per claim
            fig_avg = px.bar(provider_stats.head(10), x='Avg_Cost_per_Claim', y='PROVIDER NAME', orientation='h',
                             title='Top 10 Providers - Average Cost per Claim')
            st.plotly_chart(fig_avg, use_container_width=True)

        # Network adequacy (if we had provider speciality)
        st.info("Network adequacy and contract compliance metrics require additional data not present in this file.")

    # ---------- Member Experience ----------
    with tab6:
        st.header("Member Experience")

        if 'PROCESSING_DAYS' in df.columns:
            fig_proc = px.histogram(df, x='PROCESSING_DAYS', nbins=50,
                                     title='Distribution of Processing Days (Received to Assess)')
            st.plotly_chart(fig_proc, use_container_width=True)

            # Turnaround time compliance (e.g., within 5 days)
            within_5 = (df['PROCESSING_DAYS'] <= 5).mean() * 100
            st.metric("Claims processed within 5 days", f"{within_5:.1f}%")

        # Complaints not available; could use rejection rate as proxy for friction
        st.metric("Overall Rejection Rate", f"{rejection_rate:.1f}%")

    # -------------------------------------------------------------------
    # Footer / raw data expander
    with st.expander("View raw data preview"):
        st.dataframe(df.head(100))

else:
    st.info("Please upload an Excel file to begin.")