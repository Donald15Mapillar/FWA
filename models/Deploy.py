import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import datetime
from datetime import date
import uuid

warnings.filterwarnings('ignore')

# Configure the page with modern settings
st.set_page_config(
    page_title="FWA Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling with historical analysis additions
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

    .history-item {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .history-item:hover {
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }

    .history-item.active {
        border-left: 4px solid #667eea;
        background: #f8f9ff;
    }

    .comparison-metric {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: #f8f9fa;
        margin: 0.5rem 0;
    }

    .metric-improvement {
        color: #28a745;
        font-weight: 600;
    }

    .metric-decline {
        color: #dc3545;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


class HistoricalAnalysis:
    def __init__(self):
        self.analyses = {}

    def save_analysis(self, analysis_id, analysis_data):
        """Save an analysis to historical data"""
        self.analyses[analysis_id] = {
            'timestamp': datetime.datetime.now(),
            'data': analysis_data,
            'metadata': {
                'total_claims': len(analysis_data['flagged_claims']),
                'flagged_claims': len(
                    analysis_data['flagged_claims'][analysis_data['flagged_claims']['FWA_Flag'] == True]),
                'total_claimed_amount': analysis_data['flagged_claims']['AMOUNT CLAIMED'].sum(),
                'flagged_claimed_amount':
                    analysis_data['flagged_claims'][analysis_data['flagged_claims']['FWA_Flag'] == True][
                        'AMOUNT CLAIMED'].sum(),
                'fraud_count': len(analysis_data['flagged_claims'][
                                       analysis_data['flagged_claims']['Flag_Type'].str.contains('F', na=False)]),
                'waste_count': len(analysis_data['flagged_claims'][
                                       analysis_data['flagged_claims']['Flag_Type'].str.contains('W', na=False)]),
                'abuse_count': len(analysis_data['flagged_claims'][
                                       analysis_data['flagged_claims']['Flag_Type'].str.contains('A', na=False)])
            }
        }

    def get_analysis(self, analysis_id):
        """Retrieve a specific analysis"""
        return self.analyses.get(analysis_id)

    def get_all_analyses(self):
        """Get all historical analyses sorted by timestamp"""
        return sorted(
            [(k, v) for k, v in self.analyses.items()],
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )

    def delete_analysis(self, analysis_id):
        """Delete a specific analysis"""
        if analysis_id in self.analyses:
            del self.analyses[analysis_id]
            return True
        return False


class FWADetector:
    def __init__(self):
        self.claims_data = None
        self.tariff_data = None
        self.flagged_claims = None

    def load_data(self, claims_file, tariff_file):
        """Load claims and tariff data from uploaded files"""
        try:
            # Remove worksheet names - read first sheet by default
            self.claims_data = pd.read_excel(claims_file)
            self.tariff_data = pd.read_excel(tariff_file)
            return True, "Data loaded successfully"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"

    def detect_fwa(self):
        """Main FWA detection logic - only keeping specified checks"""
        if self.claims_data is None or self.tariff_data is None:
            return False, "Please load data first"

        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create a copy for flagged claims
            self.flagged_claims = self.claims_data.copy()

            # Initialize flag columns
            self.flagged_claims['FWA_Flag'] = False
            self.flagged_claims['Flag_Reason'] = ''
            self.flagged_claims['Flag_Type'] = ''  # F, W, or A

            status_text.text("Checking gender appropriateness...")
            self._check_gender_appropriateness()
            progress_bar.progress(15)

            status_text.text("Checking male inappropriate codes...")
            self._check_male_inappropriate_codes()
            progress_bar.progress(30)

            status_text.text("Checking female inappropriate codes...")
            self._check_female_inappropriate_codes()
            progress_bar.progress(45)

            status_text.text("Checking age appropriateness...")
            self._check_age_appropriateness()
            progress_bar.progress(55)

            status_text.text("Checking unit overdosing...")
            self._check_unit_overdosing()
            progress_bar.progress(65)

            status_text.text("Checking duplicate claims...")
            self._check_duplicate_claims()
            progress_bar.progress(75)

            status_text.text("Checking mutually exclusive tariffs...")
            self._check_mutually_exclusive_tariffs()
            progress_bar.progress(90)

            status_text.text("Finalizing analysis...")
            progress_bar.progress(100)

            status_text.text("Analysis complete!")
            time.sleep(0.5)
            status_text.empty()

            return True, "FWA analysis completed successfully"

        except Exception as e:
            return False, f"Error during FWA analysis: {str(e)}"

    def _check_gender_appropriateness(self):
        """Check if service is appropriate for patient's gender based on tariff"""
        for idx, claim in self.flagged_claims.iterrows():
            clm_code = claim['CLM CODE']
            patient_gender = claim['Gender']

            # Find matching tariff
            tariff_match = self.tariff_data[self.tariff_data['CLM CODE'] == clm_code]

            if not tariff_match.empty:
                # Check if AllowedGender column exists
                if 'AllowedGender' in tariff_match.columns:
                    allowed_gender = tariff_match.iloc[0]['AllowedGender']

                    # Check gender appropriateness
                    if allowed_gender == 'B':
                        continue  # Both genders allowed
                    elif allowed_gender == 'F' and patient_gender != 'Female':
                        self._add_flag(idx, "Gender inappropriate for service based on tariff", "F")
                    elif allowed_gender == 'M' and patient_gender != 'Male':
                        self._add_flag(idx, "Gender inappropriate for service based on tariff", "F")

    def _check_male_inappropriate_codes(self):
        """Check if male patients are claiming female-only procedure codes"""
        # List of procedure codes that males cannot claim
        male_inappropriate_codes = [
            19121, 52270, 52285, 53210, 53430, 53660, 53665, 55980, 0, 56420, 56440,
            56441, 56500, 56505, 56510, 56602, 56620, 56625, 56641, 56700, 56740,
            56750, 57000, 57010, 57100, 57105, 57120, 57130, 57210, 57240, 57245,
            57250, 57256, 57260, 57262, 57265, 57267, 57280, 57288, 57290, 57300,
            57320, 57410, 57452, 57500, 57510, 57521, 57525, 57530, 57540, 57550,
            57560, 57600, 57700, 57701, 57720, 58101, 58120, 58130, 58140, 58145,
            58146, 58150, 58155, 58210, 58260, 58265, 58270, 58300, 58301, 58320,
            58340, 58350, 58351, 58400, 58410, 58500, 58520, 58540, 58600, 58700,
            58720, 58741, 58743, 58744, 58830, 58840, 58841, 58900, 58940, 58945,
            58946, 58980, 58984, 58986, 58987, 58990, 58993, 58994, 59000, 59050,
            59230, 59231, 59232, 59401, 59435, 59438, 59439, 59440, 59441, 59443,
            59444, 59445, 59446, 59455, 59475, 59476, 59477, 59478, 59479, 59483,
            59484, 59489, 59490, 59492, 59493, 59494, 59495, 59496, 59497, 59498,
            59499, 59500, 59502, 59503, 59550, 59560, 59562, 59861, 59862, 59865
        ]

        for idx, claim in self.flagged_claims.iterrows():
            clm_code = claim['CLM CODE']
            patient_gender = claim['Gender']

            if patient_gender == 'Male' and clm_code in male_inappropriate_codes:
                self._add_flag(idx, f"Male cannot claim procedure code {clm_code}", "F")

    def _check_female_inappropriate_codes(self):
        """Check if female patients are claiming male-only procedure codes"""
        # List of procedure codes that females cannot claim
        female_inappropriate_codes = [
            19140, 52601, 52610, 52630, 52700, 53215, 53410, 53418, 53420, 53425,
            53440, 53505, 53515, 53600, 53605, 53620, 0, 54000, 54001, 54002, 54015,
            54050, 54055, 54060, 54065, 54100, 54105, 54110, 54115, 54120, 54125,
            54130, 54135, 54150, 54160, 54161, 54162, 54165, 54200, 54205, 54220,
            54300, 54305, 54320, 54325, 54330, 54380, 54385, 54390, 54400, 54420,
            54430, 54440, 54500, 54505, 54506, 54510, 54520, 54530, 54535, 54550,
            54560, 54600, 54620, 54640, 54645, 54660, 54670, 54680, 54700, 54800,
            54820, 54830, 54840, 54860, 54861, 54900, 54901, 55000, 55040, 55060,
            55100, 55120, 55150, 55170, 55200, 55250, 55300, 55400, 55450, 55500,
            55520, 55530, 55535, 55540, 55600, 55605, 55650, 55680, 55700, 55705,
            55720, 55725, 55740, 55801, 55810, 55821, 55831, 55840, 55845, 55970
        ]

        for idx, claim in self.flagged_claims.iterrows():
            clm_code = claim['CLM CODE']
            patient_gender = claim['Gender']

            if patient_gender == 'Female' and clm_code in female_inappropriate_codes:
                self._add_flag(idx, f"Female cannot claim procedure code {clm_code}", "F")

    def _check_age_appropriateness(self):
        """Check if service is appropriate for patient's age"""
        for idx, claim in self.flagged_claims.iterrows():
            current_age = claim['CURRENT AGE']
            clm_code = claim['CLM CODE']

            # Example age-based rules (customize as needed)
            if current_age < 18 and clm_code in [15831, 15832, 15833]:  # Cosmetic procedures
                self._add_flag(idx, "Age inappropriate for cosmetic procedure", "F")
            elif current_age > 65 and clm_code in [15775]:  # Hair transplant in elderly
                self._add_flag(idx, "Questionable procedure for age group", "W")
            # Add more age-specific rules as needed

    def _check_unit_overdosing(self):
        """Check if units claimed exceed maximum allowed units"""
        for idx, claim in self.flagged_claims.iterrows():
            clm_code = claim['CLM CODE']
            units_claimed = claim['UNITS']

            # Find matching tariff
            tariff_match = self.tariff_data[self.tariff_data['CLM CODE'] == clm_code]

            if not tariff_match.empty:
                # Check if MaxUnits column exists
                if 'MaxUnits' in tariff_match.columns:
                    max_units = tariff_match.iloc[0]['MaxUnits']

                    # Handle NaN values and convert to numeric
                    if pd.notna(max_units) and pd.notna(units_claimed):
                        try:
                            max_units = float(max_units)
                            units_claimed = float(units_claimed)
                            if units_claimed > max_units:
                                self._add_flag(idx,
                                               f"Units claimed ({units_claimed}) exceed maximum allowed ({max_units})",
                                               "W")
                        except (ValueError, TypeError):
                            pass  # Skip if conversion fails

    def _check_duplicate_claims(self):
        """Check for duplicate claims based on CLAIM NO and CLAIM LINE NO"""
        # Check if required columns exist
        if 'CLAIM NO' not in self.flagged_claims.columns:
            st.warning("CLAIM NO column not found. Skipping duplicate detection.")
            return

        if 'CLAIM LINE NO' not in self.flagged_claims.columns:
            st.warning("CLAIM LINE NO column not found. Using only CLAIM NO for duplicate detection.")
            # Fallback to using only CLAIM NO
            duplicate_mask = self.flagged_claims.duplicated(
                subset=['CLAIM NO'],
                keep=False
            )

            for idx in self.flagged_claims[duplicate_mask].index:
                self._add_flag(idx, "Duplicate claim - same claim number", "F")
            return

        # Method 1: Exact duplicates using CLAIM NO and CLAIM LINE NO
        duplicate_mask = self.flagged_claims.duplicated(
            subset=['CLAIM NO', 'CLAIM LINE NO'],
            keep=False
        )

        for idx in self.flagged_claims[duplicate_mask].index:
            self._add_flag(idx, "Exact duplicate claim - same claim number and claim line number", "F")

    def _check_mutually_exclusive_tariffs(self):
        """Check for mutually exclusive tariffs that cannot be claimed together"""
        # Define mutually exclusive code pairs and groups
        mutually_exclusive_pairs = [
            (76780, 76871),
            (76780, 76831)
        ]

        mutually_exclusive_groups = [
            ([90050], [90051]),  # 90050 and 90051 cannot be claimed together
            ([90000, 90001, 90003], [90002])  # 90000,90001,90003 and 90002 cannot be claimed together
        ]

        # Convert SERVICE DATE to datetime if it's not already
        try:
            self.flagged_claims['SERVICE_DATE'] = pd.to_datetime(self.flagged_claims['SERVICE DATE'])
        except:
            st.warning(
                "Could not parse SERVICE DATE. Skipping mutually exclusive tariff checks that require date comparison.")
            return

        # Check for mutually exclusive pairs on the same day for same member
        for code1, code2 in mutually_exclusive_pairs:
            self._check_mutually_exclusive_pair(code1, code2)

        # Check for mutually exclusive groups on the same day for same member
        for group1, group2 in mutually_exclusive_groups:
            self._check_mutually_exclusive_group(group1, group2)

        # Special check for 90002 - can only be claimed after a two-day period from 90000,90001,90003
        self._check_two_day_period_rule([90000, 90001, 90003], 90002)

    def _check_mutually_exclusive_pair(self, code1, code2):
        """Check if two specific codes are claimed together on the same day by the same member"""
        # Get claims for code1 and code2
        claims_code1 = self.flagged_claims[self.flagged_claims['CLM CODE'] == code1]
        claims_code2 = self.flagged_claims[self.flagged_claims['CLM CODE'] == code2]

        for idx1, claim1 in claims_code1.iterrows():
            member1 = claim1['MEMBER NO']
            date1 = claim1['SERVICE_DATE']

            # Check if same member has code2 on the same date
            matching_claims = claims_code2[
                (claims_code2['MEMBER NO'] == member1) &
                (claims_code2['SERVICE_DATE'] == date1)
                ]

            if not matching_claims.empty:
                # Flag both claims
                self._add_flag(idx1, f"Mutually exclusive tariffs - cannot claim with code {code2} on same day", "F")
                for idx2 in matching_claims.index:
                    self._add_flag(idx2, f"Mutually exclusive tariffs - cannot claim with code {code1} on same day",
                                   "F")

    def _check_mutually_exclusive_group(self, group1, group2):
        """Check if codes from two groups are claimed together on the same day by the same member"""
        # Get claims for group1 and group2
        claims_group1 = self.flagged_claims[self.flagged_claims['CLM CODE'].isin(group1)]
        claims_group2 = self.flagged_claims[self.flagged_claims['CLM CODE'].isin(group2)]

        for idx1, claim1 in claims_group1.iterrows():
            member1 = claim1['MEMBER NO']
            date1 = claim1['SERVICE_DATE']
            code1 = claim1['CLM CODE']

            # Check if same member has any code from group2 on the same date
            matching_claims = claims_group2[
                (claims_group2['MEMBER NO'] == member1) &
                (claims_group2['SERVICE_DATE'] == date1)
                ]

            if not matching_claims.empty:
                # Flag the group1 claim
                group2_str = ", ".join(map(str, group2))
                self._add_flag(idx1, f"Mutually exclusive tariffs - cannot claim with codes {group2_str} on same day",
                               "F")

                # Flag all matching group2 claims
                for idx2 in matching_claims.index:
                    group1_str = ", ".join(map(str, group1))
                    self._add_flag(idx2,
                                   f"Mutually exclusive tariffs - cannot claim with codes {group1_str} on same day",
                                   "F")

    def _check_two_day_period_rule(self, source_codes, target_code):
        """Check if target_code is claimed within two days of any source_code by the same member"""
        # Get claims for source codes and target code
        claims_source = self.flagged_claims[self.flagged_claims['CLM CODE'].isin(source_codes)]
        claims_target = self.flagged_claims[self.flagged_claims['CLM CODE'] == target_code]

        for idx_target, claim_target in claims_target.iterrows():
            member_target = claim_target['MEMBER NO']
            date_target = claim_target['SERVICE_DATE']

            # Check if same member has any source code within the previous two days
            claims_same_member = claims_source[claims_source['MEMBER NO'] == member_target]

            for idx_source, claim_source in claims_same_member.iterrows():
                date_source = claim_source['SERVICE_DATE']

                # Calculate day difference
                day_diff = (date_target - date_source).days

                # If target code is claimed within 2 days of source code, flag it
                if 0 <= day_diff <= 2:
                    source_codes_str = ", ".join(map(str, source_codes))
                    self._add_flag(idx_target,
                                   f"Mutually exclusive tariffs - code {target_code} cannot be claimed within two days of codes {source_codes_str}",
                                   "F")
                    break  # No need to check other source claims for this target

    def _add_flag(self, index, reason, flag_type):
        """Add FWA flag to claim"""
        self.flagged_claims.at[index, 'FWA_Flag'] = True
        existing_reason = self.flagged_claims.at[index, 'Flag_Reason']

        if existing_reason:
            self.flagged_claims.at[index, 'Flag_Reason'] = existing_reason + "; " + reason
        else:
            self.flagged_claims.at[index, 'Flag_Reason'] = reason

        # Set or update flag type
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

        # Calculate financial impact
        total_claimed_amount = self.flagged_claims['AMOUNT CLAIMED'].sum()
        flagged_claimed_amount = flagged_claims['AMOUNT CLAIMED'].sum()
        total_paid_amount = self.flagged_claims['TOTAL PAID'].sum()
        flagged_paid_amount = flagged_claims['TOTAL PAID'].sum()

        # Break down by flag type
        fraud_flags = flagged_claims[flagged_claims['Flag_Type'].str.contains('F', na=False)]
        waste_flags = flagged_claims[flagged_claims['Flag_Type'].str.contains('W', na=False)]
        abuse_flags = flagged_claims[flagged_claims['Flag_Type'].str.contains('A', na=False)]

        # Calculate potential recovery
        potential_recovery = flagged_claims['AMOUNT CLAIMED'].sum()

        # Gender-specific flag counts
        male_inappropriate_flags = flagged_claims[
            flagged_claims['Flag_Reason'].str.contains('Male cannot claim', na=False)]
        female_inappropriate_flags = flagged_claims[
            flagged_claims['Flag_Reason'].str.contains('Female cannot claim', na=False)]

        # Duplicate claim analysis
        duplicate_flags = flagged_claims[flagged_claims['Flag_Reason'].str.contains('duplicate claim', na=False)]

        # Mutually exclusive tariff analysis
        mutually_exclusive_flags = flagged_claims[
            flagged_claims['Flag_Reason'].str.contains('Mutually exclusive tariffs', na=False)]

        summary = f"""
        # FWA Detection Executive Summary

        ## Overview
        - **Total Claims Analyzed**: {total_claims:,}
        - **Flagged Claims**: {total_flagged:,} ({total_flagged / total_claims * 100:.1f}%)
        - **Total Claimed Amount**: ${total_claimed_amount:,.2f}
        - **Flagged Claimed Amount**: ${flagged_claimed_amount:,.2f} ({flagged_claimed_amount / total_claimed_amount * 100:.1f}%)
        - **Total Paid Amount**: ${total_paid_amount:,.2f}
        - **Flagged Paid Amount**: ${flagged_paid_amount:,.2f} ({flagged_paid_amount / total_paid_amount * 100:.1f}%)
        - **Potential Recovery**: ${potential_recovery:,.2f}

        ## Breakdown by FWA Category
        - **Fraud (F)**: {len(fraud_flags):,} claims
        - **Waste (W)**: {len(waste_flags):,} claims  
        - **Abuse (A)**: {len(abuse_flags):,} claims

        ## Key Issue Categories
        - **Duplicate Claims**: {len(duplicate_flags):,} claims (same claim & line numbers)
        - **Mutually Exclusive Tariffs**: {len(mutually_exclusive_flags):,} claims
        - **Male claiming female-only procedures**: {len(male_inappropriate_flags):,} claims
        - **Female claiming male-only procedures**: {len(female_inappropriate_flags):,} claims

        ## Key Findings
        {self._generate_key_findings(flagged_claims)}

        ## Recommendations
        1. **Immediate Action**: Review all Fraud (F) flagged claims for potential recovery
        2. **Mutually Exclusive Review**: Address claims with conflicting tariff codes
        3. **Duplicate Resolution**: Address exact duplicate claims with same claim and line numbers
        4. **Gender Compliance**: Address gender-inappropriate claims through provider education
        5. **Process Improvement**: Address Waste (W) patterns through billing guidelines
        6. **Continuous Monitoring**: Establish ongoing monitoring for high-risk providers and procedures
        """

        return summary

    def _generate_key_findings(self, flagged_claims):
        """Generate key findings from flagged claims"""
        if len(flagged_claims) == 0:
            return "No significant findings detected."

        findings = []

        # Top flagged providers
        top_providers = flagged_claims['PROVIDER NAME'].value_counts().head(3)
        if len(top_providers) > 0:
            findings.append(
                f"- **Top flagged providers**: {', '.join([f'{provider} ({count} claims)' for provider, count in top_providers.items()])}")

        # Most common procedure codes
        common_codes = flagged_claims['CLM CODE'].value_counts().head(3)
        if len(common_codes) > 0:
            code_descriptions = []
            for code, count in common_codes.items():
                # Try to get description from tariff data
                if hasattr(self, 'tariff_data') and self.tariff_data is not None:
                    desc_match = self.tariff_data[self.tariff_data['CLM CODE'] == code]
                    if not desc_match.empty and 'CODE DESCRIPTION' in desc_match.columns:
                        desc = desc_match.iloc[0]['CODE DESCRIPTION']
                        code_descriptions.append(f"Code {code} ({desc}): {count} claims")
                    else:
                        code_descriptions.append(f"Code {code}: {count} claims")
                else:
                    code_descriptions.append(f"Code {code}: {count} claims")
            findings.append(f"- **Most common flagged procedures**: {', '.join(code_descriptions)}")

        # Common flag reasons
        common_reasons = flagged_claims['Flag_Reason'].value_counts().head(5)
        if len(common_reasons) > 0:
            reasons_list = []
            for reason, count in common_reasons.items():
                # Truncate long reasons
                short_reason = reason.split(';')[0][:50] + '...' if len(reason) > 50 else reason.split(';')[0]
                reasons_list.append(f"{short_reason} ({count})")
            findings.append(f"- **Common flag reasons**: {', '.join(reasons_list)}")

        # Financial impact by flag type
        financial_by_type = flagged_claims.groupby('Flag_Type')['AMOUNT CLAIMED'].sum()
        if len(financial_by_type) > 0:
            financial_summary = []
            for flag_type, amount in financial_by_type.items():
                financial_summary.append(f"{flag_type}: ${amount:,.2f}")
            findings.append(f"- **Financial impact by type**: {', '.join(financial_summary)}")

        # Gender-specific findings
        male_claims = flagged_claims[flagged_claims['Flag_Reason'].str.contains('Male cannot claim', na=False)]
        female_claims = flagged_claims[flagged_claims['Flag_Reason'].str.contains('Female cannot claim', na=False)]

        if len(male_claims) > 0:
            findings.append(
                f"- **Male inappropriate claims**: {len(male_claims)} claims totaling ${male_claims['AMOUNT CLAIMED'].sum():,.2f}")
        if len(female_claims) > 0:
            findings.append(
                f"- **Female inappropriate claims**: {len(female_claims)} claims totaling ${female_claims['AMOUNT CLAIMED'].sum():,.2f}")

        # Duplicate findings
        duplicate_claims = flagged_claims[flagged_claims['Flag_Reason'].str.contains('duplicate claim', na=False)]
        if len(duplicate_claims) > 0:
            findings.append(f"- **Exact duplicates**: {len(duplicate_claims)} claims with same claim and line numbers")

        # Mutually exclusive findings
        mutually_exclusive_claims = flagged_claims[
            flagged_claims['Flag_Reason'].str.contains('Mutually exclusive tariffs', na=False)]
        if len(mutually_exclusive_claims) > 0:
            findings.append(
                f"- **Mutually exclusive tariffs**: {len(mutually_exclusive_claims)} claims with conflicting codes")

        return "\n".join(findings)


def create_modern_visualizations(flagged_claims, tariff_data):
    """Create modern visualization charts for FWA analysis using Plotly"""
    if flagged_claims is None or len(flagged_claims) == 0:
        st.warning("No data available for visualization")
        return

    flagged = flagged_claims[flagged_claims['FWA_Flag'] == True]

    if len(flagged) == 0:
        st.info("No flagged claims to visualize")
        return

    # Create tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["FWA Overview", "Financial Impact", "Provider Analysis"])

    with viz_tab1:
        col1, col2 = st.columns(2)

        with col1:
            # FWA Type Distribution - Pie Chart
            flag_types = []
            for types in flagged['Flag_Type'].dropna():
                if pd.notna(types):
                    flag_types.extend(types.split(','))

            if flag_types:
                type_counts = pd.Series(flag_types).value_counts()

                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue for F, W, A

                fig_pie = go.Figure(data=[go.Pie(
                    labels=type_counts.index,
                    values=type_counts.values,
                    hole=.3,
                    marker_colors=colors[:len(type_counts)]
                )])

                fig_pie.update_layout(
                    title="Distribution of FWA Types",
                    height=400,
                    showlegend=True
                )

                st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Issue Type Breakdown
            male_claims = flagged[flagged['Flag_Reason'].str.contains('Male cannot claim', na=False)]
            female_claims = flagged[flagged['Flag_Reason'].str.contains('Female cannot claim', na=False)]
            duplicate_claims = flagged[flagged['Flag_Reason'].str.contains('duplicate claim', na=False)]
            mutually_exclusive_claims = flagged[
                flagged['Flag_Reason'].str.contains('Mutually exclusive tariffs', na=False)]

            # Create combined chart for all issues
            issue_data = {
                'Type': ['Male Inappropriate', 'Female Inappropriate', 'Exact Duplicates', 'Mutually Exclusive'],
                'Count': [len(male_claims), len(female_claims), len(duplicate_claims), len(mutually_exclusive_claims)]
            }

            if any(issue_data['Count']):
                fig_issues = px.bar(
                    issue_data,
                    x='Type',
                    y='Count',
                    color='Type',
                    color_discrete_sequence=['#FF6B6B', '#45B7D1', '#4ECDC4', '#FFA726']
                )

                fig_issues.update_layout(
                    title="Key Issue Breakdown",
                    height=400,
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="Number of Claims"
                )

                st.plotly_chart(fig_issues, use_container_width=True)
            else:
                st.info("No key issues found")

    with viz_tab2:
        col1, col2 = st.columns(2)

        with col1:
            # Financial Impact by Flag Type
            financial_impact = flagged.groupby('Flag_Type')['AMOUNT CLAIMED'].sum()

            if len(financial_impact) > 0:
                fig_financial = px.bar(
                    x=financial_impact.index,
                    y=financial_impact.values,
                    color=financial_impact.index,
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
                )

                fig_financial.update_layout(
                    title="Financial Impact by FWA Type",
                    xaxis_title="FWA Type",
                    yaxis_title="Total Claimed Amount ($)",
                    height=400,
                    showlegend=False
                )

                st.plotly_chart(fig_financial, use_container_width=True)

        with col2:
            # Monthly Trend of Flagged Claims
            try:
                if 'SERVICE DATE' in flagged.columns:
                    flagged['SERVICE_DATE'] = pd.to_datetime(flagged['SERVICE DATE'])
                    monthly_trend = flagged.groupby(flagged['SERVICE_DATE'].dt.to_period('M')).size()
                    monthly_trend.index = monthly_trend.index.astype(str)

                    fig_trend = px.line(
                        x=monthly_trend.index,
                        y=monthly_trend.values,
                        markers=True
                    )

                    fig_trend.update_layout(
                        title="Monthly Trend of Flagged Claims",
                        xaxis_title="Month",
                        yaxis_title="Number of Flagged Claims",
                        height=400
                    )

                    fig_trend.update_traces(line_color='#667eea', marker_color='#764ba2')

                    st.plotly_chart(fig_trend, use_container_width=True)
            except:
                st.info("Could not generate trend chart - date format issue")

    with viz_tab3:
        col1, col2 = st.columns(2)

        with col1:
            # Top Providers with Flagged Claims
            if 'PROVIDER NAME' in flagged.columns:
                top_providers = flagged['PROVIDER NAME'].value_counts().head(10)

                if len(top_providers) > 0:
                    fig_providers = px.bar(
                        x=top_providers.values,
                        y=top_providers.index,
                        orientation='h',
                        color=top_providers.values,
                        color_continuous_scale='Viridis'
                    )

                    fig_providers.update_layout(
                        title="Top Providers with Flagged Claims",
                        xaxis_title="Number of Flagged Claims",
                        yaxis_title="Provider Name",
                        height=500,
                        showlegend=False
                    )

                    st.plotly_chart(fig_providers, use_container_width=True)

        with col2:
            # Flagged Claims by Benefit Type
            if 'BASE BENEFIT DESCRIPTION' in flagged.columns:
                benefit_counts = flagged['BASE BENEFIT DESCRIPTION'].value_counts().head(8)

                if len(benefit_counts) > 0:
                    fig_benefits = px.pie(
                        values=benefit_counts.values,
                        names=benefit_counts.index,
                        title="Flagged Claims by Benefit Type"
                    )

                    fig_benefits.update_layout(height=500)

                    st.plotly_chart(fig_benefits, use_container_width=True)


def create_comparison_visualizations(current_analysis, historical_analysis):
    """Create comparison visualizations between current and historical analysis"""
    if not current_analysis or not historical_analysis:
        st.warning("Both current and historical analysis data required for comparison")
        return

    current_flagged = current_analysis['flagged_claims'][current_analysis['flagged_claims']['FWA_Flag'] == True]
    historical_flagged = historical_analysis['data']['flagged_claims'][
        historical_analysis['data']['flagged_claims']['FWA_Flag'] == True]

    # Calculate metrics for comparison
    current_metrics = {
        'total_claims': len(current_analysis['flagged_claims']),
        'flagged_claims': len(current_flagged),
        'flagged_percentage': (len(current_flagged) / len(current_analysis['flagged_claims'])) * 100,
        'total_claimed_amount': current_analysis['flagged_claims']['AMOUNT CLAIMED'].sum(),
        'flagged_claimed_amount': current_flagged['AMOUNT CLAIMED'].sum(),
        'fraud_count': len(current_flagged[current_flagged['Flag_Type'].str.contains('F', na=False)]),
        'waste_count': len(current_flagged[current_flagged['Flag_Type'].str.contains('W', na=False)]),
        'abuse_count': len(current_flagged[current_flagged['Flag_Type'].str.contains('A', na=False)])
    }

    historical_metrics = {
        'total_claims': len(historical_analysis['data']['flagged_claims']),
        'flagged_claims': len(historical_flagged),
        'flagged_percentage': (len(historical_flagged) / len(historical_analysis['data']['flagged_claims'])) * 100,
        'total_claimed_amount': historical_analysis['data']['flagged_claims']['AMOUNT CLAIMED'].sum(),
        'flagged_claimed_amount': historical_flagged['AMOUNT CLAIMED'].sum(),
        'fraud_count': len(historical_flagged[historical_flagged['Flag_Type'].str.contains('F', na=False)]),
        'waste_count': len(historical_flagged[historical_flagged['Flag_Type'].str.contains('W', na=False)]),
        'abuse_count': len(historical_flagged[historical_flagged['Flag_Type'].str.contains('A', na=False)])
    }

    # Create comparison metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        flagged_change = current_metrics['flagged_percentage'] - historical_metrics['flagged_percentage']
        change_icon = "📈" if flagged_change > 0 else "📉" if flagged_change < 0 else "➡️"
        change_class = "metric-improvement" if flagged_change < 0 else "metric-decline" if flagged_change > 0 else ""

        st.markdown(f"""
        <div class="comparison-metric">
            <h3>Flagged %</h3>
            <div style="font-size: 2rem; font-weight: bold;">{current_metrics['flagged_percentage']:.1f}%</div>
            <div class="{change_class}">{change_icon} {flagged_change:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fraud_change = current_metrics['fraud_count'] - historical_metrics['fraud_count']
        change_icon = "📈" if fraud_change > 0 else "📉" if fraud_change < 0 else "➡️"
        change_class = "metric-improvement" if fraud_change < 0 else "metric-decline" if fraud_change > 0 else ""

        st.markdown(f"""
        <div class="comparison-metric">
            <h3>Fraud Cases</h3>
            <div style="font-size: 2rem; font-weight: bold;">{current_metrics['fraud_count']:,}</div>
            <div class="{change_class}">{change_icon} {fraud_change:+,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        waste_change = current_metrics['waste_count'] - historical_metrics['waste_count']
        change_icon = "📈" if waste_change > 0 else "📉" if waste_change < 0 else "➡️"
        change_class = "metric-improvement" if waste_change < 0 else "metric-decline" if waste_change > 0 else ""

        st.markdown(f"""
        <div class="comparison-metric">
            <h3>Waste Cases</h3>
            <div style="font-size: 2rem; font-weight: bold;">{current_metrics['waste_count']:,}</div>
            <div class="{change_class}">{change_icon} {waste_change:+,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        amount_change_pct = ((current_metrics['flagged_claimed_amount'] - historical_metrics[
            'flagged_claimed_amount']) / historical_metrics['flagged_claimed_amount']) * 100
        change_icon = "📈" if amount_change_pct > 0 else "📉" if amount_change_pct < 0 else "➡️"
        change_class = "metric-improvement" if amount_change_pct < 0 else "metric-decline" if amount_change_pct > 0 else ""

        st.markdown(f"""
        <div class="comparison-metric">
            <h3>Flagged Amount</h3>
            <div style="font-size: 1.5rem; font-weight: bold;">${current_metrics['flagged_claimed_amount']:,.0f}</div>
            <div class="{change_class}">{change_icon} {amount_change_pct:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Trend visualization
    st.markdown("### 📊 Trend Analysis")

    # Create trend data
    trend_data = {
        'Period': ['Historical', 'Current'],
        'Flagged Percentage': [historical_metrics['flagged_percentage'], current_metrics['flagged_percentage']],
        'Fraud Cases': [historical_metrics['fraud_count'], current_metrics['fraud_count']],
        'Waste Cases': [historical_metrics['waste_count'], current_metrics['waste_count']],
        'Abuse Cases': [historical_metrics['abuse_count'], current_metrics['abuse_count']]
    }

    fig_trend = go.Figure()

    # Add traces for each metric
    metrics_to_plot = ['Flagged Percentage', 'Fraud Cases', 'Waste Cases', 'Abuse Cases']
    colors = ['#667eea', '#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, metric in enumerate(metrics_to_plot):
        fig_trend.add_trace(go.Scatter(
            x=trend_data['Period'],
            y=trend_data[metric],
            name=metric,
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=8)
        ))

    fig_trend.update_layout(
        title="FWA Metrics Trend Comparison",
        xaxis_title="Analysis Period",
        yaxis_title="Metric Value",
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig_trend, use_container_width=True)


def get_download_link(df, filename):
    """Generate a download link for DataFrame"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Main flagged claims sheet
        df.to_excel(writer, index=False, sheet_name='Flagged_Claims')

        # Summary statistics sheet
        flagged_claims = df[df['FWA_Flag'] == True]
        summary_data = {
            'Metric': ['Total Claims', 'Flagged Claims', 'Flagged Percentage',
                       'Total Claimed Amount', 'Flagged Claimed Amount', 'Potential Recovery',
                       'Male Inappropriate Claims', 'Female Inappropriate Claims',
                       'Exact Duplicate Claims', 'Mutually Exclusive Tariffs'],
            'Value': [
                len(df),
                len(flagged_claims),
                f"{(len(flagged_claims) / len(df) * 100):.1f}%",
                f"${df['AMOUNT CLAIMED'].sum():,.2f}",
                f"${flagged_claims['AMOUNT CLAIMED'].sum():,.2f}",
                f"${flagged_claims['AMOUNT CLAIMED'].sum():,.2f}",
                f"{len(flagged_claims[flagged_claims['Flag_Reason'].str.contains('Male cannot claim', na=False)])}",
                f"{len(flagged_claims[flagged_claims['Flag_Reason'].str.contains('Female cannot claim', na=False)])}",
                f"{len(flagged_claims[flagged_claims['Flag_Reason'].str.contains('duplicate claim', na=False)])}",
                f"{len(flagged_claims[flagged_claims['Flag_Reason'].str.contains('Mutually exclusive tariffs', na=False)])}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, index=False, sheet_name='Summary')

    processed_data = output.getvalue()

    b64 = base64.b64encode(processed_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" class="summary-box" style="display: inline-block; padding: 0.5rem 1rem; background: #667eea; color: white; text-decoration: none; border-radius: 5px; font-weight: 500;">📥 Download Excel File</a>'
    return href


# Modern Streamlit App
def main():
    # Header with modern design
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div class="main-header">FWA Detection System</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Advanced Analytics for Fraud, Waste & Abuse Detection in Healthcare Claims</div>',
            unsafe_allow_html=True)

    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("### 🔍 Advanced Analytics")

    # Initialize session state
    if 'fwa_detector' not in st.session_state:
        st.session_state.fwa_detector = FWADetector()
    if 'tariff_uploaded' not in st.session_state:
        st.session_state.tariff_uploaded = False
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False
    if 'historical_analysis' not in st.session_state:
        st.session_state.historical_analysis = HistoricalAnalysis()
    if 'selected_historical_analysis' not in st.session_state:
        st.session_state.selected_historical_analysis = None
    if 'analysis_name' not in st.session_state:
        st.session_state.analysis_name = ""

    # Modern sidebar for file uploads
    with st.sidebar:
        st.markdown("### 📁 Data Management")

        # Tariff file upload
        st.markdown("#### 1. Upload Tariff File")
        st.markdown('<div class="upload-box">Drag and drop your tariff file here</div>', unsafe_allow_html=True)
        tariff_file = st.file_uploader("", type=['xlsx'], key='tariff', label_visibility="collapsed")

        if tariff_file:
            try:
                # Remove worksheet name - read first sheet by default
                st.session_state.tariff_data = pd.read_excel(tariff_file)
                st.session_state.tariff_uploaded = True

                st.success("✅ Tariff file uploaded successfully!")

                with st.expander("View Tariff Data Preview"):
                    st.dataframe(st.session_state.tariff_data.head(10), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error loading tariff file: {str(e)}")

        # Claims file upload
        st.markdown("#### 2. Upload Claims File")
        st.markdown('<div class="upload-box">Drag and drop your claims file here</div>', unsafe_allow_html=True)
        claims_file = st.file_uploader("", type=['xlsx'], key='claims', label_visibility="collapsed")

        # Analysis name input
        st.markdown("#### 3. Analysis Details")
        st.session_state.analysis_name = st.text_input(
            "Analysis Name",
            value=f"Analysis_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}",
            help="Give this analysis a meaningful name for historical tracking"
        )

        if st.session_state.tariff_uploaded and claims_file:
            if st.button("🚀 Run FWA Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing claims for FWA patterns..."):
                    success, message = st.session_state.fwa_detector.load_data(claims_file, tariff_file)
                    if success:
                        success, message = st.session_state.fwa_detector.detect_fwa()
                        if success:
                            st.session_state.analysis_run = True

                            # Save to historical analysis
                            analysis_id = str(uuid.uuid4())
                            analysis_data = {
                                'name': st.session_state.analysis_name,
                                'flagged_claims': st.session_state.fwa_detector.flagged_claims.copy(),
                                'tariff_data': st.session_state.tariff_data.copy()
                            }
                            st.session_state.historical_analysis.save_analysis(analysis_id, analysis_data)

                            st.success("✅ Analysis completed successfully!")
                        else:
                            st.error(f"❌ {message}")
                    else:
                        st.error(f"❌ {message}")

        # Quick stats in sidebar if analysis is run
        if st.session_state.analysis_run and hasattr(st.session_state.fwa_detector, 'flagged_claims'):
            flagged_data = st.session_state.fwa_detector.flagged_claims
            flagged_claims = flagged_data[flagged_data['FWA_Flag'] == True]

            st.markdown("---")
            st.markdown("### 📊 Quick Stats")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Claims", f"{len(flagged_data):,}")
            with col2:
                st.metric("Flagged", f"{len(flagged_claims):,}")

            col3, col4 = st.columns(2)
            with col3:
                flagged_pct = (len(flagged_claims) / len(flagged_data) * 100) if len(flagged_data) > 0 else 0
                st.metric("Flagged %", f"{flagged_pct:.1f}%")
            with col4:
                total_flagged_amount = flagged_claims['AMOUNT CLAIMED'].sum()
                st.metric("Flagged Amount", f"${total_flagged_amount:,.0f}")

    # Main content area with modern layout
    if st.session_state.tariff_uploaded:
        if st.session_state.analysis_run and hasattr(st.session_state.fwa_detector,
                                                     'flagged_claims') and st.session_state.fwa_detector.flagged_claims is not None:
            flagged_data = st.session_state.fwa_detector.flagged_claims

            # Create modern tabs - ADDED HISTORICAL ANALYSIS TAB
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📊 Executive Dashboard", "📈 Analytics", "🔍 Detailed View", "🕐 Historical Analysis", "📥 Export"])

            with tab1:
                # Key metrics at the top
                st.markdown("### 📈 Key Performance Indicators")

                flagged_claims = flagged_data[flagged_data['FWA_Flag'] == True]
                total_claims = len(flagged_data)
                total_flagged = len(flagged_claims)

                col1, col2, col3, col4, col5 = st.columns(5)

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
                    total_flagged_amount = flagged_claims['AMOUNT CLAIMED'].sum()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Flagged Amount</div>
                        <div class="metric-value">${total_flagged_amount:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col5:
                    potential_recovery = flagged_claims['AMOUNT CLAIMED'].sum()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Potential Recovery</div>
                        <div class="metric-value">${potential_recovery:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Executive Summary
                st.markdown("### 📋 Executive Summary")
                st.markdown(st.session_state.fwa_detector.generate_executive_summary())

                # Save analysis button
                st.markdown("---")
                col_save1, col_save2 = st.columns([3, 1])
                with col_save2:
                    if st.button("💾 Save to Historical Analysis", type="secondary"):
                        # Analysis is already saved when run, but this allows re-saving if needed
                        analysis_id = str(uuid.uuid4())
                        analysis_data = {
                            'name': f"{st.session_state.analysis_name}_resaved",
                            'flagged_claims': st.session_state.fwa_detector.flagged_claims.copy(),
                            'tariff_data': st.session_state.tariff_data.copy()
                        }
                        st.session_state.historical_analysis.save_analysis(analysis_id, analysis_data)
                        st.success("✅ Analysis saved to historical records!")

            with tab2:
                st.markdown("### 📊 Advanced Analytics")
                create_modern_visualizations(flagged_data, st.session_state.tariff_data)

            with tab3:
                st.markdown("### 🔍 Detailed Claims Analysis")

                # Filter options in expandable section
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
                                                    ["All", "Gender Issues", "Duplicate Claims",
                                                     "Mutually Exclusive Tariffs", "Unit Overdosing"])

                # Apply filters
                display_data = flagged_data
                if show_only_flagged:
                    display_data = display_data[display_data['FWA_Flag'] == True]

                if flag_type_filter != "All":
                    flag_char = flag_type_filter[0]  # Get F, W, or A
                    display_data = display_data[display_data['Flag_Type'].str.contains(flag_char, na=False)]

                if provider_filter != "All":
                    display_data = display_data[display_data['PROVIDER NAME'] == provider_filter]

                if issue_filter == "Gender Issues":
                    display_data = display_data[display_data['Flag_Reason'].str.contains('cannot claim', na=False)]
                elif issue_filter == "Duplicate Claims":
                    display_data = display_data[display_data['Flag_Reason'].str.contains('duplicate claim', na=False)]
                elif issue_filter == "Mutually Exclusive Tariffs":
                    display_data = display_data[
                        display_data['Flag_Reason'].str.contains('Mutually exclusive tariffs', na=False)]
                elif issue_filter == "Unit Overdosing":
                    display_data = display_data[display_data['Flag_Reason'].str.contains('Units claimed', na=False)]

                # Display the data with styling
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

                    st.info(
                        f"Showing {len(display_data)} flagged claims out of {len(flagged_claims)} total flagged claims")
                else:
                    st.warning("No claims match the current filter criteria")

            with tab4:  # NEW HISTORICAL ANALYSIS TAB
                st.markdown("### 🕐 Historical Analysis")

                # Get all historical analyses
                historical_analyses = st.session_state.historical_analysis.get_all_analyses()

                if not historical_analyses:
                    st.info("No historical analyses found. Run an analysis to start building your historical data.")
                else:
                    col_hist1, col_hist2 = st.columns([1, 2])

                    with col_hist1:
                        st.markdown("#### 📚 Saved Analyses")

                        # Display list of historical analyses
                        for analysis_id, analysis_data in historical_analyses:
                            is_active = st.session_state.selected_historical_analysis == analysis_id
                            active_class = "active" if is_active else ""

                            st.markdown(f"""
                            <div class="history-item {active_class}" onclick="this.classList.toggle('active')">
                                <strong>{analysis_data['data'].get('name', 'Unnamed Analysis')}</strong><br>
                                <small>{analysis_data['timestamp'].strftime('%Y-%m-%d %H:%M')}</small><br>
                                <small>Claims: {analysis_data['metadata']['total_claims']:,} | Flagged: {analysis_data['metadata']['flagged_claims']:,}</small>
                            </div>
                            """, unsafe_allow_html=True)

                            col_btn1, col_btn2 = st.columns(2)
                            with col_btn1:
                                if st.button("Select", key=f"select_{analysis_id}", use_container_width=True):
                                    st.session_state.selected_historical_analysis = analysis_id
                                    st.rerun()
                            with col_btn2:
                                if st.button("Delete", key=f"delete_{analysis_id}", use_container_width=True):
                                    if st.session_state.historical_analysis.delete_analysis(analysis_id):
                                        if st.session_state.selected_historical_analysis == analysis_id:
                                            st.session_state.selected_historical_analysis = None
                                        st.rerun()

                    with col_hist2:
                        if st.session_state.selected_historical_analysis:
                            selected_analysis = st.session_state.historical_analysis.get_analysis(
                                st.session_state.selected_historical_analysis)

                            if selected_analysis:
                                st.markdown(
                                    f"#### 📊 Analysis: {selected_analysis['data'].get('name', 'Unnamed Analysis')}")
                                st.markdown(f"**Date:** {selected_analysis['timestamp'].strftime('%Y-%m-%d %H:%M')}")

                                # Show comparison with current analysis
                                st.markdown("##### 📈 Comparison with Current Analysis")
                                create_comparison_visualizations(
                                    {
                                        'flagged_claims': flagged_data,
                                        'metadata': {
                                            'total_claims': len(flagged_data),
                                            'flagged_claims': len(flagged_data[flagged_data['FWA_Flag'] == True]),
                                            'total_claimed_amount': flagged_data['AMOUNT CLAIMED'].sum(),
                                            'flagged_claimed_amount': flagged_data[flagged_data['FWA_Flag'] == True][
                                                'AMOUNT CLAIMED'].sum(),
                                            'fraud_count': len(
                                                flagged_data[flagged_data['Flag_Type'].str.contains('F', na=False)]),
                                            'waste_count': len(
                                                flagged_data[flagged_data['Flag_Type'].str.contains('W', na=False)]),
                                            'abuse_count': len(
                                                flagged_data[flagged_data['Flag_Type'].str.contains('A', na=False)])
                                        }
                                    },
                                    selected_analysis
                                )

                                # Show historical analysis details
                                with st.expander("View Historical Analysis Details"):
                                    st.dataframe(selected_analysis['data']['flagged_claims'], use_container_width=True)
                        else:
                            st.info("Select a historical analysis from the list to view details and comparisons.")

            with tab5:
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
                            <li>Gender inappropriate claims breakdown</li>
                            <li>Duplicate claims analysis</li>
                            <li>Mutually exclusive tariffs analysis</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("Generate Download File", type="primary"):
                        st.markdown(get_download_link(flagged_data, "FWA_Analysis_Results.xlsx"),
                                    unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    <div class="card">
                        <h3>📈 Visualization Export</h3>
                        <p>Export visualizations for reporting:</p>
                        <ul>
                            <li>High-quality charts and graphs</li>
                            <li>Executive dashboard views</li>
                            <li>Presentation-ready formats</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("Export Visualizations"):
                        st.info("Visualization export feature coming soon!")

        else:
            # Modern empty state
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem;">
                <h2>🚀 Ready to Analyze Claims</h2>
                <p style="font-size: 1.1rem; color: #6c757d; max-width: 600px; margin: 0 auto 2rem;">
                    Upload your claims data and run the analysis to uncover potential fraud, waste, and abuse patterns.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Sample data structure
            with st.expander("📋 Expected Data Structure", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("""
                    **Claims Data Should Include:**
                    - `MEMBER NO`, `Gender`, `CLM CODE`, `SERVICE DATE`
                    - `AMOUNT CLAIMED`, `TOTAL PAID`, `TARIFF`, `UNITS`
                    - `PROVIDER NAME`, `BASE BENEFIT DESCRIPTION`
                    - `CLAIM NO`, `CLAIM LINE NO` (for duplicate detection)
                    """)

                with col2:
                    st.markdown("""
                    **Detection Capabilities:**
                    - ✅ Gender appropriateness checks
                    - ✅ Procedure code validation
                    - ✅ Age appropriateness
                    - ✅ Unit overdosing detection
                    - ✅ Duplicate claim identification (using Claim No + Line No)
                    - ✅ Mutually exclusive tariffs detection
                    - ✅ Historical analysis tracking
                    - ✅ Trend comparison and visualization
                    """)

    else:
        # Modern landing page when no tariff is uploaded
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <h2>🔍 Welcome to FWA Detection System</h2>
            <p style="font-size: 1.1rem; color: #6c757d; max-width: 700px; margin: 0 auto 2rem;">
                Advanced analytics platform for detecting Fraud, Waste, and Abuse in healthcare claims. 
                Get started by uploading your tariff file in the sidebar.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Features grid
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            <div class="card">
                <h3>🛡️ Fraud Detection</h3>
                <p>Identify intentional deception for financial gain through comprehensive pattern analysis.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <h3>💸 Waste Prevention</h3>
                <p>Detect unnecessary costs from inefficient practices and overutilization of services.</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="card">
                <h3>⚖️ Abuse Monitoring</h3>
                <p>Uncover improper practices that don't meet criminal standards but violate policies.</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="card">
                <h3>📊 Historical Tracking</h3>
                <p>Track FWA trends over time with comprehensive historical analysis and comparison tools.</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()