import base64
import calendar
import hashlib
import json
import sqlite3
import time
import warnings
from datetime import datetime
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

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Gen-Health FWA",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  (dark-teal analytical theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --surface2:  #21262d;
    --border:    #30363d;
    --accent:    #00d4aa;
    --accent2:   #3d9eff;
    --danger:    #ff5757;
    --warn:      #ffa657;
    --ok:        #3fb950;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --radius:    12px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
    background: var(--bg);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Main background ── */
.main .block-container { background: var(--bg); padding: 1.5rem 2rem 3rem; }

/* ── Metric cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 1.5rem; }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
       padding: 1.1rem 1rem; text-align: center; position: relative; overflow: hidden; }
.kpi::before { content:''; position: absolute; top:0; left:0; right:0; height:3px; background: var(--accent); }
.kpi.danger::before { background: var(--danger); }
.kpi.warn::before   { background: var(--warn); }
.kpi.ok::before     { background: var(--ok); }
.kpi.blue::before   { background: var(--accent2); }
.kpi-val { font-size: 1.75rem; font-weight: 700; line-height:1; margin-bottom: .25rem; color: var(--text); }
.kpi-lbl { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing:.06em; }

/* ── Section headings ── */
.sec-head { font-size: 1rem; font-weight: 600; color: var(--muted); text-transform: uppercase;
            letter-spacing: .08em; border-bottom: 1px solid var(--border); padding-bottom: .4rem;
            margin: 1.6rem 0 .9rem; }

/* ── Alert banners ── */
.alert { border-radius: var(--radius); padding: .85rem 1.1rem; margin: .6rem 0; font-size: .9rem; }
.alert.danger { background: rgba(255,87,87,.12); border-left: 4px solid var(--danger); }
.alert.warn   { background: rgba(255,166,87,.12); border-left: 4px solid var(--warn); }
.alert.ok     { background: rgba(63,185,80,.12);  border-left: 4px solid var(--ok); }
.alert.info   { background: rgba(61,158,255,.12); border-left: 4px solid var(--accent2); }

/* ── Insight cards ── */
.insight { background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius);
           padding: .9rem 1.1rem; margin: .5rem 0; font-size: .88rem; }

/* ── Upload area ── */
.upload-hint { border: 1.5px dashed var(--border); border-radius: var(--radius);
               padding: 1rem; text-align: center; color: var(--muted); font-size: .82rem;
               margin-bottom: .5rem; background: var(--surface2); }

/* ── Streamlit overrides ── */
div[data-testid="stTabs"] button { font-size: .82rem; font-weight: 500; color: var(--muted); }
div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
.stDataFrame { border-radius: var(--radius); overflow: hidden; }
div[data-testid="metric-container"] { background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: .8rem 1rem; }
div[data-testid="metric-container"] label { color: var(--muted) !important; font-size:.75rem; text-transform:uppercase; }
div[data-testid="metric-container"] div[data-testid="metric-value"] { color: var(--text) !important; font-size:1.4rem; }
button[kind="primary"] { background: var(--accent) !important; color: #0d1117 !important; font-weight:600; border:none; }
button[kind="primary"]:hover { opacity:.88; }
.stProgress > div > div > div > div { background: var(--accent); }
div[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#8b949e", size=11),
    title_font=dict(family="DM Sans", color="#e6edf3", size=13),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#30363d", zerolinecolor="#30363d"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b949e", size=10)),
    margin=dict(l=10, r=10, t=40, b=10),
)

ACCENT = "#00d4aa"
ACCENT2 = "#3d9eff"
DANGER = "#ff5757"
WARN = "#ffa657"
COLORS = [ACCENT, ACCENT2, WARN, DANGER, "#d2a8ff", "#79c0ff", "#56d364"]


def apply_theme(fig, height=380):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    return fig


# ─────────────────────────────────────────────
# ANALYTICS ENGINE
# ─────────────────────────────────────────────
class Analytics:
    @staticmethod
    def benfords_law(series):
        try:
            first_digits = []
            for v in series.dropna():
                s = str(abs(float(v))).replace(".", "").lstrip("0")
                if s and s[0].isdigit():
                    first_digits.append(int(s[0]))
            if not first_digits:
                return None
            obs = pd.Series(first_digits).value_counts().sort_index()
            obs_pct = obs / obs.sum() * 100
            exp = pd.Series({d: np.log10(1 + 1 / d) * 100 for d in range(1, 10)})
            chi2 = sum(((obs_pct.get(d, 0) - exp[d]) ** 2) / exp[d] for d in exp.index)
            return dict(observed=obs_pct, expected=exp, chi_square=chi2,
                        p_value=1 - stats.chi2.cdf(chi2, df=8))
        except Exception:
            return None

    @staticmethod
    def detect_outliers(data, column, method="iqr"):
        vals = pd.to_numeric(data[column], errors="coerce").dropna()
        if method == "iqr":
            Q1, Q3 = vals.quantile(.25), vals.quantile(.75)
            IQR = Q3 - Q1
            mask = (vals < Q1 - 1.5 * IQR) | (vals > Q3 + 1.5 * IQR)
        elif method == "zscore":
            mask = np.abs(stats.zscore(vals.fillna(vals.mean()))) > 3
        else:
            X = StandardScaler().fit_transform(vals.values.reshape(-1, 1))
            mask = IsolationForest(contamination=0.1, random_state=42).fit_predict(X) == -1
        out = vals[mask]
        return dict(outliers=out, count=len(out), pct=len(out) / len(vals) * 100,
                    mean=vals.mean(), median=vals.median(), std=vals.std())

    @staticmethod
    def provider_risk(claims, flagged):
        if "PROVIDER NAME" not in claims.columns:
            return None
        ps = claims.groupby("PROVIDER NAME").agg(
            claim_count=("TOTAL PAID", "count"),
            total_amount=("TOTAL PAID", "sum"),
            avg_amount=("TOTAL PAID", "mean"),
            std_amount=("TOTAL PAID", "std"),
            unique_members=("MEMBER NO", "nunique"),
        ).round(2)
        fl = flagged[flagged["FWA_Flag"]].groupby("PROVIDER NAME").agg(
            flagged_amount=("TOTAL PAID", "sum"),
            fraud_count=("Flag_Type", lambda x: x.str.contains("F", na=False).sum()),
        )
        pr = ps.merge(fl, how="left", left_index=True, right_index=True).fillna(0)
        pr["flag_rate"] = pr["fraud_count"] / pr["claim_count"] * 100
        pr["amount_flag_rate"] = pr["flagged_amount"] / pr["total_amount"].replace(0, np.nan) * 100
        for col in ["flag_rate", "amount_flag_rate", "avg_amount", "std_amount"]:
            std = pr[col].std()
            pr[f"{col}_z"] = (pr[col] - pr[col].mean()) / std if std > 0 else 0
        pr["risk_score"] = (pr["flag_rate_z"] * .4 + pr["amount_flag_rate_z"] * .3
                            + pr["avg_amount_z"] * .2 + pr["std_amount_z"] * .1)
        pr["risk_level"] = pd.cut(pr["risk_score"], bins=[-np.inf, -1, 1, np.inf],
                                  labels=["Low", "Medium", "High"])
        return pr.sort_values("risk_score", ascending=False)


# ─────────────────────────────────────────────
# IBNR CHAIN LADDER ENGINE
# ─────────────────────────────────────────────
def ibnr_parse_excel_date(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val
    if isinstance(val, (int, float)):
        try:
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(val))
        except Exception:
            return None
    try:
        return pd.to_datetime(val)
    except Exception:
        return None


def ibnr_build_claims(df, agg_type="month"):
    rows = []
    for _, row in df.iterrows():
        acc = ibnr_parse_excel_date(row.get("SERVICE DATE") or row.get("service date"))
        rep = ibnr_parse_excel_date(row.get("DATE RECEIVED") or row.get("date received"))
        paid_raw = row.get("TOTAL PAID") or row.get("total paid")
        if acc is None or rep is None:
            continue
        try:
            paid = float(paid_raw)
        except (TypeError, ValueError):
            continue
        if paid <= 0:
            continue
        diff_days = (rep - acc).days
        lag = max(0, min(48, round(diff_days / 30.44)))
        if agg_type == "month":
            period = f"{acc.year}-{acc.month:02d}"
        else:
            period = str(acc.year)
        rows.append({"period": period, "lag": lag, "paid": paid})
    return rows


def ibnr_build_triangles(claims_rows, periods_set, lags_set):
    inc_map = {p: {} for p in periods_set}
    for r in claims_rows:
        p, lag, paid = r["period"], r["lag"], r["paid"]
        if p in inc_map:
            inc_map[p][lag] = inc_map[p].get(lag, 0) + paid
    sorted_periods = sorted(periods_set)
    sorted_lags = sorted(lags_set)
    cumulative = []
    for p in sorted_periods:
        running, row_cum = 0, []
        for lag in sorted_lags:
            running += inc_map[p].get(lag, 0)
            row_cum.append(running)
        cumulative.append(row_cum)
    return cumulative, sorted_periods, sorted_lags


def ibnr_calc_factors(cum_matrix, lags):
    factors = []
    for i in range(len(lags) - 1):
        ratios = []
        for row in cum_matrix:
            cur, nxt = row[i], row[i + 1]
            if cur > 0 and nxt >= 0:
                ratios.append(nxt / cur)
        avg = max(1.0, sum(ratios) / len(ratios)) if ratios else 1.0
        factors.append({"from": lags[i], "to": lags[i + 1], "factor": avg})
    return factors


def ibnr_run_chain_ladder(claims_rows, agg_type, tail=1.05):
    if not claims_rows:
        raise ValueError("No valid claim rows parsed.")
    periods_set = set(r["period"] for r in claims_rows)
    lags_set = set(r["lag"] for r in claims_rows)
    if len(periods_set) < 2:
        raise ValueError("Need at least 2 accident periods.")
    full_lags = set(range(max(lags_set) + 1))
    cumulative, periods, lags = ibnr_build_triangles(claims_rows, periods_set, full_lags)
    factors_arr = ibnr_calc_factors(cumulative, lags)
    factor_map = {f"{f['from']}->{f['to']}": f["factor"] for f in factors_arr}

    latest_cum, latest_idx = [], []
    for row in cumulative:
        last_idx = next((j for j in range(len(row) - 1, -1, -1) if row[j] > 0), -1)
        latest_cum.append(row[last_idx] if last_idx >= 0 else 0)
        latest_idx.append(last_idx)

    ultimate_vec = []
    for i, proj in enumerate(latest_cum):
        if proj == 0:
            ultimate_vec.append(0)
            continue
        for step in range(latest_idx[i], len(lags) - 1):
            key = f"{lags[step]}->{lags[step+1]}"
            proj *= factor_map.get(key, 1.0)
        proj *= tail
        ultimate_vec.append(proj)

    ibnr = [max(0, u - c) for u, c in zip(ultimate_vec, latest_cum)]
    total_paid = sum(latest_cum)
    total_ult = sum(ultimate_vec)
    total_ibnr = total_ult - total_paid
    return dict(
        periods=periods, lags=lags, cumulative=cumulative,
        factors=factors_arr, current=latest_cum,
        ultimate=ultimate_vec, ibnr=ibnr,
        total_paid=total_paid, total_ultimate=total_ult, total_ibnr=total_ibnr
    )


def show_ibnr():
    st.markdown('<div class="sec-head">⚖️ IBNR Reserve · Chain Ladder Estimation</div>', unsafe_allow_html=True)
    st.caption("Upload a claims file (SERVICE DATE, DATE RECEIVED, TOTAL PAID) or load sample data to estimate Incurred But Not Reported reserves.")

    col_up, col_params = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Claims Excel file (.xlsx)", type=["xlsx", "xls"], key="ibnr_upload")
    with col_params:
        agg_type = st.selectbox("Accident Period Aggregation", ["month", "year"], key="ibnr_agg",
                                format_func=lambda x: "Monthly (YYYY-MM)" if x == "month" else "Annual (YYYY)")
        tail = st.slider("Tail Factor", 1.00, 1.20, 1.05, 0.01, key="ibnr_tail")

    col_run, col_sample = st.columns([1, 1])
    with col_run:
        run_btn = st.button("⚡ Run Chain Ladder", type="primary", disabled=uploaded is None, key="ibnr_run")
    with col_sample:
        sample_btn = st.button("📋 Load Sample Data", key="ibnr_sample")

    # Build sample data
    sample_data = [
        {"SERVICE DATE": "2025-01-10", "DATE RECEIVED": "2025-01-12", "TOTAL PAID": 1200},
        {"SERVICE DATE": "2025-01-15", "DATE RECEIVED": "2025-02-20", "TOTAL PAID": 3400},
        {"SERVICE DATE": "2025-02-05", "DATE RECEIVED": "2025-02-10", "TOTAL PAID": 900},
        {"SERVICE DATE": "2025-02-18", "DATE RECEIVED": "2025-04-01", "TOTAL PAID": 2100},
        {"SERVICE DATE": "2025-03-03", "DATE RECEIVED": "2025-03-08", "TOTAL PAID": 750},
        {"SERVICE DATE": "2025-03-22", "DATE RECEIVED": "2025-06-10", "TOTAL PAID": 4200},
        {"SERVICE DATE": "2025-04-07", "DATE RECEIVED": "2025-04-12", "TOTAL PAID": 1800},
        {"SERVICE DATE": "2025-04-28", "DATE RECEIVED": "2025-07-15", "TOTAL PAID": 3100},
        {"SERVICE DATE": "2025-05-14", "DATE RECEIVED": "2025-05-20", "TOTAL PAID": 2200},
        {"SERVICE DATE": "2025-05-30", "DATE RECEIVED": "2025-08-25", "TOTAL PAID": 5600},
        {"SERVICE DATE": "2025-06-12", "DATE RECEIVED": "2025-06-18", "TOTAL PAID": 980},
        {"SERVICE DATE": "2025-06-25", "DATE RECEIVED": "2025-09-30", "TOTAL PAID": 2900},
        {"SERVICE DATE": "2025-07-08", "DATE RECEIVED": "2025-07-14", "TOTAL PAID": 1350},
        {"SERVICE DATE": "2025-07-20", "DATE RECEIVED": "2025-10-12", "TOTAL PAID": 4700},
        {"SERVICE DATE": "2025-08-03", "DATE RECEIVED": "2025-08-09", "TOTAL PAID": 2400},
        {"SERVICE DATE": "2025-08-19", "DATE RECEIVED": "2025-11-22", "TOTAL PAID": 3800},
        {"SERVICE DATE": "2025-09-05", "DATE RECEIVED": "2025-09-10", "TOTAL PAID": 1100},
        {"SERVICE DATE": "2025-09-28", "DATE RECEIVED": "2025-12-15", "TOTAL PAID": 6200},
        {"SERVICE DATE": "2025-10-15", "DATE RECEIVED": "2025-10-22", "TOTAL PAID": 1700},
        {"SERVICE DATE": "2025-10-30", "DATE RECEIVED": "2026-01-18", "TOTAL PAID": 2800},
        {"SERVICE DATE": "2025-11-12", "DATE RECEIVED": "2025-11-18", "TOTAL PAID": 3100},
        {"SERVICE DATE": "2025-12-02", "DATE RECEIVED": "2025-12-08", "TOTAL PAID": 900},
        {"SERVICE DATE": "2025-12-20", "DATE RECEIVED": "2026-03-10", "TOTAL PAID": 5100},
        {"SERVICE DATE": "2026-01-10", "DATE RECEIVED": "2026-01-16", "TOTAL PAID": 2000},
        {"SERVICE DATE": "2026-01-25", "DATE RECEIVED": "2026-04-05", "TOTAL PAID": 3700},
        {"SERVICE DATE": "2026-02-08", "DATE RECEIVED": "2026-02-14", "TOTAL PAID": 1450},
        {"SERVICE DATE": "2026-02-22", "DATE RECEIVED": "2026-05-20", "TOTAL PAID": 4100},
        {"SERVICE DATE": "2026-03-11", "DATE RECEIVED": "2026-03-17", "TOTAL PAID": 2600},
        {"SERVICE DATE": "2026-04-05", "DATE RECEIVED": "2026-04-10", "TOTAL PAID": 1900},
        {"SERVICE DATE": "2026-04-22", "DATE RECEIVED": "2026-07-01", "TOTAL PAID": 3500},
    ]

    result = None

    if sample_btn:
        try:
            claims_rows = ibnr_build_claims(pd.DataFrame(sample_data), agg_type)
            result = ibnr_run_chain_ladder(claims_rows, agg_type, tail)
            st.session_state["ibnr_result"] = result
            st.success(f"Sample data loaded — {len(claims_rows)} claim rows processed.")
        except Exception as e:
            st.error(f"Error: {e}")

    if run_btn and uploaded:
        try:
            df = pd.read_excel(uploaded)
            df.columns = [c.strip().upper() for c in df.columns]
            claims_rows = ibnr_build_claims(df, agg_type)
            result = ibnr_run_chain_ladder(claims_rows, agg_type, tail)
            st.session_state["ibnr_result"] = result
            st.success(f"Analysis complete — {len(claims_rows)} claim rows processed.")
        except Exception as e:
            st.error(f"Error: {e}")

    # Display stored result
    if "ibnr_result" in st.session_state:
        result = st.session_state["ibnr_result"]
        r = result

        # KPI cards
        st.markdown(f"""
        <div class="kpi-grid">
            {kpi(f"${r['total_paid']:,.0f}", "Total Paid to Date", "blue")}
            {kpi(f"${r['total_ultimate']:,.0f}", "Ultimate Projected", "warn")}
            {kpi(f"${r['total_ibnr']:,.0f}", "IBNR Reserve", "danger")}
            {kpi(str(len(r['periods'])), "Accident Periods")}
            {kpi(str(len(r['lags'])), "Lag Buckets")}
        </div>
        """, unsafe_allow_html=True)

        tab_tri, tab_factors, tab_reserves, tab_charts = st.tabs([
            "📐 Development Triangle", "📊 Development Factors", "💰 Reserve Detail", "📈 Charts"
        ])

        with tab_tri:
            sec("Cumulative Paid Triangle")
            tri_df = pd.DataFrame(
                r["cumulative"],
                index=r["periods"],
                columns=[f"Lag {l}" for l in r["lags"]]
            ).round(2)
            tri_df.index.name = "Accident Period"
            st.dataframe(tri_df, use_container_width=True)

            # Heatmap
            fig_heat = go.Figure(go.Heatmap(
                z=r["cumulative"],
                x=[f"Lag {l}m" for l in r["lags"]],
                y=r["periods"],
                colorscale="Blues",
                text=[[f"{v:,.0f}" for v in row] for row in r["cumulative"]],
                texttemplate="%{text}",
                showscale=True,
            ))
            fig_heat.update_layout(**PLOTLY_LAYOUT, title="Cumulative Paid Development Triangle", height=420)
            st.plotly_chart(fig_heat, use_container_width=True)

        with tab_factors:
            sec("Age-to-Age Development Factors")
            factors_df = pd.DataFrame(r["factors"])
            factors_df.columns = ["From Lag", "To Lag", "Factor"]
            factors_df["Factor"] = factors_df["Factor"].round(4)
            st.dataframe(factors_df, use_container_width=True, hide_index=True)

            fig_factors = go.Figure(go.Bar(
                x=[f"{f['from']}→{f['to']}" for f in r["factors"]],
                y=[f["factor"] for f in r["factors"]],
                marker=dict(color=ACCENT2, opacity=0.85),
            ))
            fig_factors.add_hline(y=1.0, line_dash="dot", line_color=WARN, annotation_text="Factor = 1.0")
            fig_factors.update_layout(**PLOTLY_LAYOUT, title="Development Factors by Lag Interval", height=340,
                                      xaxis_title="Lag Transition", yaxis_title="Factor")
            st.plotly_chart(fig_factors, use_container_width=True)

        with tab_reserves:
            sec("IBNR Reserves by Accident Period")
            reserve_df = pd.DataFrame({
                "Accident Period": r["periods"],
                "Current Paid ($)": [round(v, 2) for v in r["current"]],
                "Ultimate ($)": [round(v, 2) for v in r["ultimate"]],
                "IBNR Reserve ($)": [round(v, 2) for v in r["ibnr"]],
            })
            st.dataframe(reserve_df, use_container_width=True, hide_index=True)

            # Download button
            csv = reserve_df.to_csv(index=False)
            st.download_button(
                label="📥 Export Reserves CSV",
                data=csv,
                file_name=f"ibnr_reserves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="ibnr_download"
            )

        with tab_charts:
            col1, col2 = st.columns(2)
            with col1:
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=r["periods"], y=r["current"], name="Paid to Date",
                    marker=dict(color=ACCENT2, opacity=0.85)
                ))
                fig_bar.add_trace(go.Bar(
                    x=r["periods"], y=r["ibnr"], name="IBNR Reserve",
                    marker=dict(color=DANGER, opacity=0.85)
                ))
                fig_bar.update_layout(**PLOTLY_LAYOUT, barmode="stack",
                                      title="Paid vs IBNR by Accident Period", height=380,
                                      xaxis_title="Period", yaxis_title="Amount ($)")
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                pct_ibnr = [
                    (b / u * 100) if u > 0 else 0
                    for b, u in zip(r["ibnr"], r["ultimate"])
                ]
                fig_pct = go.Figure(go.Scatter(
                    x=r["periods"], y=pct_ibnr, mode="lines+markers",
                    line=dict(color=WARN, width=2.5),
                    marker=dict(size=6, color=WARN),
                    fill="tozeroy",
                    fillcolor=f"rgba(255,166,87,0.08)"
                ))
                fig_pct.update_layout(**PLOTLY_LAYOUT, title="IBNR % of Ultimate by Period", height=380,
                                      xaxis_title="Period", yaxis_title="IBNR %")
                st.plotly_chart(fig_pct, use_container_width=True)

            # Waterfall: paid → ultimate
            fig_wf = go.Figure(go.Waterfall(
                x=r["periods"] + ["TOTAL"],
                measure=["relative"] * len(r["periods"]) + ["total"],
                y=r["ibnr"] + [sum(r["ibnr"])],
                name="IBNR Build-up",
                connector=dict(line=dict(color=ACCENT, width=1, dash="dot")),
                increasing=dict(marker=dict(color=DANGER)),
                totals=dict(marker=dict(color=WARN)),
            ))
            fig_wf.update_layout(**PLOTLY_LAYOUT, title="IBNR Reserve Waterfall by Period", height=380)
            st.plotly_chart(fig_wf, use_container_width=True)


# ─────────────────────────────────────────────
# FWA DETECTOR
# ─────────────────────────────────────────────
class FWADetector:
    FEMALE_CODES = {str(c) for c in [
        "19121","52270","52285","53210","53430","53660","53665","55980","56420","56440","56441",
        "56500","56505","56510","56602","56620","56625","56641","56700","56740","56750","57000",
        "57010","57100","57105","57120","57130","57210","57240","57245","57250","57256","57260",
        "57262","57265","57267","57280","57288","57290","57300","57320","57410","57452","57500",
        "57510","57521","57525","57530","57540","57550","57560","57600","57700","57701","57720",
        "58101","58120","58130","58140","58145","58146","58150","58155","58210","58260","58265",
        "58270","58300","58301","58320","58340","58350","58351","58400","58410","58500","58520",
        "58540","58600","58700","58720","58741","58743","58744","58830","58840","58841","58900",
        "58940","58945","58946","58980","58984","58986","58987","58990","58993","58994","59000",
        "59050","59230","59231","59232","59401","59435","59438","59439","59440","59441","59443",
        "59444","59445","59446","59455","59475","59476","59477","59478","59479","59483","59484",
        "59489","59490","59492","59493","59494","59495","59496","59497","59498","59499","59500",
        "59502","59503","59550","59560","59562","59861","59862","59865",
    ]}
    MALE_CODES = {str(c) for c in [
        "19140","52601","52610","52630","52700","53215","53410","53418","53420","53425","53440",
        "53505","53515","53600","53605","53620","54000","54001","54002","54015","54050","54055",
        "54060","54065","54100","54105","54110","54115","54120","54125","54130","54135","54150",
        "54160","54161","54162","54165","54200","54205","54220","54300","54305","54320","54325",
        "54330","54380","54385","54390","54400","54420","54430","54440","54500","54505","54506",
        "54510","54520","54530","54535","54550","54560","54600","54620","54640","54645","54660",
        "54670","54680","54700","54800","54820","54830","54840","54860","54861","54900","54901",
        "55000","55040","55060","55100","55120","55150","55170","55200","55250","55300","55400",
        "55450","55500","55520","55530","55535","55540","55600","55605","55650","55680","55700",
        "55705","55720","55725","55740","55801","55810","55821","55831","55840","55845","55970",
    ]}

    def __init__(self):
        self.claims = None
        self.tariff = None
        self.flagged = None
        self.analytics = Analytics()

    # ── Load ──────────────────────────────────
    def load_data(self, claims_file, tariff_file):
        CLAIMS_MAP = {
            "CLM CODE":   ["CLM CODE","CLM_CODE","CLAIM CODE","Procedure Code"],
            "Gender":     ["Gender","SEX","GENDER"],
            "CURRENT AGE":["CURRENT AGE","Age","AGE","Patient Age"],
            "UNITS":      ["UNITS","Units","QUANTITY"],
            "TOTAL PAID": ["TOTAL PAID","Total Paid","PAID AMOUNT","Amount Paid"],
            "PROVIDER NAME":["PROVIDER NAME","Provider","PROVIDER"],
            "MEMBER NO":  ["MEMBER NO","Member ID","MEMBER_ID","Patient ID"],
            "SERVICE DATE":["SERVICE DATE","Service Date","DATE OF SERVICE","ASSESS DATE","Assess Date"],
            "BASE BENEFIT DESCRIPTION":["BASE BENEFIT DESCRIPTION","Benefit Description","DESCRIPTION"],
            "CLAIM NO":   ["CLAIM NO","Claim Number","CLAIM_NUMBER"],
            "CLAIM LINE NO":["CLAIM LINE NO","Line Number","CLAIM_LINE"],
        }
        TARIFF_MAP = {
            "CLM CODE":    ["CLM CODE","CLM_CODE","Procedure Code"],
            "AllowedGender":["AllowedGender","Gender Allowed","GENDER_ALLOWED"],
            "MaxUnits":    ["MaxUnits","Maximum Units","MAX_UNITS"],
            "CODE DESCRIPTION":["CODE DESCRIPTION","Description","PROC_DESC"],
        }
        try:
            df  = pd.read_excel(claims_file, sheet_name="Sheet1")
            tdf = pd.read_excel(tariff_file, sheet_name="Sample Tariff")

            def remap(df, mapping):
                m, missing = {}, []
                for exp, variants in mapping.items():
                    found = next((c for c in df.columns if c.strip().upper() in [v.upper() for v in variants]), None)
                    if found: m[found] = exp
                    else: missing.append(exp)
                return df.rename(columns=m), missing

            self.claims, miss = remap(df, CLAIMS_MAP)
            if miss:
                return False, f"Missing columns: {', '.join(miss)}"
            self.tariff, tmiss = remap(tdf, TARIFF_MAP)
            if tmiss:
                return False, f"Missing tariff columns: {', '.join(tmiss)}"

            self.claims["CLM CODE"] = self.claims["CLM CODE"].astype(str).str.strip()
            self.tariff["CLM CODE"] = self.tariff["CLM CODE"].astype(str).str.strip()
            self.claims["TOTAL PAID"] = pd.to_numeric(self.claims["TOTAL PAID"], errors="coerce").fillna(0)
            return True, "OK"
        except Exception as e:
            return False, str(e)

    # ── Detect ────────────────────────────────
    def detect(self, name="Analysis"):
        self.flagged = self.claims.copy()
        self.flagged["FWA_Flag"]    = False
        self.flagged["Flag_Reason"] = ""
        self.flagged["Flag_Type"]   = ""

        pb = st.progress(0, text="Running gender checks…")
        self._gender_from_tariff(); pb.progress(15)
        self._gender_hardcoded();   pb.progress(30)
        self._age_cosmetic();       pb.progress(40)
        self._unit_overdose();      pb.progress(55)
        self._duplicates();         pb.progress(70)
        self._age_chronic_meds();   pb.progress(80)

        pb.progress(90, text="Running analytics…")
        benford = self.analytics.benfords_law(self.flagged["TOTAL PAID"])
        if benford:
            st.session_state["benford"] = benford
        pr = self.analytics.provider_risk(self.claims, self.flagged)
        if pr is not None:
            st.session_state["provider_risk"] = pr
            for p in pr[pr["risk_level"] == "High"].index:
                for idx in self.flagged[self.flagged["PROVIDER NAME"] == p].index:
                    self._flag(idx, f"High-risk provider: {p} (score {pr.loc[p,'risk_score']:.2f})", "F")

        pb.progress(100, text="Done ✓")
        time.sleep(.3); pb.empty()

    # ── Checks ────────────────────────────────
    def _gender_from_tariff(self):
        lkp = self.tariff.set_index("CLM CODE")["AllowedGender"].to_dict()
        for idx, row in self.flagged.iterrows():
            ag = str(lkp.get(row["CLM CODE"], "B")).strip().upper()
            if ag == "B": continue
            if ag == "F" and row["Gender"] != "Female":
                self._flag(idx, f"Gender mismatch: Male on female-only code {row['CLM CODE']}", "F")
            elif ag == "M" and row["Gender"] != "Male":
                self._flag(idx, f"Gender mismatch: Female on male-only code {row['CLM CODE']}", "F")

    def _gender_hardcoded(self):
        for idx, row in self.flagged.iterrows():
            code = str(row["CLM CODE"]).lstrip("0")
            if row["Gender"] == "Male" and code in self.FEMALE_CODES:
                self._flag(idx, f"Male claiming female-only procedure {row['CLM CODE']}", "F")
            elif row["Gender"] == "Female" and code in self.MALE_CODES:
                self._flag(idx, f"Female claiming male-only procedure {row['CLM CODE']}", "F")

    def _age_cosmetic(self):
        cosmetic_u18 = {"15831","15832","15833"}
        for idx, row in self.flagged.iterrows():
            try:
                age = float(row["CURRENT AGE"])
                code = str(row["CLM CODE"])
                if age < 18 and code in cosmetic_u18:
                    self._flag(idx, "Age <18 claiming cosmetic procedure", "F")
                elif age > 65 and code == "15775":
                    self._flag(idx, "Questionable cosmetic procedure for age >65", "W")
            except Exception:
                pass

    def _unit_overdose(self):
        lkp = self.tariff.set_index("CLM CODE")["MaxUnits"].to_dict()
        for idx, row in self.flagged.iterrows():
            mx = lkp.get(row["CLM CODE"])
            try:
                if pd.notna(mx) and float(row["UNITS"]) > float(mx):
                    self._flag(idx, f"Units {row['UNITS']} > max {mx}", "W")
            except Exception:
                pass

    def _duplicates(self):
        cols = ["CLAIM NO","CLAIM LINE NO"] if "CLAIM LINE NO" in self.flagged.columns else ["CLAIM NO"]
        dupes = self.flagged.duplicated(subset=cols, keep=False)
        for idx in self.flagged[dupes].index:
            self._flag(idx, "Exact duplicate claim", "F")

    def _age_chronic_meds(self):
        if "BASE BENEFIT DESCRIPTION" not in self.flagged.columns:
            return
        chronic = self.flagged["BASE BENEFIT DESCRIPTION"].astype(str).str.contains(
            "chronic|maintenance|long-term", case=False, na=False)
        for idx, row in self.flagged[chronic].iterrows():
            try:
                if float(row["CURRENT AGE"]) < 17:
                    self._flag(idx, f"Age {row['CURRENT AGE']} (<17) claiming chronic medication", "F")
            except Exception:
                pass

    def _flag(self, idx, reason, ftype):
        self.flagged.at[idx, "FWA_Flag"] = True
        existing = self.flagged.at[idx, "Flag_Reason"]
        self.flagged.at[idx, "Flag_Reason"] = (existing + "; " + reason) if existing else reason
        existing_t = self.flagged.at[idx, "Flag_Type"]
        if ftype not in str(existing_t):
            self.flagged.at[idx, "Flag_Type"] = (existing_t + "," + ftype) if existing_t else ftype


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
def bar_h(x, y, title, color=ACCENT, height=380):
    fig = go.Figure(go.Bar(x=x, y=y, orientation="h",
                           marker=dict(color=color, opacity=.85)))
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=height)
    fig.update_yaxes(tickfont_size=10)
    return fig


def pie_chart(labels, values, title, colors=None, hole=.45):
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=hole,
                           marker=dict(colors=colors or COLORS),
                           textinfo="label+percent",
                           hoverinfo="label+value+percent"))
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=360)
    return fig


def line_chart(x, y, title, color=ACCENT):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers",
                               line=dict(color=color, width=2),
                               marker=dict(size=4, color=color),
                               fill="tozeroy",
                               fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},.08)"))
    fig.update_layout(**PLOTLY_LAYOUT, title=title)
    return fig


def kpi(val, lbl, variant=""):
    return f'<div class="kpi {variant}"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>'


def alert(msg, kind="info"):
    return f'<div class="alert {kind}">{msg}</div>'


def sec(title):
    st.markdown(f'<div class="sec-head">{title}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DASHBOARD SECTIONS
# ─────────────────────────────────────────────
def show_executive(fd):
    fl = fd[fd["FWA_Flag"]]
    total, tflag = len(fd), len(fl)
    total_paid  = fd["TOTAL PAID"].sum()
    flag_paid   = fl["TOTAL PAID"].sum()
    fraud_n = fl["Flag_Type"].str.contains("F",na=False).sum()
    waste_n = fl["Flag_Type"].str.contains("W",na=False).sum()
    abuse_n = fl["Flag_Type"].str.contains("A",na=False).sum()
    dup_n   = fl["Flag_Reason"].str.contains("duplicate",na=False,case=False).sum()

    # ── KPI row ──
    st.markdown(f"""
    <div class="kpi-grid">
        {kpi(f"{total:,}", "Total Claims")}
        {kpi(f"{tflag:,}", "Flagged Claims", "danger")}
        {kpi(f"{tflag/total*100:.1f}%", "Flag Rate", "warn")}
        {kpi(f"${flag_paid:,.0f}", "Flagged Amt", "danger")}
        {kpi(f"${total_paid:,.0f}", "Total Paid")}
        {kpi(f"{fraud_n:,}", "Fraud (F)", "danger")}
        {kpi(f"{waste_n:,}", "Waste (W)", "warn")}
        {kpi(f"{dup_n:,}", "Duplicates", "blue")}
    </div>
    """, unsafe_allow_html=True)

    # ── Alerts ──
    chronic_age = fl["Flag_Reason"].str.contains("<17.*chronic|chronic.*<17", na=False, case=False).sum()
    if chronic_age:
        st.markdown(alert(f"⚠️ <b>{chronic_age}</b> chronic medication claims from patients under 17", "danger"), unsafe_allow_html=True)
    if dup_n:
        st.markdown(alert(f"🔁 <b>{dup_n}</b> exact duplicate claims detected", "warn"), unsafe_allow_html=True)

    # ── Charts ──
    col1, col2, col3 = st.columns(3)

    with col1:
        sec("FWA Category Split")
        st.plotly_chart(pie_chart(
            ["Fraud","Waste","Abuse"],
            [fraud_n, waste_n, abuse_n],
            "",
            [DANGER, WARN, ACCENT]
        ), use_container_width=True)

    with col2:
        sec("Gender in Flagged")
        if "Gender" in fl.columns:
            gc = fl["Gender"].value_counts()
            st.plotly_chart(pie_chart(gc.index.tolist(), gc.values.tolist(), "",
                                      [ACCENT2, ACCENT, WARN]), use_container_width=True)

    with col3:
        sec("Top 8 Flagged Providers")
        if "PROVIDER NAME" in fl.columns:
            tp = fl["PROVIDER NAME"].value_counts().head(8)
            st.plotly_chart(bar_h(tp.values, tp.index.tolist(), "", DANGER, 360), use_container_width=True)

    # ── Daily trend ──
    sec("Daily Claims Trend")
    if "SERVICE DATE" in fd.columns:
        fd2 = fd.copy()
        fd2["_dt"] = pd.to_datetime(fd2["SERVICE DATE"], errors="coerce")
        daily_all = fd2.groupby(fd2["_dt"].dt.date).size().rename("All")
        daily_fl  = fd2[fd2["FWA_Flag"]].groupby(fd2[fd2["FWA_Flag"]]["_dt"].dt.date).size().rename("Flagged")
        merged = pd.concat([daily_all, daily_fl], axis=1).fillna(0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged.index, y=merged["All"], name="All Claims",
                                 line=dict(color=ACCENT2, width=2), fill="tozeroy",
                                 fillcolor="rgba(61,158,255,.06)"))
        fig.add_trace(go.Scatter(x=merged.index, y=merged["Flagged"], name="Flagged",
                                 line=dict(color=DANGER, width=2)))
        fig.update_layout(**PLOTLY_LAYOUT, height=280, title="")
        st.plotly_chart(fig, use_container_width=True)


def show_analytics(fd):
    a = Analytics()
    t1, t2, t3, t4, t5 = st.tabs([
        "📊 Distributions", "📈 Temporal", "🏥 Providers", "📉 Benford's Law", "🔍 Outliers"
    ])

    with t1:
        sec("Amount Distributions")
        col1, col2 = st.columns(2)
        with col1:
            amounts = pd.to_numeric(fd["TOTAL PAID"], errors="coerce").dropna()
            fig = go.Figure(go.Histogram(x=amounts, nbinsx=60,
                                         marker=dict(color=ACCENT, opacity=.75),
                                         name="Paid Amounts"))
            fig.update_layout(**PLOTLY_LAYOUT, title="Distribution of Total Paid", height=340)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure(go.Box(y=amounts, boxpoints="outliers", name="",
                                    marker=dict(color=ACCENT2, size=3),
                                    line=dict(color=ACCENT2)))
            fig2.update_layout(**PLOTLY_LAYOUT, title="Box Plot – Paid Amounts", height=340)
            st.plotly_chart(fig2, use_container_width=True)

        sec("Summary Statistics")
        stats_data = {
            "Count": f"{len(amounts):,}",
            "Mean":  f"${amounts.mean():,.2f}",
            "Median":f"${amounts.median():,.2f}",
            "Std Dev":f"${amounts.std():,.2f}",
            "Min":   f"${amounts.min():,.2f}",
            "Max":   f"${amounts.max():,.2f}",
            "Skewness": f"{amounts.skew():.4f}",
            "Kurtosis": f"{amounts.kurtosis():.4f}",
            "CoV": f"{amounts.std()/amounts.mean():.3f}",
        }
        cols = st.columns(len(stats_data))
        for col, (k, v) in zip(cols, stats_data.items()):
            col.metric(k, v)

    with t2:
        sec("Temporal Patterns")
        if "SERVICE DATE" not in fd.columns:
            st.info("No SERVICE DATE column."); return
        fd2 = fd.copy()
        fd2["_dt"] = pd.to_datetime(fd2["SERVICE DATE"], errors="coerce")
        fd2 = fd2.dropna(subset=["_dt"])

        col1, col2 = st.columns(2)
        with col1:
            wk = fd2["_dt"].dt.day_name().value_counts().reindex(
                ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).fillna(0)
            fig = go.Figure(go.Bar(x=wk.index, y=wk.values,
                                   marker=dict(color=ACCENT, opacity=.8)))
            fig.update_layout(**PLOTLY_LAYOUT, title="Claims by Day of Week", height=320)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            mo = fd2["_dt"].dt.month_name().value_counts()
            fig2 = go.Figure(go.Bar(x=mo.index, y=mo.values,
                                    marker=dict(color=ACCENT2, opacity=.8)))
            fig2.update_layout(**PLOTLY_LAYOUT, title="Claims by Month", height=320)
            st.plotly_chart(fig2, use_container_width=True)

        # Flag type over time
        sec("Flagged Claims Timeline")
        fl = fd2[fd2["FWA_Flag"]]
        if len(fl):
            monthly = fl.groupby(fl["_dt"].dt.to_period("M")).size()
            fig3 = line_chart([str(p) for p in monthly.index], monthly.values, "Monthly Flagged Claims")
            st.plotly_chart(fig3, use_container_width=True)

    with t3:
        sec("Provider Risk Overview")
        if "provider_risk" in st.session_state:
            pr = st.session_state["provider_risk"]

            col1, col2 = st.columns([2, 1])
            with col1:
                top = pr.head(15)
                fig = go.Figure()
                colors_risk = [DANGER if l=="High" else WARN if l=="Medium" else ACCENT
                               for l in top["risk_level"].astype(str)]
                fig.add_trace(go.Bar(x=top["risk_score"], y=top.index, orientation="h",
                                     marker=dict(color=colors_risk, opacity=.85),
                                     text=[f"{s:.2f}" for s in top["risk_score"]],
                                     textposition="inside"))
                fig.update_layout(**PLOTLY_LAYOUT, title="Provider Risk Scores (Top 15)", height=460)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                risk_dist = pr["risk_level"].value_counts()
                st.plotly_chart(pie_chart(
                    risk_dist.index.tolist(), risk_dist.values.tolist(), "Risk Distribution",
                    [DANGER, WARN, ACCENT]
                ), use_container_width=True)

                sec("Top 5 by Flagged Amount")
                top5 = pr.nlargest(5, "flagged_amount")[["claim_count","flagged_amount","flag_rate"]]
                top5["flag_rate"] = top5["flag_rate"].map(lambda x: f"{x:.1f}%")
                top5["flagged_amount"] = top5["flagged_amount"].map(lambda x: f"${x:,.0f}")
                st.dataframe(top5, use_container_width=True)
        else:
            st.info("Provider risk data not yet available. Run the analysis first.")

    with t4:
        sec("Benford's Law – First Digit Analysis")
        if "benford" not in st.session_state:
            bd = a.benfords_law(fd["TOTAL PAID"])
            if bd: st.session_state["benford"] = bd

        if "benford" in st.session_state:
            bd = st.session_state["benford"]
            digits = list(range(1, 10))
            obs    = [bd["observed"].get(d, 0) for d in digits]
            exp    = [bd["expected"].get(d, 0) for d in digits]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=digits, y=obs, name="Observed",
                                 marker=dict(color=ACCENT2, opacity=.8)))
            fig.add_trace(go.Scatter(x=digits, y=exp, name="Benford Expected",
                                     mode="lines+markers",
                                     line=dict(color=DANGER, width=2.5, dash="dot"),
                                     marker=dict(size=6)))
            fig.update_layout(**PLOTLY_LAYOUT, title="First Digit Distribution vs Benford's Law",
                              xaxis_title="First Digit", yaxis_title="Frequency (%)", height=380)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Chi-Square", f"{bd['chi_square']:.4f}")
            col2.metric("P-Value", f"{bd['p_value']:.6f}")
            if bd["p_value"] < 0.05:
                col3.metric("Verdict", "⚠️ Anomaly")
                st.markdown(alert("P-value < 0.05: distribution deviates from Benford's Law — potential data manipulation or systematic fraud.", "danger"), unsafe_allow_html=True)
            else:
                col3.metric("Verdict", "✅ Normal")
                st.markdown(alert("P-value ≥ 0.05: distribution follows Benford's Law — data appears natural.", "ok"), unsafe_allow_html=True)

    with t5:
        sec("Statistical Outlier Detection")
        col1, col2, col3 = st.columns([2, 2, 1])
        avail_cols = [c for c in ["TOTAL PAID","UNITS"] if c in fd.columns]
        method_map = {"IQR": "iqr", "Z-Score": "zscore", "Isolation Forest": "isolation_forest"}
        with col1: method = st.selectbox("Method", list(method_map.keys()))
        with col2: col_sel = st.selectbox("Column", avail_cols)
        with col3:
            st.write("")
            run = st.button("Run", type="primary")

        if run:
            with st.spinner("Detecting outliers…"):
                res = a.detect_outliers(fd, col_sel, method_map[method])
            if res:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Outliers Found", f"{res['count']:,}")
                c2.metric("Outlier %", f"{res['pct']:.2f}%")
                c3.metric("Mean", f"${res['mean']:,.2f}" if col_sel=="TOTAL PAID" else f"{res['mean']:.2f}")
                c4.metric("Median", f"${res['median']:,.2f}" if col_sel=="TOTAL PAID" else f"{res['median']:.2f}")

                vals = pd.to_numeric(fd[col_sel], errors="coerce").dropna()
                is_out = vals.index.isin(res["outliers"].index)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=vals[~is_out].index, y=vals[~is_out],
                                         mode="markers", name="Normal",
                                         marker=dict(color=ACCENT, size=4, opacity=.5)))
                fig.add_trace(go.Scatter(x=vals[is_out].index, y=vals[is_out],
                                         mode="markers", name="Outlier",
                                         marker=dict(color=DANGER, size=6, opacity=.8, symbol="x")))
                fig.update_layout(**PLOTLY_LAYOUT, title=f"Outlier View – {col_sel}", height=340)
                st.plotly_chart(fig, use_container_width=True)


def show_detail(fd):
    sec("Filter & Explore")
    col1, col2, col3, col4 = st.columns(4)
    with col1: only_fl = st.checkbox("Flagged only", True)
    with col2: ft = st.selectbox("Flag type", ["All","F – Fraud","W – Waste","A – Abuse"])
    with col3: prov = st.selectbox("Provider", ["All"] + sorted(fd["PROVIDER NAME"].dropna().unique().tolist()))
    with col4:
        issues = ["All","Gender Issues","Duplicates","Unit Overdose","Age–Chronic Meds"]
        issue  = st.selectbox("Issue type", issues)

    d = fd.copy()
    if only_fl: d = d[d["FWA_Flag"]]
    if ft != "All": d = d[d["Flag_Type"].str.contains(ft[0], na=False)]
    if prov != "All": d = d[d["PROVIDER NAME"] == prov]
    if issue == "Gender Issues":       d = d[d["Flag_Reason"].str.contains("Gender|gender|Male|Female", na=False)]
    elif issue == "Duplicates":        d = d[d["Flag_Reason"].str.contains("duplicate", na=False, case=False)]
    elif issue == "Unit Overdose":     d = d[d["Flag_Reason"].str.contains("Units", na=False)]
    elif issue == "Age–Chronic Meds":  d = d[d["Flag_Reason"].str.contains("<17", na=False)]

    st.caption(f"{len(d):,} records shown")
    st.dataframe(d, use_container_width=True, hide_index=True, height=480,
                 column_config={"FWA_Flag": st.column_config.CheckboxColumn("Flagged")})


def show_export(fd):
    sec("Download Results")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Download Full Analysis (Excel)", type="primary"):
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                fd.to_excel(w, index=False, sheet_name="All_Claims")
                fd[fd["FWA_Flag"]].to_excel(w, index=False, sheet_name="Flagged_Claims")
                # summary sheet
                fl = fd[fd["FWA_Flag"]]
                summ = pd.DataFrame({
                    "Metric": ["Total Claims","Flagged","Flag %","Total Paid","Flagged Paid",
                               "Fraud (F)","Waste (W)","Abuse (A)","Duplicates"],
                    "Value": [
                        len(fd), len(fl), f"{len(fl)/len(fd)*100:.1f}%",
                        f"${fd['TOTAL PAID'].sum():,.2f}", f"${fl['TOTAL PAID'].sum():,.2f}",
                        fl["Flag_Type"].str.contains("F",na=False).sum(),
                        fl["Flag_Type"].str.contains("W",na=False).sum(),
                        fl["Flag_Type"].str.contains("A",na=False).sum(),
                        fl["Flag_Reason"].str.contains("duplicate",na=False,case=False).sum(),
                    ]
                })
                summ.to_excel(w, index=False, sheet_name="Summary")
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(
                f'<a href="data:application/octet-stream;base64,{b64}" download="FWA_Analysis.xlsx">'
                f'Click to download</a>', unsafe_allow_html=True)
    with col2:
        if st.button("📥 Flagged Only (Excel)"):
            buf2 = BytesIO()
            with pd.ExcelWriter(buf2, engine="openpyxl") as w:
                fd[fd["FWA_Flag"]].to_excel(w, index=False, sheet_name="Flagged")
            b64 = base64.b64encode(buf2.getvalue()).decode()
            st.markdown(
                f'<a href="data:application/octet-stream;base64,{b64}" download="Flagged_Claims.xlsx">'
                f'Click to download</a>', unsafe_allow_html=True)


def show_insights(fd):
    fl = fd[fd["FWA_Flag"]]
    sec("Key Findings")
    insights = []

    if "PROVIDER NAME" in fl.columns:
        tp = fl["PROVIDER NAME"].value_counts().head(3)
        insights.append(("🏥 Top flagged providers",
                          " | ".join(f"<b>{p}</b> ({c})" for p,c in tp.items())))

    if "TOTAL PAID" in fl.columns:
        pct = fl["TOTAL PAID"].sum() / fd["TOTAL PAID"].sum() * 100
        insights.append(("💰 Financial exposure",
                          f"<b>${fl['TOTAL PAID'].sum():,.2f}</b> flagged — {pct:.1f}% of total spend"))

    dup = fl["Flag_Reason"].str.contains("duplicate",na=False,case=False).sum()
    if dup: insights.append(("🔁 Duplicates", f"<b>{dup}</b> exact duplicate claims"))

    ca = fl["Flag_Reason"].str.contains("<17",na=False).sum()
    if ca: insights.append(("👶 Age compliance", f"<b>{ca}</b> chronic medication claims from patients under 17"))

    if "provider_risk" in st.session_state:
        pr = st.session_state["provider_risk"]
        h = (pr["risk_level"] == "High").sum()
        m = (pr["risk_level"] == "Medium").sum()
        insights.append(("🎯 Provider risk", f"<b>{h}</b> high-risk, <b>{m}</b> medium-risk providers"))

    for title, body in insights:
        st.markdown(f'<div class="insight"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)

    sec("Recommendations")
    recs = [
        "Review all <b>Fraud (F)</b> flagged claims immediately for recovery action.",
        "Resolve <b>exact duplicate claims</b> through automated pre-submission validation.",
        "Provide <b>gender-appropriateness training</b> to high-frequency flagging providers.",
        "Audit <b>chronic medication claims</b> where patient age is under 17.",
        "Establish <b>continuous monitoring</b> for providers with risk score above 2.0.",
        "Investigate claims where Benford's Law deviation is statistically significant.",
    ]
    for r in recs:
        st.markdown(f'<div class="insight">• {r}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("### 🛡️ Gen-Health FWA")
        st.caption("Fraud · Waste · Abuse Detection")
        st.divider()

        st.markdown("**1. Tariff File**")
        st.markdown('<div class="upload-hint">Excel with "Sample Tariff" sheet</div>', unsafe_allow_html=True)
        tariff_file = st.file_uploader("", type=["xlsx"], key="tariff_up", label_visibility="collapsed")

        st.markdown("**2. Claims File**")
        st.markdown('<div class="upload-hint">Excel with "Sheet1"</div>', unsafe_allow_html=True)
        claims_file = st.file_uploader("", type=["xlsx"], key="claims_up", label_visibility="collapsed")

        analysis_name = st.text_input("Analysis label",
                                      value="FWA " + datetime.now().strftime("%Y-%m-%d"))
        st.divider()

        run = st.button("🚀 Run Analysis", type="primary", use_container_width=True,
                        disabled=not (tariff_file and claims_file))

        if run and tariff_file and claims_file:
            det = FWADetector()
            with st.spinner("Loading data…"):
                ok, msg = det.load_data(claims_file, tariff_file)
            if ok:
                det.detect(analysis_name)
                st.session_state["detector"] = det
                st.session_state["analysis_name"] = analysis_name
                st.success("Analysis complete ✓")
                st.rerun()
            else:
                st.error(msg)

        if "detector" in st.session_state:
            det = st.session_state["detector"]
            fl = det.flagged[det.flagged["FWA_Flag"]]
            st.divider()
            st.caption("Last run summary")
            st.markdown(f"**Claims:** {len(det.flagged):,}")
            st.markdown(f"**Flagged:** {len(fl):,}  ({len(fl)/len(det.flagged)*100:.1f}%)")
            st.markdown(f"**Exposure:** ${fl['TOTAL PAID'].sum():,.2f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    sidebar()

    if "detector" not in st.session_state:
        # Welcome screen
        st.markdown("""
        <div style="text-align:center; padding: 4rem 1rem 2rem;">
            <div style="font-size:3rem; margin-bottom:.5rem;">🛡️</div>
            <h1 style="font-weight:700; color:#e6edf3; font-size:2.2rem; margin:0;">Gen-Health FWA Detection</h1>
            <p style="color:#8b949e; font-size:1.05rem; margin-top:.5rem; max-width:560px; margin-inline:auto;">
                Upload your tariff and claims files in the sidebar to begin actuarial fraud, waste & abuse analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        for col, icon, title, body in [
            (col1,"📊","Statistical Analysis","Outlier detection, Benford's law, distribution profiling"),
            (col2,"🎯","Provider Scoring","Composite risk scores, flag rates, financial exposure"),
            (col3,"⚧","Gender & Age Checks","Procedure-level gender rules + chronic medication age gates"),
            (col4,"🔁","Duplicate Detection","Exact duplicate claims by claim & line number"),
            (col5,"⚖️","IBNR Reserves","Chain ladder estimation of incurred-but-not-reported reserves"),
        ]:
            col.markdown(f"""
            <div class="insight" style="text-align:center;">
                <div style="font-size:1.6rem">{icon}</div>
                <div style="font-weight:600;margin:.4rem 0 .2rem;color:#e6edf3">{title}</div>
                <div style="color:#8b949e;font-size:.82rem">{body}</div>
            </div>""", unsafe_allow_html=True)
        st.divider()
        st.markdown("**Run IBNR analysis independently:**")
        show_ibnr()
        return

    det  = st.session_state["detector"]
    fd   = det.flagged
    name = st.session_state.get("analysis_name","Analysis")

    st.markdown(f"<h2 style='font-weight:700;color:#e6edf3;margin-bottom:.1rem;'>{name}</h2>", unsafe_allow_html=True)
    st.caption(f"Analyzed {datetime.now().strftime('%d %b %Y, %H:%M')}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Dashboard", "📈 Analytics", "🔍 Detail View", "📥 Export", "💡 Insights", "⚖️ IBNR Reserves"
    ])

    with tab1: show_executive(fd)
    with tab2: show_analytics(fd)
    with tab3: show_detail(fd)
    with tab4: show_export(fd)
    with tab5: show_insights(fd)
    with tab6: show_ibnr()


if __name__ == "__main__":
    main()
