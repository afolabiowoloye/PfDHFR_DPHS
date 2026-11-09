import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="DHFR/DHPS Prevalence Explorer", layout="wide")

try:
    from scipy.stats import chi2
except ImportError:
    chi2 = None

BASE_DIR = Path(__file__).resolve().parent
DATA_URL = "https://github.com/afolabiowoloye/PfDHFR_DPHS/raw/main/DHFR_DHPS_analysis.xlsx"
LOGO_PATH = BASE_DIR / "synergy_logo.png"
SHEET_NAME = "Sheet1"
DHFR_SNPS = ["N51I", "C59R", "S108N", "I164L"]
DHPS_SNPS = ["I431V", "S436A", "A437G", "K540E", "A581G", "A613S"]
DHFR_HAPLOTYPES = ["Double_DHFR", "Triple_DHFR"]
DHPS_HAPLOTYPES = ["Double_DHPS", "Triple_DHPS", "Quad_DHPS", "Quint_DHPS"]

COUNTRY_CODE_MAP = {
    "angola": "AGO",
    "bangladesh": "BGD",
    "benin": "BEN",
    "burkina faso": "BFA",
    "cambodia": "KHM",
    "cameroon": "CMR",
    "central african republic": "CAF",
    "chad": "TCD",
    "china": "CHN",
    "colombia": "COL",
    "congo brazzaville": "COG",
    "republic of the congo": "COG",
    "cote d'ivoire": "CIV",
    "côte d'ivoire": "CIV",
    "democratic republic of the congo": "COD",
    "dr congo": "COD",
    "ecuador": "ECU",
    "equatorial guinea": "GNQ",
    "ethiopia": "ETH",
    "gabon": "GAB",
    "gambia": "GMB",
    "the gambia": "GMB",
    "ghana": "GHA",
    "guinea": "GIN",
    "guinea-bissau": "GNB",
    "haiti": "HTI",
    "india": "IND",
    "indonesia": "IDN",
    "iran": "IRN",
    "kenya": "KEN",
    "lao pdr": "LAO",
    "laos": "LAO",
    "liberia": "LBR",
    "madagascar": "MDG",
    "malawi": "MWI",
    "mali": "MLI",
    "mauritania": "MRT",
    "mozambique": "MOZ",
    "myanmar": "MMR",
    "china-myanmar": "MMR",
    "namibia": "NAM",
    "niger": "NER",
    "nigeria": "NGA",
    "pakistan": "PAK",
    "papua new guinea": "PNG",
    "peru": "PER",
    "philippines": "PHL",
    "qatar": "QAT",
    "rwanda": "RWA",
    "sao tome and principe": "STP",
    "sao tome and prinicpe": "STP",
    "são tomé and príncipe": "STP",
    "sao tome and prncipe": "STP",
    "saudi arabia": "SAU",
    "senegal": "SEN",
    "sierra leone": "SLE",
    "south africa": "ZAF",
    "south sudan": "SSD",
    "sudan": "SDN",
    "suriname": "SUR",
    "tanzania": "TZA",
    "thailand": "THA",
    "togo": "TGO",
    "uganda": "UGA",
    "venezuela": "VEN",
    "yemen": "YEM",
    "zambia": "ZMB",
    "zimbabwe": "ZWE",
}


def normalize_country_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    variants = {
        "–": "-",
        "—": "-",
        "−": "-",
        "C�te d�Ivoire": "Côte d'Ivoire",
        "S�o Tom� and Pr�ncipe": "São Tomé and Príncipe",
        "S�o Tom� and Pr�ncipe": "São Tomé and Príncipe",
    }
    for old, new in variants.items():
        text = text.replace(old, new)
    return text.strip()


def country_to_iso3(value: object) -> object:
    text = normalize_country_name(value)
    if not text:
        return np.nan
    key = text.casefold()
    key = key.replace("  ", " ")
    if key in COUNTRY_CODE_MAP:
        return COUNTRY_CODE_MAP[key]
    alias_map = {
        "cote d'ivoire": "CIV",
        "côte d'ivoire": "CIV",
        "sao tome and principe": "STP",
        "são tomé and príncipe": "STP",
        "sao tome and prinicpe": "STP",
        "congo brazzaville": "COG",
        "china-myanmar": "MMR",
    }
    return alias_map.get(key, np.nan)


@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_excel(url, sheet_name=SHEET_NAME)

    for col in DHFR_SNPS + DHPS_SNPS + DHFR_HAPLOTYPES + DHPS_HAPLOTYPES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def mid_year(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            # Normalize dash variants and extract 4-digit years from mixed text.
            cleaned = x.strip().replace("–", "-").replace("—", "-").replace("−", "-")
            years = re.findall(r"\d{4}", cleaned)
            if len(years) >= 2:
                return float(np.mean([float(years[0]), float(years[1])]))
            if len(years) == 1:
                return float(years[0])
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if pd.notna(numeric):
                return float(numeric)
            return np.nan
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    df["mid_sampling_year"] = df["Yr of Sampling"].apply(mid_year)
    df["DHFR_denominator"] = df["Sample genotyped DHFR"].fillna(df["Sample size"])
    df["DHPS_denominator"] = df["Sample genotyped DHPS"].fillna(df["Sample size"])
    return df


@st.cache_data(show_spinner=False)
def get_snp_options(df: pd.DataFrame, gene: str) -> list[str]:
    if gene == "DHFR":
        return DHFR_SNPS
    return DHPS_SNPS


@st.cache_data(show_spinner=False)
def get_haplotype_options(df: pd.DataFrame, gene: str) -> list[str]:
    if gene == "DHFR":
        return DHFR_HAPLOTYPES
    return DHPS_HAPLOTYPES


def get_denominator_col(marker: str) -> str:
    if marker.endswith("_DHFR") or marker in DHFR_SNPS:
        return "DHFR_denominator"
    return "DHPS_denominator"


@st.cache_data(show_spinner=False)
def get_country_options(df: pd.DataFrame, marker: str) -> list[str]:
    return get_geographic_options(df, marker, "Country")


@st.cache_data(show_spinner=False)
def get_geographic_options(df: pd.DataFrame, marker: str, geographic_column: str) -> list[str]:
    denom_col = get_denominator_col(marker)
    mask = df[denom_col].notna() & df[marker].notna() & df[geographic_column].notna()
    available = df.loc[mask, geographic_column].dropna().astype(str).unique().tolist()

    preferred_order = {
        "Region": ["Africa", "SEA", "EMR", "WPR", "South America"],
        "Africa_subregion": [
            "West Africa",
            "East Africa",
            "Central Africa",
            "Southern Africa",
            "Northern Africa",
        ],
    }
    if geographic_column in preferred_order:
        return preferred_order[geographic_column]
    return sorted(available)


@st.cache_data(show_spinner=False)
def get_year_options(df: pd.DataFrame) -> list[int]:
    years = pd.to_numeric(df["mid_sampling_year"], errors="coerce").dropna().round().astype(int)
    return sorted(pd.Series(years.unique()).tolist())


@st.cache_data(show_spinner=False)
def build_country_trend(
    df: pd.DataFrame,
    marker: str,
    location: str,
    geographic_column: str = "Country",
) -> pd.DataFrame:
    denom_col = get_denominator_col(marker)
    subset = df[
        (df[geographic_column].astype(str) == location)
        & df[marker].notna()
        & df[denom_col].notna()
    ].copy()
    if subset.empty:
        return pd.DataFrame(columns=["mid_sampling_year", "prevalence", "count", "denominator", "ci_lower", "ci_upper", "Author", "Year", "study_id"])

    subset["prevalence"] = subset[marker] / subset[denom_col]
    subset["count"] = pd.to_numeric(subset[marker], errors="coerce")
    subset["denominator"] = pd.to_numeric(subset[denom_col], errors="coerce")

    z = 1.96
    n = subset["denominator"].to_numpy(dtype=float)
    p = subset["prevalence"].to_numpy(dtype=float)
    valid = n > 0

    ci_lower = np.full(len(subset), np.nan, dtype=float)
    ci_upper = np.full(len(subset), np.nan, dtype=float)
    if np.any(valid):
        n_v = n[valid]
        p_v = p[valid]
        denom = 1.0 + (z**2 / n_v)
        center = (p_v + (z**2 / (2.0 * n_v))) / denom
        margin = (z / denom) * np.sqrt((p_v * (1.0 - p_v) / n_v) + (z**2 / (4.0 * (n_v**2))))
        ci_lower[valid] = np.clip(center - margin, 0, 1)
        ci_upper[valid] = np.clip(center + margin, 0, 1)

    subset["ci_lower"] = ci_lower
    subset["ci_upper"] = ci_upper

    author_col = next((c for c in ["Author detail"] if c in df.columns), None)
    year_col = next((c for c in ["Yr of Pub", "Yr of Sampling", "Year"] if c in df.columns), None)

    subset["Author"] = subset[author_col].fillna("").astype(str) if author_col else ""
    subset["Year"] = subset[year_col] if year_col else np.nan

    subset = subset[["mid_sampling_year", "prevalence", "count", "denominator", "ci_lower", "ci_upper", "Author", "Year"]].dropna(subset=["mid_sampling_year", "prevalence"])
    subset = subset.sort_values("mid_sampling_year")
    subset["study_id"] = subset.index.astype(str)
    return subset


@st.cache_data(show_spinner=False)
def build_choropleth_data(df: pd.DataFrame, marker: str, year: int) -> pd.DataFrame:
    denom_col = get_denominator_col(marker)
    subset = df[(df["mid_sampling_year"].astype(float) == float(year)) & df[marker].notna() & df[denom_col].notna()].copy()
    if subset.empty:
        return pd.DataFrame(columns=["Country", "prevalence", "count", "denominator"])

    country_summary = (
        subset.groupby("Country", as_index=False)
        .agg(count=(marker, "sum"), denominator=(denom_col, "sum"))
    )
    country_summary["prevalence"] = country_summary["count"] / country_summary["denominator"]
    return country_summary


@st.cache_data(show_spinner=False)
def build_period_prevalence(df: pd.DataFrame, marker: str) -> pd.DataFrame:
    denom_col = get_denominator_col(marker)
    work_df = df[["mid_sampling_year", marker, denom_col, "Country"]].copy()
    work_df["mid_sampling_year"] = pd.to_numeric(work_df["mid_sampling_year"], errors="coerce")
    work_df = work_df.dropna(subset=["mid_sampling_year", marker, denom_col, "Country"])

    periods = [
        ("2000-2004", 2000, 2004),
        ("2005-2012", 2005, 2012),
        ("2013-2026", 2013, 2026),
    ]

    rows = []
    for label, start, end in periods:
        subset = work_df[(work_df["mid_sampling_year"].between(start, end, inclusive="both"))]
        if subset.empty:
            continue

        study_df = subset[[marker, denom_col]].copy()
        study_df[marker] = pd.to_numeric(study_df[marker], errors="coerce")
        study_df[denom_col] = pd.to_numeric(study_df[denom_col], errors="coerce")
        study_df = study_df.dropna()
        study_df = study_df[(study_df[denom_col] > 0) & (study_df[marker] >= 0) & (study_df[marker] <= study_df[denom_col])]
        if study_df.empty:
            continue

        x = study_df[marker].to_numpy(dtype=float)
        n = study_df[denom_col].to_numpy(dtype=float)
        k = len(study_df)

        total_count = float(np.sum(x))
        total_denom = float(np.sum(n))

        # Logit random-effects meta-analysis with continuity correction.
        x_cc = x + 0.5
        n_minus_x_cc = (n - x) + 0.5
        p_cc = x_cc / (n + 1.0)
        yi = np.log(p_cc / (1.0 - p_cc))
        vi = (1.0 / x_cc) + (1.0 / n_minus_x_cc)
        wi = 1.0 / vi

        y_fixed = np.sum(wi * yi) / np.sum(wi)
        q = float(np.sum(wi * ((yi - y_fixed) ** 2)))
        df_q = k - 1
        c_val = float(np.sum(wi) - (np.sum(wi**2) / np.sum(wi))) if k > 1 else np.nan

        tau2 = max(0.0, (q - df_q) / c_val) if (k > 1 and c_val > 0) else 0.0
        w_re = 1.0 / (vi + tau2)
        mu_re = np.sum(w_re * yi) / np.sum(w_re)
        se_re = math.sqrt(1.0 / np.sum(w_re))

        z_val = 1.96
        ci_low_logit = mu_re - z_val * se_re
        ci_high_logit = mu_re + z_val * se_re

        pooled_prevalence = 1.0 / (1.0 + np.exp(-mu_re))
        ci_lower = 1.0 / (1.0 + np.exp(-ci_low_logit))
        ci_upper = 1.0 / (1.0 + np.exp(-ci_high_logit))

        i2 = max(0.0, ((q - df_q) / q) * 100.0) if (k > 1 and q > 0) else 0.0
        p_q = float(chi2.sf(q, df_q)) if (chi2 is not None and df_q > 0) else np.nan

        rows.append(
            {
                "period": label,
                "pooled_prevalence": pooled_prevalence,
                "Period": label,
                "Studies (k)": int(k),
                "Total Mutants": int(round(total_count)),
                "Total Genotyped": int(round(total_denom)),
                "Pooled Prev (%)": pooled_prevalence * 100.0,
                "95% CI": f"{ci_lower * 100.0:.2f} - {ci_upper * 100.0:.2f}",
                "I2 (%)": i2,
                "Tau2": tau2,
                "p-value (Q)": p_q,
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_period_choropleth_data(df: pd.DataFrame, marker: str, selected_periods: list[str]) -> pd.DataFrame:
    denom_col = get_denominator_col(marker)
    work_df = df[["mid_sampling_year", marker, denom_col, "Country"]].copy()
    work_df["mid_sampling_year"] = pd.to_numeric(work_df["mid_sampling_year"], errors="coerce")
    work_df[marker] = pd.to_numeric(work_df[marker], errors="coerce")
    work_df[denom_col] = pd.to_numeric(work_df[denom_col], errors="coerce")
    work_df = work_df.dropna(subset=["mid_sampling_year", marker, denom_col, "Country"])

    period_bounds = {
        "2000-2004": (2000, 2004),
        "2005-2012": (2005, 2012),
        "2013-2026": (2013, 2026),
    }

    rows = []
    for period in selected_periods:
        start, end = period_bounds[period]
        subset = work_df[work_df["mid_sampling_year"].between(start, end, inclusive="both")]
        if subset.empty:
            continue
        country_summary = (
            subset.groupby("Country", as_index=False)
            .agg(count=(marker, "sum"), denominator=(denom_col, "sum"))
        )
        country_summary["prevalence"] = np.divide(
            country_summary["count"],
            country_summary["denominator"],
            out=np.full(len(country_summary), np.nan, dtype=float),
            where=(country_summary["denominator"] > 0),
        )
        country_summary = country_summary.dropna(subset=["prevalence"]).copy()
        country_summary["period"] = period
        rows.append(country_summary)

    if not rows:
        return pd.DataFrame(columns=["Country", "count", "denominator", "prevalence", "period"])

    return pd.concat(rows, ignore_index=True)


@st.cache_data(show_spinner=False)
def build_period_scatter_data(df: pd.DataFrame, marker: str, selected_periods: list[str]) -> pd.DataFrame:
    denom_col = get_denominator_col(marker)
    work_df = df[["mid_sampling_year", marker, denom_col, "Country", "Sample size"]].copy()
    work_df["mid_sampling_year"] = pd.to_numeric(work_df["mid_sampling_year"], errors="coerce")
    work_df[marker] = pd.to_numeric(work_df[marker], errors="coerce")
    work_df[denom_col] = pd.to_numeric(work_df[denom_col], errors="coerce")
    work_df["Sample size"] = pd.to_numeric(work_df["Sample size"], errors="coerce")
    work_df = work_df.dropna(subset=["mid_sampling_year", marker, denom_col, "Country"])

    work_df["prevalence"] = np.divide(
        work_df[marker],
        work_df[denom_col],
        out=np.full(len(work_df), np.nan, dtype=float),
        where=(work_df[denom_col] > 0),
    )
    work_df = work_df.dropna(subset=["prevalence"]).copy()
    work_df["prevalence"] = work_df["prevalence"].clip(lower=0, upper=1)

    sample_size = work_df["Sample size"]
    bins = [-1, 500, 1000, 5000, np.inf]
    labels = ["<500", "501-1000", "1001-5000", ">5000"]
    work_df["sample_size_bin"] = pd.cut(sample_size, bins=bins, labels=labels, include_lowest=True, right=True)
    work_df["sample_size_bin"] = work_df["sample_size_bin"].astype("string").fillna("Unknown")

    period_bounds = {
        "2000-2004": (2000, 2004),
        "2005-2012": (2005, 2012),
        "2013-2026": (2013, 2026),
    }

    rows = []
    for period in selected_periods:
        start, end = period_bounds[period]
        subset = work_df[work_df["mid_sampling_year"].between(start, end, inclusive="both")]
        if subset.empty:
            continue
        subset = subset[["mid_sampling_year", "Country", "prevalence", "sample_size_bin", "Sample size"]].copy()
        subset = subset.dropna(subset=["prevalence"])
        subset["period"] = period
        rows.append(subset)

    if not rows:
        return pd.DataFrame(columns=["mid_sampling_year", "Country", "prevalence", "sample_size_bin", "Sample size", "period"])

    return pd.concat(rows, ignore_index=True)


# Load and prepare data
df = load_data(DATA_URL)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3fa9f5 0%, #2475c8 100%);
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #fff8f0;
    }

    .app-hero {
        padding: 2rem 2.4rem 2.1rem;
        margin: 0 0 1.6rem;
        color: #ffffff;
        background: linear-gradient(115deg, #123b70 0%, #0b2a50 100%);
        border-top: 7px solid #f2c94c;
        border-bottom: 5px solid #b51e35;
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(13, 57, 48, 0.13);
    }

    .app-hero .eyebrow {
        margin: 0 0 0.7rem;
        color: #ffdf72;
        font-size: 0.95rem;
        font-weight: 750;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .app-hero h1 {
        margin: 0;
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.15;
        font-weight: 750;
    }

    .app-hero .hero-description {
        max-width: 850px;
        margin: 0.8rem 0 0;
        color: #edf4ff;
        font-size: 1.05rem;
        line-height: 1.65;
    }

    @media (max-width: 640px) {
        .app-hero {
            padding: 1.5rem;
            border-radius: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    if LOGO_PATH.is_file():
        st.image(str(LOGO_PATH), width=190)
    else:
        st.warning(f"Sidebar logo not found: {LOGO_PATH.name}")
    st.markdown("### DHFR / DHPS Explorer")
    st.caption("Global molecular surveillance")
    st.divider()
    analysis_group = st.selectbox(
        "Analysis area",
        [
            "Country-level analysis",
            "Africa subregion",
            "WHO region",
            "World analysis",
        ],
        key="analysis_group",
    )
    analysis_type = st.selectbox(
        analysis_group,
        ["SNP", "haplotype"],
        key="analysis_type",
    )

st.markdown(
    """
    <header class="app-hero">
      <p class="eyebrow">Malaria molecular surveillance · Global evidence</p>
      <h1><em>P. falciparum</em> DHFR/DHPS explorer</h1>
      <p class="hero-description">
        Explore country-level prevalence trends, pooled estimates, and global maps
        of antimalarial resistance markers.
      </p>
    </header>
    """,
    unsafe_allow_html=True,
)

view_family = "World choropleth" if analysis_group == "World analysis" else "Country trend"
view_mode = f"{view_family} ({analysis_type})"
geographic_column, geographic_label = {
    "Country-level analysis": ("Country", "country"),
    "Africa subregion": ("Africa_subregion", "Africa subregion"),
    "WHO region": ("Region", "region"),
}.get(analysis_group, ("Country", "country"))

if view_mode == "Country trend (SNP)":
    with st.sidebar:
        st.header(f"{geographic_label.title()} trend controls")
        gene = st.selectbox("Select gene", ["DHFR", "DHPS"], index=0, key="trend_gene")
        snp = st.selectbox("Select SNP", get_snp_options(df, gene), index=0, key="trend_snp")
        location_options = get_geographic_options(df, snp, geographic_column)
        location_label = {
            "Country": "Select country",
            "Africa_subregion": "Select Africa subregion",
            "Region": "Select region",
        }[geographic_column]
        selected_location = st.selectbox(
            location_label,
            location_options,
            index=0 if location_options else None,
            key="trend_snp_location",
        )
        show_period_button = st.button("Show pooled prevalence by period", width='stretch')

    selected_geo_df = (
        df[df[geographic_column].astype(str) == selected_location]
        if selected_location
        else df.iloc[0:0]
    )
    location_context = f"in {selected_location}" if selected_location else ""
    st.subheader(f"Prevalence trend for {snp} {location_context}".strip())
    if not selected_location:
        st.info(f"No {geographic_label} data is available for the selected SNP.")
    else:
        trend_df = build_country_trend(df, snp, selected_location, geographic_column)
        if trend_df.empty:
            st.info(f"No trend data is available for the selected SNP and {geographic_label}.")
        else:
            fig = go.Figure()
            marker_sizes = 6 + (trend_df["prevalence"].clip(0, 1) * 20)
            ci_mid = (trend_df["ci_lower"] + trend_df["ci_upper"]) / 2.0

            fig.add_trace(
                go.Scatter(
                    x=trend_df["mid_sampling_year"],
                    y=trend_df["ci_lower"],
                    mode="lines",
                    line=dict(color="rgba(255, 204, 153, 0.75)", width=1, shape="spline", smoothing=1.0),
                    name="95% CI",
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=trend_df["mid_sampling_year"],
                    y=trend_df["ci_upper"],
                    mode="lines",
                    line=dict(color="rgba(255, 204, 153, 0.75)", width=1, shape="spline", smoothing=1.0),
                    fill="tonexty",
                    fillcolor="rgba(255, 204, 153, 0.35)",
                    name="95% CI",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=trend_df["mid_sampling_year"],
                    y=ci_mid,
                    mode="lines",
                    name="Trend line",
                    line=dict(color="#d62728", width=1.2, shape="spline", smoothing=1.0),
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=trend_df["mid_sampling_year"],
                    y=trend_df["prevalence"],
                    mode="markers",
                    name="Observed prevalence",
                    marker=dict(color="#008CFF", size=marker_sizes, sizemode="diameter"),
                    customdata=trend_df[["Author", "Year", "count", "denominator", "ci_lower", "ci_upper"]].fillna("").to_numpy(),
                    hovertemplate=(
                        "Year: %{x:.0f}<br>Prevalence: %{y:.2%}<br>95% CI: %{customdata[4]:.2%} to %{customdata[5]:.2%}"
                        "<br>Mutations: %{customdata[2]} / %{customdata[3]}<br>Author: %{customdata[0]}"
                        #"<br>Publication/Sampling year: %{customdata[1]}<extra></extra>"
                    ),
                )
            )

            fig.update_layout(title=f"{snp} prevalence trend in {selected_location}")
            fig.update_layout(yaxis_range=[0, 1.1], xaxis_title="Mid sampling year", yaxis_title="Prevalence")
            st.plotly_chart(fig, width='stretch')

    if show_period_button:
        st.subheader(
            f"Random-effects pooled prevalence for {snp} in {selected_location or geographic_label} by period"
        )
        period_df = build_period_prevalence(selected_geo_df, snp)
        if period_df.empty:
            st.info("No period-level data is available for the selected SNP.")
        else:
            period_df["pooled_prevalence"] = period_df["pooled_prevalence"].clip(0, 1)
            display_df = period_df[["period", "Studies (k)", "Total Mutants", "Total Genotyped", "Pooled Prev (%)", "95% CI", "I2 (%)", "Tau2", "p-value (Q)"]].copy()
            display_df["Pooled Prev (%)"] = display_df["Pooled Prev (%)"].map(lambda x: f"{x:.2f}")
            display_df["I2 (%)"] = display_df["I2 (%)"].map(lambda x: f"{x:.2f}")
            display_df["Tau2"] = display_df["Tau2"].map(lambda x: f"{x:.5f}")
            display_df["p-value (Q)"] = display_df["p-value (Q)"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
            display_df = display_df.rename(columns={"I2 (%)": "I² (%)", "Tau2": "Tau²"})
            st.dataframe(display_df, width='stretch')
            fig = px.bar(
                period_df,
                x="period",
                y="pooled_prevalence",
                text=period_df["pooled_prevalence"].map(lambda x: f"{x:.2%}"),
                title=f"{snp} pooled prevalence by period",
                color="period",
            )
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            fig.update_layout(xaxis_title="Period", yaxis_title="Pooled prevalence")
            st.plotly_chart(fig, width='stretch')

elif view_mode == "World choropleth (SNP)":
    with st.sidebar:
        st.header("World choropleth controls")
        gene = st.selectbox("Select gene", ["DHFR", "DHPS"], index=0, key="choropleth_gene")
        snp = st.selectbox("Select SNP", get_snp_options(df, gene), index=0, key="choropleth_snp")
        period_options = ["2000-2004", "2005-2012", "2013-2026"]
        selected_periods = st.multiselect(
            "Select Temporal Policy Epoch(s)",
            options=period_options,
            default=period_options,
            key="choropleth_periods",
        )

    st.subheader(f"Pooled prevalence of {snp} across the selected periods")
    if not selected_periods:
        st.info("Select at least one period to display the choropleth.")
    else:
        choropleth_df = build_period_choropleth_data(df, snp, selected_periods)
        if choropleth_df.empty:
            st.info("No data is available for the selected SNP and periods.")
        else:
            choropleth_df = choropleth_df.copy()
            choropleth_df["country_iso3"] = choropleth_df["Country"].map(country_to_iso3)
            choropleth_df = choropleth_df.dropna(subset=["country_iso3"])
            if choropleth_df.empty:
                st.info("No country ISO-3 codes are available for the selected SNP and periods.")
            else:
                fig = px.choropleth(
                    choropleth_df,
                    locations="country_iso3",
                    locationmode="ISO-3",
                    color="prevalence",
                    hover_name="Country",
                    hover_data={"prevalence": ":.2%", "count": True, "denominator": True},
                    color_continuous_scale="magma_r",
                    range_color=(0, 1),
                    title=f"{snp} pooled prevalence by country for {', '.join(selected_periods)}",
                )
                fig.update_layout(
                    geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=650,
                )
                fig.update_traces(colorbar=dict(title="Prevalence", tickformat=".0%"))
                st.plotly_chart(fig, width='stretch')

            st.subheader(f"Country-level prevalence of {snp} across selected periods")
            scatter_df = build_period_scatter_data(df, snp, selected_periods)
            if scatter_df.empty:
                st.info("No country-level point data is available for the selected SNP and periods.")
            else:
                color_map = {
                    "<500": "#1f77b4",
                    "501-1000": "#ff7f0e",
                    "1001-5000": "#2ca02c",
                    ">5000": "#d62728",
                    "Unknown": "#7f7f7f",
                }
                fig2 = px.scatter(
                    scatter_df,
                    x="mid_sampling_year",
                    y="Country",
                    size="prevalence",
                    color="sample_size_bin",
                    hover_name="Country",
                    hover_data={"mid_sampling_year": True, "prevalence": ":.2%", "Sample size": True, "period": True},
                    size_max=18,
                    color_discrete_map=color_map,
                    category_orders={"sample_size_bin": ["<500", "501-1000", "1001-5000", ">5000", "Unknown"], "Country": sorted(scatter_df["Country"].unique())},
                    title=f"{snp} prevalence by country and year for {', '.join(selected_periods)}",
                )
                fig2.update_layout(xaxis_title="Year", yaxis_title="Country", legend_title="Sample size bin")
                fig2.update_traces(marker=dict(opacity=0.8))
                st.plotly_chart(fig2, width='stretch')

elif view_mode == "Country trend (haplotype)":
    with st.sidebar:
        st.header(f"{geographic_label.title()} trend controls")
        gene = st.selectbox("Select gene", ["DHFR", "DHPS"], index=0, key="trend_hap_gene")
        haplotype = st.selectbox("Select haplotype", get_haplotype_options(df, gene), index=0, key="trend_hap_marker")
        location_options = get_geographic_options(df, haplotype, geographic_column)
        location_label = {
            "Country": "Select country",
            "Africa_subregion": "Select Africa subregion",
            "Region": "Select region",
        }[geographic_column]
        selected_location = st.selectbox(
            location_label,
            location_options,
            index=0 if location_options else None,
            key="trend_hap_location",
        )
        show_period_button = st.button("Show pooled prevalence by period", width='stretch', key="trend_hap_period_button")

    selected_geo_df = (
        df[df[geographic_column].astype(str) == selected_location]
        if selected_location
        else df.iloc[0:0]
    )
    location_context = f"in {selected_location}" if selected_location else ""
    st.subheader(f"Prevalence trend for {haplotype} {location_context}".strip())
    if not selected_location:
        st.info(f"No {geographic_label} data is available for the selected haplotype.")
    else:
        trend_df = build_country_trend(df, haplotype, selected_location, geographic_column)
        if trend_df.empty:
            st.info(f"No trend data is available for the selected haplotype and {geographic_label}.")
        else:
            fig = go.Figure()
            marker_sizes = 6 + (trend_df["prevalence"].clip(0, 1) * 20)
            ci_mid = (trend_df["ci_lower"] + trend_df["ci_upper"]) / 2.0

            fig.add_trace(
                go.Scatter(
                    x=trend_df["mid_sampling_year"],
                    y=trend_df["ci_lower"],
                    mode="lines",
                    line=dict(color="rgba(255, 204, 153, 0.75)", width=1, shape="spline", smoothing=1.0),
                    name="95% CI",
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=trend_df["mid_sampling_year"],
                    y=trend_df["ci_upper"],
                    mode="lines",
                    line=dict(color="rgba(255, 204, 153, 0.75)", width=1, shape="spline", smoothing=1.0),
                    fill="tonexty",
                    fillcolor="rgba(255, 204, 153, 0.35)",
                    name="95% CI",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=trend_df["mid_sampling_year"],
                    y=ci_mid,
                    mode="lines",
                    name="Trend line",
                    line=dict(color="#d62728", width=1.2, shape="spline", smoothing=1.0),
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=trend_df["mid_sampling_year"],
                    y=trend_df["prevalence"],
                    mode="markers",
                    name="Observed prevalence",
                    marker=dict(color="#008CFF", size=marker_sizes, sizemode="diameter"),
                    customdata=trend_df[["Author", "Year", "count", "denominator", "ci_lower", "ci_upper"]].fillna("").to_numpy(),
                    hovertemplate=(
                        "Year: %{x:.0f}<br>Prevalence: %{y:.2%}<br>95% CI: %{customdata[4]:.2%} to %{customdata[5]:.2%}"
                        "<br>Haplotype positives: %{customdata[2]} / %{customdata[3]}<br>Author: %{customdata[0]}"
                        #"<br>Publication/Sampling year: %{customdata[1]}<extra></extra>"
                    ),
                )
            )

            fig.update_layout(title=f"{haplotype} prevalence trend in {selected_location}")
            fig.update_layout(yaxis_range=[0, 1.1], xaxis_title="Mid sampling year", yaxis_title="Prevalence")
            st.plotly_chart(fig, width='stretch')

    if show_period_button:
        st.subheader(
            f"Pooled prevalence for {haplotype} in {selected_location or geographic_label} by period"
        )
        period_df = build_period_prevalence(selected_geo_df, haplotype)
        if period_df.empty:
            st.info("No period-level data is available for the selected haplotype.")
        else:
            period_df["pooled_prevalence"] = period_df["pooled_prevalence"].clip(0, 1)
            display_df = period_df[["period", "Studies (k)", "Total Mutants", "Total Genotyped", "Pooled Prev (%)", "95% CI", "I2 (%)", "Tau2", "p-value (Q)"]].copy()
            display_df["Pooled Prev (%)"] = display_df["Pooled Prev (%)"].map(lambda x: f"{x:.2f}")
            display_df["I2 (%)"] = display_df["I2 (%)"].map(lambda x: f"{x:.2f}")
            display_df["Tau2"] = display_df["Tau2"].map(lambda x: f"{x:.5f}")
            display_df["p-value (Q)"] = display_df["p-value (Q)"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
            display_df = display_df.rename(columns={"I2 (%)": "I² (%)", "Tau2": "Tau²"})
            st.dataframe(display_df, width='stretch')
            fig = px.bar(
                period_df,
                x="period",
                y="pooled_prevalence",
                text=period_df["pooled_prevalence"].map(lambda x: f"{x:.2%}"),
                title=f"{haplotype} pooled prevalence by period",
                color="period",
            )
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            fig.update_layout(xaxis_title="Period", yaxis_title="Pooled prevalence")
            st.plotly_chart(fig, width='stretch')

elif view_mode == "World choropleth (haplotype)":
    with st.sidebar:
        st.header("World choropleth controls")
        gene = st.selectbox("Select gene", ["DHFR", "DHPS"], index=0, key="choropleth_hap_gene")
        haplotype = st.selectbox("Select haplotype", get_haplotype_options(df, gene), index=0, key="choropleth_hap_marker")
        period_options = ["2000-2004", "2005-2012", "2013-2026"]
        selected_periods = st.multiselect(
            "Select Temporal Policy Epoch(s)",
            options=period_options,
            default=period_options,
            key="choropleth_hap_periods",
        )

    st.subheader(f"Pooled prevalence of {haplotype} across the selected periods")
    if not selected_periods:
        st.info("Select at least one period to display the choropleth.")
    else:
        choropleth_df = build_period_choropleth_data(df, haplotype, selected_periods)
        if choropleth_df.empty:
            st.info("No data is available for the selected haplotype and periods.")
        else:
            choropleth_df = choropleth_df.copy()
            choropleth_df["country_iso3"] = choropleth_df["Country"].map(country_to_iso3)
            choropleth_df = choropleth_df.dropna(subset=["country_iso3"])
            if choropleth_df.empty:
                st.info("No country ISO-3 codes are available for the selected haplotype and periods.")
            else:
                fig = px.choropleth(
                    choropleth_df,
                    locations="country_iso3",
                    locationmode="ISO-3",
                    color="prevalence",
                    hover_name="Country",
                    hover_data={"prevalence": ":.2%", "count": True, "denominator": True},
                    color_continuous_scale="magma_r",
                    range_color=(0, 1),
                    title=f"{haplotype} pooled prevalence by country for {', '.join(selected_periods)}",
                )
                fig.update_layout(
                    geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=650,
                )
                fig.update_traces(colorbar=dict(title="Prevalence", tickformat=".0%"))
                st.plotly_chart(fig, width='stretch')

            st.subheader(f"Country-level prevalence of {haplotype} across selected periods")
            scatter_df = build_period_scatter_data(df, haplotype, selected_periods)
            if scatter_df.empty:
                st.info("No country-level point data is available for the selected haplotype and periods.")
            else:
                color_map = {
                    "<500": "#1f77b4",
                    "501-1000": "#ff7f0e",
                    "1001-5000": "#2ca02c",
                    ">5000": "#d62728",
                    "Unknown": "#7f7f7f",
                }
                fig2 = px.scatter(
                    scatter_df,
                    x="mid_sampling_year",
                    y="Country",
                    size="prevalence",
                    color="sample_size_bin",
                    hover_name="Country",
                    hover_data={"mid_sampling_year": True, "prevalence": ":.2%", "Sample size": True, "period": True},
                    size_max=18,
                    color_discrete_map=color_map,
                    category_orders={"sample_size_bin": ["<500", "501-1000", "1001-5000", ">5000", "Unknown"], "Country": sorted(scatter_df["Country"].unique())},
                    title=f"{haplotype} prevalence by country and year for {', '.join(selected_periods)}",
                )
                fig2.update_layout(xaxis_title="Year", yaxis_title="Country", legend_title="Sample size bin")
                fig2.update_traces(marker=dict(opacity=0.8))
                st.plotly_chart(fig2, width='stretch')
