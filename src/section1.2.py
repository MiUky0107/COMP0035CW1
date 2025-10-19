"""
COMP0035 Coursework 1 — Section 1.2 Data Preparation.

This script cleans the raw CSV and prepares subsets/figures for three example questions (Q1–Q3):
- Standardise schema and types.
- Convert salary-like fields to numeric.
- Convert percentage strings to floats in the 0–1 range.
- Save processed tables and single-plot figures.

Outputs:
- Cleaned preview and all intermediate tables: data/processed
- Figures: figs
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "7-GraduateEmploymentSurveyNTUNUSSITSMUSUSSSUTD.csv"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
FIGS_DIR = PROJECT_ROOT / "figs"
PROC_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Utilities ----------
def _to_float_pct(x: object) -> float | np.nan:
    """Convert '89.5%' or '89.5 %' to 0.895."""
    if pd.isna(x):
        return np.nan
    try:
        s = str(x).replace("%", "").strip()
        v = float(s)
        return v / 100.0
    except Exception:
        return np.nan


def _to_number(x: object) -> float | np.nan:
    """Convert salary-like strings such as '3,500', '4500', '$4,200' to float."""
    if pd.isna(x):
        return np.nan
    try:
        s = str(x).replace("$", "").replace(",", "").strip()
        return float(s)
    except Exception:
        return np.nan


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise schema and numeric types for preparation tasks."""
    df = df.copy()

    # Ensure expected canonical column names exist if the dataset uses aliases.
    # (Adjust here only if your raw CSV uses different headers.)
    # Example:
    # df.rename(columns={"University": "university"}, inplace=True)

    # Percent columns -> 0–1 floats.
    pct_cols = []
    for c in ["employment_rate_overall", "employment_rate_ft_perm"]:
        if c in df.columns:
            df[c] = df[c].apply(_to_float_pct)
            pct_cols.append(c)

    # Salary columns -> numeric floats.
    sal_cols = []
    for c in [
        "basic_monthly_mean",
        "basic_monthly_median",
        "gross_monthly_mean",
        "gross_monthly_median",
    ]:
        if c in df.columns:
            df[c] = df[c].apply(_to_number)
            sal_cols.append(c)

    # Year -> numeric (keep as int where possible).
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Drop exact duplicates to avoid double counting.
    df = df.drop_duplicates()

    # Keep a preview for the report.
    df.head(20).to_csv(PROC_DIR / "preview_head20.csv", index=False)

    # Save numeric/categorical descriptions for traceability.
    num_desc = df.select_dtypes(include=["number", "Int64"]).describe().T
    num_desc.to_csv(PROC_DIR / "num_desc.csv")
    cat_desc = df.select_dtypes(exclude=["number", "Int64"]).describe().T
    cat_desc.to_csv(PROC_DIR / "cat_desc.csv")

    return df


# ---------- Small plotting helpers (single-plot) ----------
def save_plot_line(df_xy: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
    """Save a simple line plot for series y over x."""
    if x not in df_xy.columns or y not in df_xy.columns:
        return
    if df_xy.empty:
        return
    plt.figure()
    plt.plot(df_xy[x], df_xy[y], marker="o")
    plt.title(title)
    plt.xlabel(x.title())
    plt.ylabel(y.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_plot_bar(df_xy: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
    """Save a simple bar chart for df[x] vs df[y]."""
    if x not in df_xy.columns or y not in df_xy.columns:
        return
    if df_xy.empty:
        return
    plt.figure()
    plt.bar(df_xy[x], df_xy[y])
    plt.title(title)
    plt.xlabel(x.title())
    plt.ylabel(y.replace("_", " ").title())
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------- Q1: Highest median salary (latest window) ----------
def q1_table_top_median_salary(
    df: pd.DataFrame,
    group_field: str = "degree",
    year_min: int | None = None,
    year_max: int | None = None,
    top_n: int = 15,
) -> pd.DataFrame:
    """Return top-N groups by median gross monthly salary within a year window."""
    d = df.copy()

    if "gross_monthly_median" not in d.columns or group_field not in d.columns:
        return pd.DataFrame()

    if year_min is not None:
        d = d[d["year"] >= year_min]
    if year_max is not None:
        d = d[d["year"] <= year_max]

    d = d.dropna(subset=["gross_monthly_median"])

    g = (
        d.groupby(group_field)["gross_monthly_median"]
        .median()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    g.columns = [group_field, "gross_monthly_median"]
    return g


# ---------- Q2: Employment rate trend (overall) ----------
def q2_series_employment_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean overall employment rate by year (0–1 scale)."""
    metric = "employment_rate_overall"
    if "year" not in df.columns or metric not in df.columns:
        return pd.DataFrame()

    d = df.dropna(subset=[metric, "year"]).copy()
    ser = d.groupby("year")[metric].mean().sort_index()
    return ser.reset_index().rename(columns={metric: "employment_rate_overall"})


# ---------- Q3: Degree level effect ----------
def q3_table_degree_level_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Return aggregated outcomes by degree level (Bachelor/Master/PhD/Other)."""

    def map_level(s: str) -> str:
        s = (s or "").lower()
        if "bachelor" in s or "bsc" in s or "ba" in s or "beng" in s:
            return "Bachelor"
        if "master" in s or "msc" in s or "meng" in s or "ma" in s:
            return "Master"
        if "phd" in s or "doctor" in s:
            return "PhD"
        return "Other"

    req_cols = [
        "degree",
        "employment_rate_overall",
        "gross_monthly_median",
        "year",
    ]
    if not set(req_cols).issubset(df.columns):
        return pd.DataFrame()

    d = df.dropna(subset=["degree"]).copy()
    d["degree_level"] = d["degree"].astype(str).map(map_level)

    agg = (
        d.groupby("degree_level", dropna=False)
        .agg(
            n=("degree_level", "size"),
            employment_rate_overall=("employment_rate_overall", "mean"),
            gross_monthly_median=("gross_monthly_median", "median"),
        )
        .reset_index()
        .sort_values(["gross_monthly_median", "employment_rate_overall"], ascending=False)
    )
    return agg


# ---------- Main ----------
def main() -> int:
    """Clean raw data and prepare Q1–Q3 outputs with example figures."""
    # Load raw CSV (support UTF-8 BOM).
    try:
        df_raw = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df_raw = pd.read_csv(RAW_CSV)

    # Clean dataset.
    df_clean = clean_dataset(df_raw)

    # Persist a full cleaned copy if you need it downstream (optional).
    df_clean.to_csv(PROC_DIR / "clean_full.csv", index=False)

    # ------ Q1: top-N gross_monthly_median by group ------
    year_max = int(pd.to_numeric(df_clean["year"], errors="coerce").dropna().max())
    year_min = max(
        year_max - 2, int(pd.to_numeric(df_clean["year"], errors="coerce").dropna().min())
    )

    q1 = q1_table_top_median_salary(
        df_clean,
        group_field="degree",
        year_min=year_min,
        year_max=year_max,
        top_n=15,
    )
    q1.to_csv(PROC_DIR / "q1_top_median_salary_by_degree.csv", index=False)
    save_plot_bar(
        q1,
        x="degree",
        y="gross_monthly_median",
        title=f"Top Median Gross Monthly Salary by Degree ({year_min}–{year_max})",
        out_path=FIGS_DIR / "q1_top_median_salary_by_degree.png",
    )

    # ------ Q2: overall employment rate trend (0–1) ------
    q2 = q2_series_employment_trend(df_clean)
    q2.to_csv(PROC_DIR / "q2_overall_employment_rate_trend.csv", index=False)
    save_plot_line(
        q2,
        x="year",
        y="employment_rate_overall",
        title="Overall Employment Rate Trend (mean, 0–1)",
        out_path=FIGS_DIR / "q2_overall_employment_rate_trend.png",
    )

    # ------ Q3: degree level effect ------
    q3 = q3_table_degree_level_effect(df_clean)
    q3.to_csv(PROC_DIR / "q3_degree_level_effect.csv", index=False)
    save_plot_bar(
        q3,
        x="degree_level",
        y="gross_monthly_median",
        title="Degree Level vs Gross Monthly Median",
        out_path=FIGS_DIR / "q3_degree_level_vs_salary.png",
    )

    print("✅ Section 1.2 data preparation completed.")
    print(f"   Input : {RAW_CSV}")
    print(f"   Output tables: {PROC_DIR}")
    print(f"   Figures: {FIGS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
