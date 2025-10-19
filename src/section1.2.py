"""
section1.2.py

COMP0035 Coursework 1 — Section 1.2
- Define target audience and 3 questions (documented in the report; code focuses on prep)
- Clean columns (percent strings -> float, salary strings -> numeric)
- Prepare data subsets per question
- Make simple visuals (matplotlib, single-plot)
- Save processed datasets for reproducibility
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


RAW_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "7-GraduateEmploymentSurveyNTUNUSSITSMUSUSSSUTD.csv"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FIGS_DIR = Path(__file__).resolve().parents[1] / "figs"


def _to_float_pct(x: object) -> float | np.nan:
    """
    Convert strings like '89.5%' or '89.5 %' to 0.895 float.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace(" ", "")
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s) / 100.0
    except ValueError:
        return np.nan


def _to_number(x: object) -> float | np.nan:
    """
    Convert salary fields like '3,500', '4500', '$4,200' to numeric float.
    """
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = s.replace("$", "").replace(",", "").strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise schema and types.
    """
    df = df.copy()
    # Rename columns to snake_case (already simple here)
    df.columns = [c.strip().lower() for c in df.columns]

    # Convert percentages
    for c in ["employment_rate_overall", "employment_rate_ft_perm"]:
        if c in df.columns:
            df[c] = df[c].apply(_to_float_pct)

    # Convert salary-like numbers
    for c in [
        "basic_monthly_mean",
        "basic_monthly_median",
        "gross_monthly_mean",
        "gross_monthly_median",
        "gross_mthly_25_percentile",
        "gross_mthly_75_percentile",
    ]:
        if c in df.columns:
            df[c] = df[c].apply(_to_number)

    # Trim whitespace for categoricals
    for c in ["university", "school", "degree"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    # Drop impossible years if any (e.g., 0 or future)
    if "year" in df.columns:
        df = df[(df["year"] >= 2000) & (df["year"] <= 2100)]

    return df


def q1_table_top_median_salary(df: pd.DataFrame, group_field: str = "degree", year_min: int | None = None, year_max: int | None = None, top_n: int = 15) -> pd.DataFrame:
    """
    Q1: 哪些学科/项目拥有最高中位数总月薪（可设定按 degree 或 school 分组）。
    """
    d = df.copy()
    if year_min is not None:
        d = d[d["year"] >= year_min]
    if year_max is not None:
        d = d[d["year"] <= year_max]

    metric = "gross_monthly_median"
    grouped = (
        d.groupby(group_field, dropna=True)[metric]
        .median()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    return grouped


def q2_series_employment_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Q2: 2018-2023 整体就业率趋势（按年份平均 overall employment rate）。
    """
    d = df.copy()
    metric = "employment_rate_overall"
    trend = (
        d.groupby("year", dropna=True)[metric]
        .mean()
        .sort_index()
        .reset_index()
    )
    return trend


def q3_table_degree_level_effect(df: pd.DataFrame) -> pd.DataFrame:
    """
    Q3: 学历层次（本科/硕士等）对就业结果的影响（按 degree 层级聚合）。
    这里简化：从 degree 字符串中提取 'Bachelor', 'Master', 'PhD' 等关键词。
    """
    d = df.copy()
    def map_level(s: str) -> str:
        s2 = (s or "").lower()
        if "phd" in s2 or "doctoral" in s2:
            return "PhD"
        if "master" in s2 or "msc" in s2 or "ma " in s2 or s2.startswith("m"):
            return "Master"
        if "bachelor" in s2 or "bsc" in s2 or s2.startswith("b"):
            return "Bachelor"
        return "Other"

    d["degree_level"] = d["degree"].astype("string").apply(map_level)

    agg = (
        d.groupby("degree_level", dropna=True)
        .agg(
            employment_rate_overall=("employment_rate_overall", "mean"),
            gross_monthly_median=("gross_monthly_median", "median"),
            n=("degree_level", "count"),
        )
        .reset_index()
        .sort_values(by=["employment_rate_overall","gross_monthly_median"], ascending=False)
    )
    return agg


def save_plot_line(df_xy: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
    plt.figure()
    plt.plot(df_xy[x], df_xy[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_plot_bar(df_xy: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
    plt.figure()
    plt.bar(df_xy[x], df_xy[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> int:
    df_raw = pd.read_csv(RAW_CSV)
    df_clean = clean_dataset(df_raw)
    # Save cleaned data for reproducibility
    out_clean = PROC_DIR / "ntu_employment_clean.csv"
    df_clean.to_csv(out_clean, index=False)

    # Q1: Top median salary by degree (last 3 years if available)
    year_max = int(df_clean["year"].max())
    year_min = max(year_max - 2, int(df_clean["year"].min()))
    q1 = q1_table_top_median_salary(df_clean, group_field="degree", year_min=year_min, year_max=year_max, top_n=15)
    q1.to_csv(PROC_DIR / "q1_top_median_salary_by_degree.csv", index=False)
    save_plot_bar(q1, x="degree", y="gross_monthly_median",
                  title=f"Top Median Gross Monthly Salary by Degree ({year_min}-{year_max})",
                  out_path=FIGS_DIR / "q1_top_median_salary_by_degree.png")

    # Q2: Employment rate trend by year
    q2 = q2_series_employment_trend(df_clean)
    q2.to_csv(PROC_DIR / "q2_employment_rate_trend.csv", index=False)
    save_plot_line(q2, x="year", y="employment_rate_overall",
                   title="Average Overall Employment Rate by Year",
                   out_path=FIGS_DIR / "q2_employment_rate_trend.png")

    # Q3: Degree level effect
    q3 = q3_table_degree_level_effect(df_clean)
    q3.to_csv(PROC_DIR / "q3_degree_level_effect.csv", index=False)
    save_plot_bar(q3, x="degree_level", y="employment_rate_overall",
                  title="Employment Rate by Degree Level (avg)",
                  out_path=FIGS_DIR / "q3_degree_level_employment.png")

    print("[DONE] Data preparation complete.")
    print("Clean CSV:", out_clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
