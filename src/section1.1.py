"""
COMP0035 Coursework 1 — Section 1.1.

Data Description & Exploration (concise).

Usage:
    python src/section1.1.py --csv "D:/path/to/GraduateEmploymentSurvey.csv"
    # or just: python src/section1.1.py  (auto-search a CSV in project)

Outputs:
- Tables -> <project_root>/eda_output
- Figures -> <project_root>/figs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------
# Paths / logging
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "eda_output"
FIG_DIR = PROJECT_ROOT / "figs"
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "7-GraduateEmploymentSurveyNTUNUSSITSMUSUSSSUTD.csv"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("s1.1")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def auto_find_csv(start: Path) -> Path | None:
    """Find a likely CSV (prefer data/raw, then shortest path)."""
    patterns = ["*GraduateEmploymentSurvey*.csv", "*.csv"]
    bad_parts = {".venv", ".git", "node_modules", "__pycache__"}
    for pat in patterns:
        cands = sorted(
            (p for p in start.rglob(pat) if not any(b in p.parts for b in bad_parts)),
            key=lambda p: (0 if {"data", "raw"} <= set(p.parts) else 1, len(str(p))),
        )
        if cands:
            return cands[0]
    return None


def load_csv(path: Path) -> pd.DataFrame:
    """Read CSV with utf-8-sig fallback."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)


def save_table(df: pd.DataFrame, name: str) -> None:
    """Save dataframe to eda_output as CSV."""
    (TABLE_DIR / f"{name}.csv").write_text(df.to_csv(index=True), encoding="utf-8")


# ---------------------------------------------------------------------
# Light coercions for plotting
# ---------------------------------------------------------------------
def _to_percent(x: object) -> float | pd.NA:
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().replace(" ", "")
    if s.endswith("%"):
        s = s[:-1]
    return pd.to_numeric(s, errors="coerce")


def _to_money(x: object) -> float | pd.NA:
    if pd.isna(x):
        return pd.NA
    s = str(x).replace("$", "").replace(",", "").strip()
    return pd.to_numeric(s, errors="coerce")


def coerce_for_plots(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce percent/salary/year to numeric for plotting."""
    out = df.copy()
    for col in {"employment_rate_overall", "employment_rate_ft_perm"} & set(out.columns):
        out[col] = out[col].apply(_to_percent)
    for col in [
        "basic_monthly_mean",
        "basic_monthly_median",
        "gross_monthly_mean",
        "gross_monthly_median",
        "gross_mthly_25_percentile",
        "gross_mthly_75_percentile",
    ]:
        if col in out.columns:
            out[col] = out[col].apply(_to_money)
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
    return out


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------
def structural_summary(df: pd.DataFrame) -> dict[str, object]:
    """Return basic structure: shape, columns, dtypes, memory bytes."""
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
    }


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing counts/%, dtype, uniques (+ memory row)."""
    n = len(df)
    rpt = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isna().sum(),
            "missing_pct": (df.isna().sum() / max(n, 1) * 100).round(2),
            "n_unique": df.nunique(dropna=True),
        }
    )
    rpt.loc["__memory_usage_bytes__", "dtype"] = str(int(df.memory_usage(deep=True).sum()))
    return rpt.sort_values(by=["missing_pct", "n_unique"], ascending=[False, True])


def descriptive_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numeric + categorical describe tables."""
    num_desc = df.select_dtypes(include="number").describe().T
    cat_desc = df.select_dtypes(include=["object", "category"]).describe().T
    return num_desc, cat_desc


def save_topn_freq(df: pd.DataFrame, col: str, top_n: int = 10) -> None:
    """Save top-N frequency table for a categorical column."""
    if col not in df.columns:
        return
    out = df[col].astype("string").value_counts(dropna=False).head(top_n)
    save_table(out.to_frame("count"), f"top{top_n}_freq_{col}")


# ---------------------------------------------------------------------
# Plots (single-plot-per-file)
# ---------------------------------------------------------------------
def _save_hist(df: pd.DataFrame, col: str, title: str, xlabel: str) -> None:
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return
    plt.figure()
    df[col].dropna().plot(kind="hist", bins=20, title=title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"hist_{col}.png", dpi=160)
    plt.close()


def _save_box_by(
    df: pd.DataFrame,
    value_col: str,
    by_col: str,
    title: str,
    ylabel: str,
) -> None:
    if {value_col, by_col} - set(df.columns):
        return
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return
    tmp = df[[value_col, by_col]].dropna()
    if tmp.empty:
        return
    plt.figure()
    tmp.boxplot(column=value_col, by=by_col, rot=45)
    plt.title(title)
    plt.suptitle("")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"box_{value_col}_by_{by_col}.png", dpi=160)
    plt.close()


def _save_trend(
    df: pd.DataFrame,
    y_col: str,
    x_col: str = "year",
    agg: str = "median",
    label: str = "Median",
) -> None:
    if {x_col, y_col} - set(df.columns):
        return
    if not pd.api.types.is_numeric_dtype(df[x_col]):
        return
    tmp = pd.DataFrame(
        {
            x_col: pd.to_numeric(df[x_col], errors="coerce"),
            y_col: pd.to_numeric(df[y_col], errors="coerce"),
        }
    ).dropna()
    if tmp.empty:
        return
    ts = getattr(tmp.groupby(x_col)[y_col], agg)().sort_index()
    plt.figure()
    title = f"{label} {y_col.replace('_', ' ').title()} over time"
    ts.plot(kind="line", marker="o", title=title)
    plt.xlabel(x_col)
    plt.ylabel(y_col.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"trend_{y_col}_over_time.png", dpi=160)
    plt.close()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> int:
    """Generate EDA tables and figures for Section 1.1."""
    parser = argparse.ArgumentParser(description="EDA for Graduate Employment Survey.")
    parser.add_argument("--csv", type=str, default=None, help="Path to dataset CSV.")
    args = parser.parse_args()

    # Resolve CSV path.
    if args.csv:
        csv_path = Path(args.csv)
        log.info("Using --csv: %s", csv_path)
    elif DEFAULT_CSV.exists():
        csv_path = DEFAULT_CSV
        log.info("Using default CSV: %s", csv_path)
    else:
        log.warning("Default CSV not found. Auto-detecting CSV in project...")
        auto = auto_find_csv(PROJECT_ROOT)
        if auto is None:
            log.error("Could not find any CSV. Provide --csv PATH.")
            return 2
        csv_path = auto
        log.info("Auto-detected CSV: %s", csv_path)

    # Load + light coercions.
    df_raw = load_csv(csv_path)
    df_plot = coerce_for_plots(df_raw)

    # Structural summary (printed).
    log.info("Structural summary: %s", structural_summary(df_raw))

    # Tables.
    num_desc, cat_desc = descriptive_stats(df_raw)
    save_table(num_desc, "describe_numeric")
    save_table(cat_desc, "describe_categorical")
    save_table(data_quality_report(df_raw), "data_quality_report")
    df_raw.head(20).to_csv(PROJECT_ROOT / "preview_head20.csv", index=False, encoding="utf-8")

    # Histograms (if present).
    for col, title, xlabel in [
        ("year", "Year Distribution", "Year"),
        (
            "employment_rate_overall",
            "Overall Employment Rate Distribution",
            "Employment Rate (%)",
        ),
        (
            "employment_rate_ft_perm",
            "Full-time Permanent Employment Rate Distribution",
            "Employment Rate (%)",
        ),
        ("basic_monthly_mean", "Basic Monthly Mean Salary Distribution", "Salary (SGD)"),
        ("gross_monthly_mean", "Gross Monthly Mean Salary Distribution", "Salary (SGD)"),
        (
            "gross_monthly_median",
            "Gross Monthly Median Salary Distribution",
            "Salary (SGD)",
        ),
    ]:
        if col in df_plot.columns:
            _save_hist(df_plot, col, title, xlabel)

    # Top10 frequencies for categorical columns.
    for col in df_raw.select_dtypes(include=["object", "category"]).columns:
        save_topn_freq(df_raw, col, top_n=10)

    # Boxplots by university.
    for val_col, ylabel in [
        ("employment_rate_overall", "Employment Rate (%)"),
        ("basic_monthly_mean", "Salary (SGD)"),
        ("gross_monthly_median", "Salary (SGD)"),
    ]:
        if "university" in df_plot.columns and val_col in df_plot.columns:
            _save_box_by(
                df_plot,
                val_col,
                "university",
                f"{val_col.replace('_', ' ').title()} by University",
                ylabel,
            )

    # Yearly trends (median).
    for y in ["gross_monthly_median", "basic_monthly_median", "employment_rate_overall"]:
        if y in df_plot.columns:
            _save_trend(df_plot, y, "year", agg="median", label="Median")

    log.info("Section 1.1 EDA complete.")
    log.info(" - Input : %s", csv_path)
    log.info(" - Tables: %s", TABLE_DIR.resolve())
    log.info(" - Figures: %s", FIG_DIR.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
