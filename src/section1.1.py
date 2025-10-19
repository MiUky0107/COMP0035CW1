"""
COMP0035 Coursework 1 — Section 1.1
Data Description & Exploration (融合增强版)

功能：
- 支持 --csv 路径；若未提供则自动在项目内搜索 *GraduateEmploymentSurvey*.csv
- 结构化概览：shape/columns/dtypes/memory
- 数据质量报告：缺失计数、缺失百分比、唯一值、dtype、总内存
- 描述性统计表（数值/全量）
- 直方图（数值列）、Top10 频数（类别列）
- 箱线图（按 university 分组的关键数值列）
- 趋势线（year 聚合的中位数/均值）
- 所有图均为“单图单文件”，符合课程要求

运行示例：
    python src/section1.1.py --csv "D:/COMP0035/7-GraduateEmploymentSurveyNTUNUSSITSMUSUSSSUTD.csv"
或：python src/section1.1.py   # 自动搜索
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 配置：输出目录
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "eda_output"   # 报告用表格
FIG_DIR = PROJECT_ROOT / "figs"           # 图片
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 路径 & 读取
# =========================
def auto_find_csv(start: Path) -> Path | None:
    """在项目目录内递归搜索最可能的数据集 CSV。"""
    patterns = ["*GraduateEmploymentSurvey*.csv", "*.csv"]
    for pat in patterns:
        candidates = sorted(
            [p for p in start.rglob(pat) if ".venv" not in p.parts and ".git" not in p.parts],
            key=lambda p: (0 if ("data" in p.parts and "raw" in p.parts) else 1, len(str(p)))
        )
        if candidates:
            return candidates[0]
    return None


def load_csv(csv_path: Path) -> pd.DataFrame:
    """安全读取 CSV，兼容 utf-8-sig。"""
    try:
        try:
            return pd.read_csv(csv_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] CSV not found at: {csv_path}")
        raise
    except Exception as e:
        print(f"[ERROR] Failed reading CSV: {e}")
        traceback.print_exc()
        raise


# =========================
# 清洗辅助（仅做最小必要转换，保证 1.1 图表可用）
# =========================
def _to_percent_number(x):
    """'97.5%' or '97.5 %' -> 97.5 ；其他失败返回 NaN。"""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().replace(" ", "")
    if s.endswith("%"):
        s = s[:-1]
    return pd.to_numeric(s, errors="coerce")


def _to_money_number(x):
    """'$4,200' 或 '4,200' 或 '4200' -> 4200.0；其他失败 NaN。"""
    if pd.isna(x):
        return pd.NA
    s = str(x).replace("$", "").replace(",", "").strip()
    return pd.to_numeric(s, errors="coerce")


def coerce_numeric_for_plots(df: pd.DataFrame) -> pd.DataFrame:
    """
    为了 1.1 能正确作图，对明显的百分比与金额列做轻量转换（保持百分制）。
    不做 0–1 归一；正式标准化留给 Section 1.2。
    """
    out = df.copy()
    # 百分比列（保持 0-100 标度）
    pct_cols = [c for c in out.columns if c.lower() in {"employment_rate_overall", "employment_rate_ft_perm"}]
    for c in pct_cols:
        out[c] = out[c].apply(_to_percent_number)

    # 金额列
    money_cols = [
        "basic_monthly_mean", "basic_monthly_median",
        "gross_monthly_mean", "gross_monthly_median",
        "gross_mthly_25_percentile", "gross_mthly_75_percentile",
    ]
    for c in money_cols:
        if c in out.columns:
            out[c] = out[c].apply(_to_money_number)

    # year 必须是数值
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce")

    return out


# =========================
# 表格输出
# =========================
def save_table(df: pd.DataFrame, name: str) -> None:
    """保存为 CSV；Markdown 可选（报告里一般只用 CSV/图）。"""
    (TABLE_DIR / f"{name}.csv").write_text(df.to_csv(index=True), encoding="utf-8")


def structural_summary(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
    }


def descriptive_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num_desc = df.select_dtypes(include="number").describe().T
    cat_desc = df.select_dtypes(include=["object", "category"]).describe().T
    return num_desc, cat_desc


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rpt = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().sum() / max(total, 1) * 100).round(2),
        "n_unique": df.nunique(dropna=True)
    })
    # 附注总内存（放到特殊行）
    rpt.loc["__memory_usage_bytes__", "dtype"] = str(int(df.memory_usage(deep=True).sum()))
    return rpt.sort_values(by=["missing_pct", "n_unique"], ascending=[False, True])


# =========================
# 画图（单图单文件）
# =========================
def save_hist(df: pd.DataFrame, col: str, title: str, xlabel: str):
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return
    plt.figure()
    df[col].dropna().plot(kind="hist", bins=20, title=title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"hist_{col}.png", dpi=160)
    plt.close()


def save_box_by(df: pd.DataFrame, value_col: str, by_col: str, title: str, ylabel: str):
    if value_col not in df.columns or by_col not in df.columns:
        return
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return
    # 只取非空
    tmp = df[[value_col, by_col]].dropna()
    if tmp.empty:
        return
    plt.figure()
    # pandas 的 boxplot 支持 by 分组，会生成单张图
    tmp.boxplot(column=value_col, by=by_col, rot=45)
    plt.title(title)
    plt.suptitle("")  # 去掉自动的子标题
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"box_{value_col}_by_{by_col}.png", dpi=160)
    plt.close()


def save_trend_over_time(df: pd.DataFrame, y_col: str, x_col: str = "year", agg: str = "median", title_prefix: str = "Median"):
    if x_col not in df.columns or y_col not in df.columns:
        return
    if not pd.api.types.is_numeric_dtype(df[x_col]):
        return
    ser_x = pd.to_numeric(df[x_col], errors="coerce")
    ser_y = pd.to_numeric(df[y_col], errors="coerce")
    tmp = pd.DataFrame({x_col: ser_x, y_col: ser_y}).dropna()
    if tmp.empty:
        return
    if agg == "median":
        ts = tmp.groupby(x_col)[y_col].median()
    elif agg == "mean":
        ts = tmp.groupby(x_col)[y_col].mean()
    else:
        raise ValueError("agg must be 'median' or 'mean'")
    ts = ts.sort_index()

    plt.figure()
    ts.plot(kind="line", marker="o", title=f"{title_prefix} {y_col.replace('_', ' ').title()} over time")
    plt.xlabel(x_col)
    plt.ylabel(y_col.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"trend_{y_col}_over_time.png", dpi=160)
    plt.close()


def save_topn_freq(df: pd.DataFrame, col: str, top_n: int = 10):
    if col not in df.columns:
        return
    vc = df[col].astype("string").value_counts(dropna=False).head(top_n)
    out = vc.to_frame(name="count")
    save_table(out, f"top10_freq_{col}")


# =========================
# 主流程
# =========================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to dataset CSV (optional).")
    args = parser.parse_args()

    default_csv = PROJECT_ROOT / "data" / "raw" / "7-GraduateEmploymentSurveyNTUNUSSITSMUSUSSSUTD.csv"
    if args.csv:
        csv_path = Path(args.csv)
        print(f"[INFO] Using --csv: {csv_path}")
    elif default_csv.exists():
        csv_path = default_csv
        print(f"[INFO] Using default CSV at: {csv_path}")
    else:
        print("[WARN] Default CSV not found. Auto-detecting CSV in project...")
        auto = auto_find_csv(PROJECT_ROOT)
        if auto is None:
            print("[ERROR] Could not find any CSV. Provide --csv PATH.")
            return 2
        csv_path = auto
        print(f"[INFO] Auto-detected CSV: {csv_path}")

    # 读取
    df_raw = load_csv(csv_path)

    # 结构概览 + 保存
    summary = structural_summary(df_raw)
    print("[INFO] Structural summary:", summary)

    # 描述性统计（原始表）
    num_desc, cat_desc = descriptive_stats(df_raw)
    save_table(num_desc, "describe_numeric")
    save_table(cat_desc, "describe_categorical")

    # 数据质量报告（原始表）
    dq = data_quality_report(df_raw)
    save_table(dq, "data_quality_report")

    # 预览前 20 行（报告可引用）
    df_raw.head(20).to_csv(PROJECT_ROOT / "preview_head20.csv", index=False, encoding="utf-8")

    # 为作图做轻量转换（仅 1.1 可视化使用）
    df_plot = coerce_numeric_for_plots(df_raw)

    # —— 直方图（关键数值列）——
    for col, title, xlabel in [
        ("year", "Year Distribution", "Year"),
        ("employment_rate_overall", "Overall Employment Rate Distribution", "Employment Rate (%)"),
        ("employment_rate_ft_perm", "Full-time Permanent Employment Rate Distribution", "Employment Rate (%)"),
        ("basic_monthly_mean", "Basic Monthly Mean Salary Distribution", "Salary (SGD)"),
        ("gross_monthly_mean", "Gross Monthly Mean Salary Distribution", "Salary (SGD)"),
        ("gross_monthly_median", "Gross Monthly Median Salary Distribution", "Salary (SGD)"),
    ]:
        if col in df_plot.columns:
            save_hist(df_plot, col, title, xlabel)

    # —— 类别列 Top10 频数（保存为表格，报告可选用）——
    for col in df_raw.select_dtypes(include=["object", "category"]).columns.tolist():
        save_topn_freq(df_raw, col, top_n=10)

    # —— 箱线图（按大学分组）——
    for val_col, ylabel in [
        ("employment_rate_overall", "Employment Rate (%)"),
        ("basic_monthly_mean", "Salary (SGD)"),
        ("gross_monthly_median", "Salary (SGD)"),
    ]:
        if "university" in df_plot.columns and val_col in df_plot.columns:
            save_box_by(df_plot, val_col, "university", f"{val_col.replace('_',' ').title()} by University", ylabel)

    # —— 趋势线（按年份聚合）——
    for y_col in ["gross_monthly_median", "basic_monthly_median", "employment_rate_overall"]:
        if y_col in df_plot.columns:
            save_trend_over_time(df_plot, y_col, "year", agg="median", title_prefix="Median")

    print("[DONE] Section 1.1 EDA complete.")
    print(f" - Input : {csv_path}")
    print(f" - Tables: {TABLE_DIR.resolve()}")
    print(f" - Figures: {FIG_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
