"""
COMP0035 Coursework 1 — Section 2.0 Database Creation (SQLite).

This script builds a normalised SQLite database from the cleaned dataset
produced in Section 1.2 (data/processed/clean_full.csv).

Schema (3NF-ish):
- university(id, name UNIQUE)
- school(id, name, university_id, UNIQUE(name, university_id))
- degree(id, name UNIQUE)
- survey_year(id, year UNIQUE)
- employment_stats(
      id, university_id, school_id, degree_id, year_id,
      employment_rate_overall, employment_rate_ft_perm,
      basic_monthly_mean, basic_monthly_median,
      gross_monthly_mean, gross_monthly_median
  )

Foreign keys are enforced and basic indexes are created for joins/filters.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "clean_full.csv"
DB_PATH = PROJECT_ROOT / "ntu_employment.sqlite"


# ---------- SQL DDL ----------
DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS university (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS degree (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS survey_year (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS school (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT NOT NULL,
    university_id  INTEGER NOT NULL,
    UNIQUE (name, university_id),
    FOREIGN KEY (university_id) REFERENCES university(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS employment_stats (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    university_id             INTEGER NOT NULL,
    school_id                 INTEGER NOT NULL,
    degree_id                 INTEGER NOT NULL,
    year_id                   INTEGER NOT NULL,

    employment_rate_overall   REAL,
    employment_rate_ft_perm   REAL,
    basic_monthly_mean        REAL,
    basic_monthly_median      REAL,
    gross_monthly_mean        REAL,
    gross_monthly_median      REAL,

    FOREIGN KEY (university_id) REFERENCES university(id) ON DELETE RESTRICT,
    FOREIGN KEY (school_id)     REFERENCES school(id)      ON DELETE RESTRICT,
    FOREIGN KEY (degree_id)     REFERENCES degree(id)      ON DELETE RESTRICT,
    FOREIGN KEY (year_id)       REFERENCES survey_year(id) ON DELETE RESTRICT
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_emp_university ON employment_stats(university_id);
CREATE INDEX IF NOT EXISTS idx_emp_school     ON employment_stats(school_id);
CREATE INDEX IF NOT EXISTS idx_emp_degree     ON employment_stats(degree_id);
CREATE INDEX IF NOT EXISTS idx_emp_year       ON employment_stats(year_id);
"""


# ---------- Helpers ----------
def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes if they do not exist."""
    conn.executescript(DDL)
    conn.commit()


def upsert_lookup(
    conn: sqlite3.Connection,
    table: str,
    name: str,
    *,
    extra: dict | None = None,
) -> int:
    """
    Insert-or-select a lookup row, returning its integer primary key.

    Special rules:
    - table = 'school' requires extra={'university_id': int} and uses
      UNIQUE(name, university_id).
    - table = 'survey_year' accepts extra={'year': int} to insert year value.
    """
    cur = conn.cursor()
    extra = extra or {}

    if table == "university":
        cur.execute("INSERT OR IGNORE INTO university(name) VALUES (?)", (name,))
        conn.commit()
        cur.execute("SELECT id FROM university WHERE name = ?", (name,))
        return int(cur.fetchone()[0])

    if table == "degree":
        cur.execute("INSERT OR IGNORE INTO degree(name) VALUES (?)", (name,))
        conn.commit()
        cur.execute("SELECT id FROM degree WHERE name = ?", (name,))
        return int(cur.fetchone()[0])

    if table == "survey_year":
        # year can be the same as name if name is a numeric string; prefer 'year' key.
        year_val = int(extra.get("year", int(str(name))))
        cur.execute("INSERT OR IGNORE INTO survey_year(year) VALUES (?)", (year_val,))
        conn.commit()
        cur.execute("SELECT id FROM survey_year WHERE year = ?", (year_val,))
        return int(cur.fetchone()[0])

    if table == "school":
        if "university_id" not in extra:
            raise ValueError("school requires extra={'university_id': <int>}")
        uni_id = int(extra["university_id"])
        cur.execute(
            "INSERT OR IGNORE INTO school(name, university_id) VALUES (?, ?)",
            (name, uni_id),
        )
        conn.commit()
        cur.execute(
            "SELECT id FROM school WHERE name = ? AND university_id = ?",
            (name, uni_id),
        )
        return int(cur.fetchone()[0])

    raise ValueError(f"Unsupported lookup table: {table}")


def load_clean_dataframe() -> pd.DataFrame:
    """Load the cleaned dataset from Section 1.2."""
    try:
        return pd.read_csv(CLEAN_CSV, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(CLEAN_CSV)


def insert_facts(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert rows into employment_stats with proper foreign key references."""
    cur = conn.cursor()
    inserted = 0

    # Required columns
    required = {
        "university",
        "school",
        "degree",
        "year",
        "employment_rate_overall",
        "employment_rate_ft_perm",
        "basic_monthly_mean",
        "basic_monthly_median",
        "gross_monthly_mean",
        "gross_monthly_median",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in cleaned CSV: {sorted(missing)}")

    # Iterate rows and insert
    for _, row in df.iterrows():
        uni_name = str(row["university"]) if pd.notna(row["university"]) else "Unknown"
        school_name = str(row["school"]) if pd.notna(row["school"]) else "Unknown"
        degree_name = str(row["degree"]) if pd.notna(row["degree"]) else "Unknown"

        # Convert year to int if possible
        try:
            year_val = int(row["year"])
        except Exception:
            continue  # skip if no valid year

        # Lookup keys
        uni_id = upsert_lookup(conn, "university", uni_name)
        school_id = upsert_lookup(
            conn,
            "school",
            school_name,
            extra={"university_id": uni_id},
        )
        degree_id = upsert_lookup(conn, "degree", degree_name)
        year_id = upsert_lookup(conn, "survey_year", str(year_val), extra={"year": year_val})

        # Metric values (allow NaN -> None)
        def nz(x):
            return None if pd.isna(x) else float(x)

        cur.execute(
            """
            INSERT INTO employment_stats(
                university_id, school_id, degree_id, year_id,
                employment_rate_overall, employment_rate_ft_perm,
                basic_monthly_mean, basic_monthly_median,
                gross_monthly_mean, gross_monthly_median
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uni_id,
                school_id,
                degree_id,
                year_id,
                nz(row["employment_rate_overall"]),
                nz(row["employment_rate_ft_perm"]),
                nz(row["basic_monthly_mean"]),
                nz(row["basic_monthly_median"]),
                nz(row["gross_monthly_mean"]),
                nz(row["gross_monthly_median"]),
            ),
        )
        inserted += 1

    conn.commit()
    return inserted


# ---------- Main ----------
def main() -> int:
    """Create the SQLite database and insert facts from the cleaned CSV."""
    if not CLEAN_CSV.exists():
        print(f"Error: cleaned CSV not found at {CLEAN_CSV}")
        return 1

    df = load_clean_dataframe()
    if df.empty:
        print("Error: cleaned dataframe is empty.")
        return 1

    with sqlite3.connect(DB_PATH) as conn:
        # Enforce FKs for this session too
        conn.execute("PRAGMA foreign_keys = ON;")

        ensure_schema(conn)
        n_rows = insert_facts(conn, df)

    print("✅ Section 2.0 database creation completed.")
    print(f"   Input  : {CLEAN_CSV}")
    print(f"   Output : {DB_PATH}")
    print(f"   Inserted rows into employment_stats: {n_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
