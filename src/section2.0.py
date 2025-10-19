"""
section2.0.py

COMP0035 Coursework 1 — Section 2.2
- Create a normalised SQLite database based on the prepared dataset
- Tables: university, school, degree, survey_year, employment_stats
- Insert records with foreign keys
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import pandas as pd


DB_PATH = Path(__file__).resolve().parents[1] / "ntu_employment.sqlite"
CLEAN_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "ntu_employment_clean.csv"


DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS university (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS school (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    university_id INTEGER NOT NULL,
    UNIQUE(name, university_id),
    FOREIGN KEY (university_id) REFERENCES university(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS degree (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS survey_year (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS employment_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    university_id INTEGER NOT NULL,
    school_id INTEGER NOT NULL,
    degree_id INTEGER NOT NULL,
    survey_year_id INTEGER NOT NULL,
    employment_rate_overall REAL,
    employment_rate_ft_perm REAL,
    basic_monthly_mean REAL,
    basic_monthly_median REAL,
    gross_monthly_mean REAL,
    gross_monthly_median REAL,
    gross_mthly_25_percentile REAL,
    gross_mthly_75_percentile REAL,
    FOREIGN KEY (university_id) REFERENCES university(id) ON DELETE CASCADE,
    FOREIGN KEY (school_id) REFERENCES school(id) ON DELETE CASCADE,
    FOREIGN KEY (degree_id) REFERENCES degree(id) ON DELETE CASCADE,
    FOREIGN KEY (survey_year_id) REFERENCES survey_year(id) ON DELETE CASCADE
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.commit()


def upsert_lookup(conn: sqlite3.Connection, table: str, name: str, extra: dict | None = None) -> int:
    """
    Insert-or-select a lookup row, returning its id.
    """
    if extra is None:
        extra = {}
    cur = conn.cursor()
    if table == "university":
        cur.execute("INSERT OR IGNORE INTO university(name) VALUES (?)", (name,))
        conn.commit()
        cur.execute("SELECT id FROM university WHERE name = ?", (name,))
        return cur.fetchone()[0]
    elif table == "degree":
        cur.execute("INSERT OR IGNORE INTO degree(name) VALUES (?)", (name,))
        conn.commit()
        cur.execute("SELECT id FROM degree WHERE name = ?", (name,))
        return cur.fetchone()[0]
    elif table == "school":
        # needs university_id
        uni_id = extra["university_id"]
        cur.execute("INSERT OR IGNORE INTO school(name, university_id) VALUES (?, ?)", (name, uni_id))
        conn.commit()
        cur.execute("SELECT id FROM school WHERE name = ? AND university_id = ?", (name, uni_id))
        return cur.fetchone()[0]
    elif table == "survey_year":
        y = extra["year"]
        cur.execute("INSERT OR IGNORE INTO survey_year(year) VALUES (?)", (y,))
        conn.commit()
        cur.execute("SELECT id FROM survey_year WHERE year = ?", (y,))
        return cur.fetchone()[0]
    else:
        raise ValueError(f"Unknown table: {table}")


def load_clean_dataframe() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV)


def insert_facts(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """
    Insert rows into employment_stats with FK references.
    """
    cur = conn.cursor()
    inserted = 0
    for _, row in df.iterrows():
        uni_id = upsert_lookup(conn, "university", str(row["university"]))
        school_id = upsert_lookup(conn, "school", str(row["school"]), extra={"university_id": uni_id})
        degree_id = upsert_lookup(conn, "degree", str(row["degree"]))
        year_id = upsert_lookup(conn, "survey_year", str(row["year"]), extra={"year": int(row["year"])})

        cur.execute(
            """
            INSERT INTO employment_stats(
                university_id, school_id, degree_id, survey_year_id,
                employment_rate_overall, employment_rate_ft_perm,
                basic_monthly_mean, basic_monthly_median,
                gross_monthly_mean, gross_monthly_median,
                gross_mthly_25_percentile, gross_mthly_75_percentile
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uni_id, school_id, degree_id, year_id,
                row.get("employment_rate_overall", None),
                row.get("employment_rate_ft_perm", None),
                row.get("basic_monthly_mean", None),
                row.get("basic_monthly_median", None),
                row.get("gross_monthly_mean", None),
                row.get("gross_monthly_median", None),
                row.get("gross_mthly_25_percentile", None),
                row.get("gross_mthly_75_percentile", None),
            )
        )
        inserted += 1

    conn.commit()
    return inserted


def main() -> int:
    df = load_clean_dataframe()

    with sqlite3.connect(DB_PATH) as conn:
        ensure_schema(conn)
        inserted = insert_facts(conn, df)

        # simple sanity checks
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM employment_stats")
        n = cursor.fetchone()[0]
        print(f"[INFO] employment_stats rows: {n} (inserted {inserted})")

        # counts for lookups
        for t in ["university","school","degree","survey_year"]:
            cursor.execute(f"SELECT COUNT(*) FROM {t}")
            print(f"[INFO] {t} rows:", cursor.fetchone()[0])

    print("[DONE] Database created:", DB_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
