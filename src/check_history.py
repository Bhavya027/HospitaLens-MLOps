
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

files = list(DATA_DIR.glob("*.csv"))
if len(files) != 1:
    raise SystemExit(
        f"Expected one CSV in {DATA_DIR}, found {len(files)}"
    )

file_path = files[0]

columns = [
    "hospital_pk",
    "collection_week",
    "inpatient_beds_used_7_day_avg",
    "inpatient_beds_7_day_coverage",
]

print("Reading historical bed data in chunks...")

parts = []

for chunk in pd.read_csv(
    file_path,
    usecols=columns,
    chunksize=100_000,
    low_memory=False,
):
    # Convert reported bed usage to numeric values
    chunk["inpatient_beds_used_7_day_avg"] = pd.to_numeric(
        chunk["inpatient_beds_used_7_day_avg"]
        .astype(str)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )

    chunk["collection_week"] = pd.to_datetime(
        chunk["collection_week"], errors="coerce"
    )

    # Exclude common sentinel values from the target
    target = chunk["inpatient_beds_used_7_day_avg"]
    chunk.loc[target.isin([-999999, -99999, -9999]), 
              "inpatient_beds_used_7_day_avg"] = float("nan")

    parts.append(
        chunk.dropna(
            subset=[
                "hospital_pk",
                "collection_week",
                "inpatient_beds_used_7_day_avg",
            ]
        )
    )

df = pd.concat(parts, ignore_index=True)

# Sort each hospital's records chronologically
df = df.sort_values(["hospital_pk", "collection_week"])

# Count usable target observations per hospital
history = df.groupby("hospital_pk").agg(
    first_week=("collection_week", "min"),
    last_week=("collection_week", "max"),
    usable_weeks=("collection_week", "size"),
)

print("\n========== HISTORY CHECK ==========")
print(f"Hospitals with valid target records: {len(history):,}")
print(f"Valid target records: {len(df):,}")
print(f"Median usable weeks per hospital: {history['usable_weeks'].median():.0f}")
print(f"Hospitals with 20+ observations: {(history['usable_weeks'] >= 20).sum():,}")
print(f"Hospitals with 50+ observations: {(history['usable_weeks'] >= 50).sum():,}")
print(f"Hospitals with 100+ observations: {(history['usable_weeks'] >= 100).sum():,}")

# Check gaps between reported weeks
gaps = (
    df.groupby("hospital_pk")["collection_week"]
    .diff()
    .dt.days
    .dropna()
)

print(f"Adjacent observations exactly 7 days apart: {(gaps == 7).mean():.2%}")
print(f"Median days between adjacent observations: {gaps.median():.0f}")

print("\nHistory check completed.")
