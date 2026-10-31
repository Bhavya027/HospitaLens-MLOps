
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

files = list(DATA_DIR.glob("*.csv"))

if len(files) != 1:
    raise SystemExit(
        f"Expected exactly one CSV in {DATA_DIR}, found {len(files)}"
    )

file_path = files[0]

# Only load the columns needed for this audit
columns = [
    "hospital_pk",
    "collection_week",
    "state",
    "hospital_name",
    "total_beds_7_day_avg",
    "inpatient_beds_used_7_day_avg",
    "inpatient_beds_7_day_coverage",
    "all_adult_hospital_inpatient_beds_7_day_avg",
    "all_adult_hospital_inpatient_bed_occupied_7_day_avg",
    "all_adult_hospital_inpatient_bed_occupied_7_day_coverage",
]

print(f"Dataset: {file_path.name}")
print("Reading selected columns in chunks. This may take a while...")

total_rows = 0
date_min = None
date_max = None
missing_counts = {column: 0 for column in columns}
sentinel_counts = {column: 0 for column in columns}
numeric_stats = {}
duplicate_keys = 0
previous_keys = set()

for chunk in pd.read_csv(
    file_path,
    usecols=columns,
    chunksize=100_000,
    low_memory=False,
):
    total_rows += len(chunk)

    dates = pd.to_datetime(
        chunk["collection_week"], errors="coerce"
    )
    chunk_min = dates.min()
    chunk_max = dates.max()

    if pd.notna(chunk_min):
        date_min = chunk_min if date_min is None else min(date_min, chunk_min)
    if pd.notna(chunk_max):
        date_max = chunk_max if date_max is None else max(date_max, chunk_max)

    for column in columns:
        missing_counts[column] += chunk[column].isna().sum()

    # Check possible reporting placeholders in numeric fields
    numeric_columns = [
        "total_beds_7_day_avg",
        "inpatient_beds_used_7_day_avg",
        "all_adult_hospital_inpatient_beds_7_day_avg",
        "all_adult_hospital_inpatient_bed_occupied_7_day_avg",
    ]

    for column in numeric_columns:
        values = pd.to_numeric(
            chunk[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

        sentinel_counts[column] = sentinel_counts.get(column, 0) + (
            values.isin([-999999, -99999, -9999]).sum()
        )

        valid = values[~values.isin([-999999, -99999, -9999])]
        valid = valid.dropna()

        if not valid.empty:
            if column not in numeric_stats:
                numeric_stats[column] = {
                    "min": valid.min(),
                    "max": valid.max(),
                    "valid_count": len(valid),
                }
            else:
                stats = numeric_stats[column]
                stats["min"] = min(stats["min"], valid.min())
                stats["max"] = max(stats["max"], valid.max())
                stats["valid_count"] += len(valid)

    # Count duplicate hospital-week keys within the file
    keys = chunk[["hospital_pk", "collection_week"]].astype(str)
    for key in map(tuple, keys.to_numpy()):
        if key in previous_keys:
            duplicate_keys += 1
        else:
            previous_keys.add(key)

print("\n========== DATA AUDIT ==========")
print(f"Total rows: {total_rows:,}")
print(f"Date range: {date_min} to {date_max}")
print(f"Duplicate hospital-week keys: {duplicate_keys:,}")

print("\nMissing values:")
for column, count in missing_counts.items():
    print(f"{column}: {count:,} ({count / total_rows:.2%})")

print("\nPossible sentinel values:")
for column, count in sentinel_counts.items():
    print(f"{column}: {count:,}")

print("\nValid numeric ranges (after excluding checked sentinels):")
for column, stats in numeric_stats.items():
    print(
        f"{column}: min={stats['min']}, max={stats['max']}, "
        f"valid_count={stats['valid_count']:,}"
    )

print("\nAudit completed.")
