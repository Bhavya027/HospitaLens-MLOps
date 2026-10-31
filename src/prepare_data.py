
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = list(DATA_DIR.glob("*.csv"))
if len(files) != 1:
    raise SystemExit(f"Expected one original CSV, found {len(files)}")

source_path = files[0]

columns = [
    "hospital_pk",
    "collection_week",
    "state",
    "total_beds_7_day_avg",
    "inpatient_beds_used_7_day_avg",
]

print(f"Reading original dataset: {source_path.name}")

parts = []

for chunk in pd.read_csv(
    source_path,
    usecols=columns,
    chunksize=100_000,
    low_memory=False,
):
    chunk["collection_week"] = pd.to_datetime(
        chunk["collection_week"], errors="coerce"
    )

    for column in [
        "total_beds_7_day_avg",
        "inpatient_beds_used_7_day_avg",
    ]:
        chunk[column] = pd.to_numeric(
            chunk[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        chunk.loc[
            chunk[column].isin([-999999, -99999, -9999]),
            column,
        ] = float("nan")

    chunk = chunk.dropna(
        subset=["hospital_pk", "collection_week",
                "inpatient_beds_used_7_day_avg"]
    )
    chunk = chunk[chunk["inpatient_beds_used_7_day_avg"] >= 0]

    # Capacity must be positive to calculate occupancy percentage.
    chunk.loc[chunk["total_beds_7_day_avg"] <= 0,
              "total_beds_7_day_avg"] = float("nan")

    parts.append(chunk)

df = pd.concat(parts, ignore_index=True)
df = df.drop_duplicates(
    subset=["hospital_pk", "collection_week"], keep="last"
)
df = df.sort_values(
    ["hospital_pk", "collection_week"]
).reset_index(drop=True)

grouped = df.groupby("hospital_pk", sort=False)

# Historical occupancy features
df["previous_occupancy_beds"] = grouped[
    "inpatient_beds_used_7_day_avg"
].shift(1)

df["two_weeks_ago_occupancy_beds"] = grouped[
    "inpatient_beds_used_7_day_avg"
].shift(2)

previous_date = grouped["collection_week"].shift(1)
two_weeks_ago_date = grouped["collection_week"].shift(2)

# Only use lag values when reporting weeks are exactly consecutive.
df.loc[
    (df["collection_week"] - previous_date).dt.days != 7,
    "previous_occupancy_beds",
] = float("nan")

df.loc[
    (df["collection_week"] - two_weeks_ago_date).dt.days != 14,
    "two_weeks_ago_occupancy_beds",
] = float("nan")

# Target: same hospital's occupancy exactly seven days later.
df["target_date"] = grouped["collection_week"].shift(-1)
df["target_occupancy_beds"] = grouped[
    "inpatient_beds_used_7_day_avg"
].shift(-1)

df["days_to_target"] = (
    df["target_date"] - df["collection_week"]
).dt.days

df = df[df["days_to_target"] == 7].copy()

df["current_occupancy_beds"] = df[
    "inpatient_beds_used_7_day_avg"
]

df["current_occupancy_rate"] = (
    df["current_occupancy_beds"] / df["total_beds_7_day_avg"]
)

# Remove impossible proportions caused by inconsistent source records.
df.loc[
    (df["current_occupancy_rate"] < 0)
    | (df["current_occupancy_rate"] > 1.5),
    "current_occupancy_rate",
] = float("nan")

df = df.rename(columns={"collection_week": "feature_date"})

output_columns = [
    "hospital_pk",
    "state",
    "feature_date",
    "target_date",
    "current_occupancy_beds",
    "previous_occupancy_beds",
    "two_weeks_ago_occupancy_beds",
    "total_beds_7_day_avg",
    "current_occupancy_rate",
    "target_occupancy_beds",
]

output_path = OUTPUT_DIR / "hospital_weekly_features.csv"
df[output_columns].to_csv(output_path, index=False)

print("\n========== UPDATED PREPARATION RESULTS ==========")
print(f"Usable next-week rows: {len(df):,}")
print(f"Unique hospitals: {df['hospital_pk'].nunique():,}")
print(f"Feature date range: {df['feature_date'].min()} to {df['feature_date'].max()}")
print(f"Target date range: {df['target_date'].min()} to {df['target_date'].max()}")
print("\nMissing values in features:")
print(df[output_columns].isna().sum().to_string())
print(f"\nSaved: {output_path}")
