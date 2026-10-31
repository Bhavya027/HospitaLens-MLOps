
import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

csv_files = list(data_dir.glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV file found in the data folder.")
    print(f"Expected folder: {data_dir}")
    raise SystemExit(1)

print("CSV files found:")
for i, file in enumerate(csv_files, start=1):
    print(f"{i}. {file.name} ({file.stat().st_size / (1024**2):.2f} MB)")

if len(csv_files) != 1:
    print("\nKeep only the intended dataset in this folder for now.")
    raise SystemExit(1)

file_path = csv_files[0]

# Read only five rows first; do not load the entire file
sample = pd.read_csv(file_path, nrows=5, low_memory=False)

print("\nCOLUMN NAMES:")
for column in sample.columns:
    print(column)

print("\nDATA TYPES:")
print(sample.dtypes.to_string())

print("\nFIRST FIVE ROWS:")
print(sample.to_string(index=False))

print("\nFile inspection completed.")
