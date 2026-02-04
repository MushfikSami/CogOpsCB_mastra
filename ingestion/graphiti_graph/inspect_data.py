import csv
import re
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Extract failed rows from logs into a new CSV.")
    parser.add_argument("source_csv", type=str, help="Path to the original source CSV file.")
    parser.add_argument("log_file", type=str, help="Path to the log file containing failures.")
    parser.add_argument("--output", type=str, default="failed.csv", help="Name of the output file (default: failed.csv)")
    
    args = parser.parse_args()

    # 1. Validate Files
    if not os.path.exists(args.source_csv):
        print(f"Error: Source CSV '{args.source_csv}' not found.")
        sys.exit(1)
    if not os.path.exists(args.log_file):
        print(f"Error: Log file '{args.log_file}' not found.")
        sys.exit(1)

    # 2. Scan Log for Failed Indices
    print(f"📖 Scanning log file: {args.log_file}...")
    failed_indices = set()
    
    # Regex to capture the number inside "❌ [Row 123]"
    pattern = re.compile(r"❌\s*\[Row\s+(\d+)\]")

    with open(args.log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "❌" in line:
                match = pattern.search(line)
                if match:
                    idx = int(match.group(1))
                    failed_indices.add(idx)

    if not failed_indices:
        print("✅ No failure markers (❌) found in the log file.")
        sys.exit(0)

    print(f"🔍 Found {len(failed_indices)} unique failed rows.")

    # 3. Read Source CSV and Filter
    print(f"📂 Reading source CSV: {args.source_csv}...")
    
    failed_data = []
    headers = []

    try:
        with open(args.source_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            for index, row in enumerate(reader):
                # If this index was found in the logs, keep it
                if index in failed_indices:
                    failed_data.append(row)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # 4. Write to failed.csv
    print(f"💾 Writing {len(failed_data)} rows to {args.output}...")
    
    try:
        with open(args.output, mode='w', encoding='utf-8', newline='') as f:
            if headers:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(failed_data)
            else:
                print("Warning: No headers found in source CSV.")
    except Exception as e:
        print(f"Error writing output CSV: {e}")
        sys.exit(1)

    print(f"✅ Done! Failed rows saved to: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()