#!/usr/bin/env python3
import csv
import gzip
import re
import urllib.request
from pathlib import Path

URL = "https://dumps.wikimedia.org/bnwiki/latest/bnwiki-latest-all-titles-in-ns0.gz"
RAW_FILE = "bnwiki-latest-all-titles-in-ns0.gz"
OUTPUT_CSV = "wiki_page_titles.csv"

BENGALI_PATTERN = re.compile(r"[ঀ-৿]")


def download_gzip(url: str, path: str) -> None:
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"Downloaded: {path}")


def is_bengali(text: str) -> bool:
    return bool(BENGALI_PATTERN.search(text))


def clean_title(title: str) -> str:
    """Strip the emoji prefix from wiki dump titles.

    The dump format is either:
      "emoji_text"_BanglaTitle   -> returns the suffix after _
      "BanglaTitle"              -> returns the inner text
    """
    if not title.startswith('"'):
        return title

    # Pattern: "something"_rest
    m = re.match(r'^"([^"]*)"\_(.+)$', title)
    if m:
        return m.group(2)  # suffix after the underscore

    # Pattern: "BanglaText" (fully quoted)
    m2 = re.match(r'^"(.+)"$', title)
    if m2:
        return m2.group(1)

    return title.strip('"')


def filter_and_write_csv(gz_path: str, csv_path: str) -> None:
    print(f"Unzipping {gz_path} and filtering Bengali titles ...")
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        with open(csv_path, "w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["page_title"])
            for row in reader:
                if not row:
                    continue
                raw = row[0]
                clean = clean_title(raw)
                if is_bengali(clean):
                    writer.writerow([clean])


def main() -> None:
    work_dir = Path(__file__).parent

    gz_path = work_dir / RAW_FILE
    download_gzip(URL, str(gz_path))

    csv_path = work_dir / OUTPUT_CSV
    filter_and_write_csv(str(gz_path), str(csv_path))

    gz_path.unlink()
    print(f"Done. Output: {csv_path} ({csv_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
