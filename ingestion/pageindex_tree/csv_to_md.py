import argparse
import os
import pandas as pd


def convert_csv_to_markdown(csv_path):
    print(f"Reading data from {csv_path}...")

    # Load the CSV, handling any missing values
    df = pd.read_csv(csv_path)
    df = df.fillna("")

    markdown_content = ""

    # Track the current hierarchy to avoid repeating headings
    current_cat = ""
    current_subcat = ""
    current_service = ""

    for index, row in df.iterrows():
        cat = str(row.get('category', '')).strip()
        subcat = str(row.get('sub_category', '')).strip()
        service = str(row.get('service', '')).strip()
        topic = str(row.get('topic', '')).strip()
        text = str(row.get('text', '')).strip()
        url = str(row.get('url', '')).strip()
        passage_id = str(row.get('passage_id', '')).strip()

        # Level 1: Category
        if cat and cat != current_cat:
            markdown_content += f"\n# {cat}\n"
            current_cat = cat
            current_subcat = ""
            current_service = ""

        # Level 2: Sub-Category
        if subcat and subcat != current_subcat:
            markdown_content += f"\n## {subcat}\n"
            current_subcat = subcat
            current_service = ""

        # Level 3: Service
        if service and service != current_service:
            markdown_content += f"\n### {service}\n"
            current_service = service

        # Level 4: Topic
        if topic:
            markdown_content += f"\n#### {topic}\n\n"

        # Add passage_id and url metadata
        if passage_id or url:
            markdown_content += "**Passage ID:** " + passage_id
            if url:
                markdown_content += "\n**URL:** [" + url + "](" + url + ")\n"
            else:
                markdown_content += "\n"
            markdown_content += "\n"

        # Add the core text
        if text:
            # Replace literal \n in the CSV with actual markdown line breaks
            formatted_text = text.replace("\\n", "\n")
            markdown_content += f"{formatted_text}\n\n"

        # Add a visual separator between entries
        markdown_content += "---\n"

    # Save with UTF-8 encoding
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"✅ Successfully converted to Markdown! Saved as: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CSV to Markdown with hierarchical structure"
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file"
    )
    args = parser.parse_args()

    # Get the directory and filename from input
    input_dir = os.path.dirname(args.input_csv)
    input_filename = os.path.basename(args.input_csv)
    # Replace .csv with .md
    output_filename = os.path.splitext(input_filename)[0] + ".md"
    output_path = os.path.join(input_dir, output_filename)

    convert_csv_to_markdown(args.input_csv)
