import argparse
import os
import pandas as pd


def convert_csv_to_markdown(csv_path, output_path=None):
    print(f"Reading data from {csv_path}...")

    # Load the CSV, handling any missing values
    df = pd.read_csv(csv_path)
    df = df.fillna("")

    if output_path is None:
        input_dir = os.path.dirname(csv_path)
        input_filename = os.path.basename(csv_path)
        output_path = os.path.join(input_dir, os.path.splitext(input_filename)[0] + ".md")

    markdown_content = ""

    # Track the current hierarchy to avoid repeating headings
    current_cat = ""
    current_subcat = ""
    current_service = ""

    for _, row in df.iterrows():
        uuid = str(row.get('uuid', '')).strip()
        cat = str(row.get('category', '')).strip()
        subcat = str(row.get('sub_category', '')).strip()
        service = str(row.get('service', '')).strip()
        topic = str(row.get('topic', '')).strip()
        text = str(row.get('text', '')).strip()
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

        # Add uuid and passage_id metadata
        if uuid:
            markdown_content += f"**UUID:** {uuid}\n"
        if passage_id:
            markdown_content += f"**Passage ID:** {passage_id}\n"
        if uuid or passage_id:
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

    print(f"Successfully converted to Markdown! Saved as: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CSV to Markdown with hierarchical structure"
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output Markdown file path (default: same directory with .md extension)"
    )
    args = parser.parse_args()

    convert_csv_to_markdown(args.input_csv, args.output)
