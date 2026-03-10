import argparse
import os
import json
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pageindex.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set environment variables for local vLLM model
os.environ["OPENAI_BASE_URL"] = os.getenv('OPENAI_BASE_URL', 'http://localhost:5000/v1/')
os.environ["OPENAI_API_KEY"] = os.getenv('VLLM_API_KEY', 'no-key')
os.environ["CHATGPT_API_KEY"] = os.getenv('VLLM_API_KEY', 'no-key')


def main():
    logger.info("=" * 60)
    logger.info("PageIndex Pipeline Started")
    logger.info("=" * 60)

    parser = argparse.ArgumentParser(
        description='Process Markdown document and generate tree structure'
    )

    # Markdown input
    parser.add_argument(
        '--md_path',
        type=str,
        required=True,
        help='Path to the Markdown file'
    )

    # Output location
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save the output JSON (default: ./results)'
    )

    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        default=os.getenv('VLLM_MODEL_NAME', 'qwen35'),
        help='Model to use (default: from VLLM_MODEL_NAME in .env or qwen35)'
    )

    # Node output options
    parser.add_argument(
        '--if-add-node-id',
        type=str,
        default='yes',
        help='Whether to add node id to the node (yes/no)'
    )
    parser.add_argument(
        '--if-add-node-summary',
        type=str,
        default='yes',
        help='Whether to add summary to the node (yes/no)'
    )
    parser.add_argument(
        '--if-add-doc-description',
        type=str,
        default='no',
        help='Whether to add doc description to the doc (yes/no)'
    )
    parser.add_argument(
        '--if-add-node-text',
        type=str,
        default='no',
        help='Whether to add text to the node (yes/no)'
    )

    # Markdown-specific options
    parser.add_argument(
        '--if-thinning',
        type=str,
        default='no',
        help='Whether to apply tree thinning for markdown (yes/no)'
    )
    parser.add_argument(
        '--thinning-threshold',
        type=int,
        default=5000,
        help='Minimum token threshold for thinning (markdown only)'
    )
    parser.add_argument(
        '--summary-token-threshold',
        type=int,
        default=200,
        help='Token threshold for generating summaries (markdown only)'
    )

    args = parser.parse_args()

    # Log configuration
    logger.info(f"Input MD Path: {args.md_path}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Thinning: {args.if_thinning.lower() == 'yes'}")
    logger.info(f"Thinning Threshold: {args.thinning_threshold} tokens")
    logger.info(f"Summary Token Threshold: {args.summary_token_threshold}")
    logger.info(f"Add Node ID: {args.if_add_node_id}")
    logger.info(f"Add Node Summary: {args.if_add_node_summary}")
    logger.info(f"Add Doc Description: {args.if_add_doc_description}")
    logger.info(f"Add Node Text: {args.if_add_node_text}")
    logger.info("-" * 60)

    # Validate Markdown file
    if not args.md_path.lower().endswith(('.md', '.markdown')):
        raise ValueError("Markdown file must have .md or .markdown extension")
    if not os.path.isfile(args.md_path):
        raise ValueError(f"Markdown file not found: {args.md_path}")

    logger.info(f"Validated markdown file: {args.md_path}")

    # Load config with defaults from config.yaml
    config_loader = ConfigLoader()

    # Create options dict with user args
    user_opt = {
        'model': args.model,
        'if_add_node_summary': args.if_add_node_summary,
        'if_add_doc_description': args.if_add_doc_description,
        'if_add_node_text': args.if_add_node_text,
        'if_add_node_id': args.if_add_node_id
    }

    # Load config with defaults from config.yaml
    opt = config_loader.load(user_opt)
    logger.info("Configuration loaded from config.yaml")

    # Run the async md_to_tree function
    logger.info("Starting tree extraction...")
    toc_with_page_number = asyncio.run(md_to_tree(
        md_path=args.md_path,
        if_thinning=args.if_thinning.lower() == 'yes',
        min_token_threshold=args.thinning_threshold,
        if_add_node_summary=opt.if_add_node_summary,
        summary_token_threshold=args.summary_token_threshold,
        model=opt.model,
        if_add_doc_description=opt.if_add_doc_description,
        if_add_node_text=opt.if_add_node_text,
        if_add_node_id=opt.if_add_node_id
    ))
    logger.info("Tree extraction completed")

    logger.info("Preparing output file...")

    # Save results
    md_name = os.path.splitext(os.path.basename(args.md_path))[0]
    output_dir = args.output_dir
    output_file = f'{output_dir}/{md_name}_structure.json'
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)

    logger.info("-" * 60)
    logger.info(f"SUCCESS: Tree structure saved to: {output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
