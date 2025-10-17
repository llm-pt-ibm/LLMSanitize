import json
import argparse
from typing import List
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)


def clean_text_for_comparison(text: str) -> str:
    """
    Cleans the text by removing non-alphabetic characters and converting it
    to lowercase to standardize the comparison.
    """
    if not isinstance(text, str):
        return ''
    return ''.join(char.lower() for char in text if char.isalpha())


def load_texts_from_source(dataset_name: str, text_key: str) -> List[str]:
    """
    Loads a dataset from Hugging Face and returns a list of cleaned texts.
    """
    logging.info(f"🔍 Loading source dataset '{dataset_name}'...")
    try:
        dataset = load_dataset(dataset_name, split="train")
        texts = [clean_text_for_comparison(item) for item in dataset[text_key]]
        logging.info(f"✅ Source dataset loaded. Total examples: {len(texts)}")
        return texts
    except Exception as e:
        logging.error(f"❌ Error loading source dataset '{dataset_name}': {e}")
        return []


def load_texts_from_local_file(file_path: str, text_key: str) -> List[str]:
    """
    Loads texts from a local JSONL file.
    """
    logging.info(f"🔍 Loading local contamination file '{file_path}'...")
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                texts.append(clean_text_for_comparison(data.get(text_key, '')))
        logging.info(f"✅ Local file loaded. Total examples: {len(texts)}")
        return texts
    except FileNotFoundError:
        logging.error(f"❌ Error: Local file '{file_path}' not found.")
        return []
    except Exception as e:
        logging.error(f"❌ Error reading local file '{file_path}': {e}")
        return []


def get_contamination_percentage(
    source_texts: List[str],
    contaminated_texts: List[str]
) -> float:
    """
    Calculates the percentage of source texts present in the contamination file.
    """
    if not source_texts or not contaminated_texts:
        return 0.0

    source_set = set(source_texts)
    contaminated_set = set(contaminated_texts)

    intersection = source_set.intersection(contaminated_set)
    
    total_source_items = len(source_set)
    found_items = len(intersection)
    
    contamination_percentage = (found_items / total_source_items) * 100 if total_source_items > 0 else 0

    logging.info(f"📌 Analysis Summary:")
    logging.info(f"Total unique items in source: {total_source_items}")
    logging.info(f"Source items found in local file: {found_items}")
    logging.info(f"📊 Contamination percentage: {contamination_percentage:.4f}%")

    return contamination_percentage


def main():
    """
    Main function to analyze dataset contamination.
    """
    parser = argparse.ArgumentParser(
        description="Analyzes the percentage of texts from a source dataset that are present in a local contamination file."
    )
    parser.add_argument(
        "--source_dataset",
        type=str,
        required=True,
        help="Name of the source dataset on Hugging Face (e.g., 'eduagarcia/oab_exams')."
    )
    parser.add_argument(
        "--source_key",
        type=str,
        required=True,
        help="Key for the text field in the source dataset (e.g., 'question')."
    )
    parser.add_argument(
        "--local_file",
        type=str,
        required=True,
        help="Path to the local contamination file (e.g., 'contaminated_texts.jsonl')."
    )
    parser.add_argument(
        "--local_key",
        type=str,
        required=True,
        help="Key for the text field in the local contamination file (e.g., 'text')."
    )

    args = parser.parse_args()

    source_texts = load_texts_from_source(args.source_dataset, args.source_key)
    local_texts = load_texts_from_local_file(args.local_file, args.local_key)
    
    if source_texts and local_texts:
        contamination_rate = get_contamination_percentage(source_texts, local_texts)
        print(f"\n--- Contamination Analysis Complete ---")
        print(f"Contamination rate of dataset '{args.source_dataset}' in local file '{args.local_file}': {contamination_rate:.4f}%")
    else:
        print("\n--- Analysis Incomplete ---")
        print("Could not load both datasets. Please check the names and paths.")


if __name__ == "__main__":
    main()