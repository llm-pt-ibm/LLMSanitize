import json
import time
import logging
import multiprocessing as mp
import functools
import numpy as np
from requests.exceptions import RequestException
from urllib3.exceptions import ProtocolError
from datasets import load_dataset
from llmsanitize.utils.string_utils import build_substrings, overlap_substrings_sample
from llmsanitize.utils.logger import get_child_logger

# --- LOGGING CONFIGURATION ---
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logger = get_child_logger("gpt4_stream_resilient")


# --- HELPER FUNCTIONS ---

def clean_text_gpt4(text):
    """Removes non-alphabetic characters from the text."""
    return ''.join(i for i in text if i.isalpha())


def save_contaminated_text(text, filename):
    """Appends a found contaminated text to the .jsonl results file."""
    record = {"contaminated_text": text}
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_contaminated_texts(filename):
    """Loads already found texts from a previous run."""
    existing_texts = set()
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                existing_texts.add(json.loads(line)["contaminated_text"])
    except FileNotFoundError:
        logger.info(f"Result file '{filename}' not found. Creating a new one.")
    return existing_texts


def save_progress(filename, processed_count, contaminated_total, checked_total):
    """Saves the complete state of the run (progress and counters)."""
    state = {
        "processed_examples": processed_count,
        "n_contaminated_total": contaminated_total,
        "total_checked_total": checked_total
    }
    with open(filename, "w") as f:
        json.dump(state, f)


def load_progress(filename):
    """Loads the complete state. Returns zeros if the file does not exist."""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return (
                data.get("processed_examples", 0),
                data.get("n_contaminated_total", 0),
                data.get("total_checked_total", 0)
            )
    except (FileNotFoundError, json.JSONDecodeError):
        return 0, 0, 0


def save_final_report(filename, report_data):
    """Saves the final report dictionary to a formatted JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)


def process_batch(batch, eval_data, string_size, n_samples, seed):
    """Worker function, seeded for reproducibility."""
    np.random.seed(seed)
    train_substrings = build_substrings(batch, string_size, clean_text_gpt4)
    contaminated_flags = overlap_substrings_sample(eval_data, train_substrings, string_size, n_samples, clean_text_gpt4, seed)

    newly_contaminated_texts = []
    for i, is_contaminated in enumerate(contaminated_flags):
        if is_contaminated:
            newly_contaminated_texts.append(eval_data[i])

    n_contaminated_in_batch = sum(contaminated_flags)
    total_checked_in_batch = len(contaminated_flags)
    return newly_contaminated_texts, n_contaminated_in_batch, total_checked_in_batch, len(batch)


def generate_batches(stream, text_key, batch_size):
    """Reads the stream and yields batches of data for the workers."""
    batch = []
    for example in stream:
        batch.append(example[text_key])
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# --- MAIN FUNCTION ---

def main_gpt4_stream(
   train_data_name,
   eval_data,
   eval_data_name,
   eval_set_key,
   text_key,
   num_proc,
   seed
):
    from datasets.utils.logging import disable_progress_bar
    disable_progress_bar()

    NUM_WORKERS = num_proc
    BATCH_SIZE = 6000

    logger.info("Strategy: Contamination Density (optimized for large datasets).")
    logger.info(f"Optimization: Using {NUM_WORKERS} parallel processes with the imap strategy.")
    logger.info(f"Optimization: Batch size per process: {BATCH_SIZE}.")

    eval_data_questions = eval_data["question"]
    string_size = 50
    n_samples = 3

    output_file = f"contaminated_{eval_data_name.replace('/', '_')}_vs_{train_data_name.replace('/', '_')}.jsonl"
    progress_file = f"progress_{eval_data_name.replace('/', '_')}_vs_{train_data_name.replace('/', '_')}.json"
    report_file = f"report_{eval_data_name.replace('/', '_')}_vs_{train_data_name.replace('/', '_')}.json"

    contaminated_texts_set = load_existing_contaminated_texts(output_file)
    processed_examples, n_contaminated_total, total_checked_total = load_progress(progress_file)

    if processed_examples > 0:
        logger.info(f"RESUMING: Continuing from {processed_examples} already processed examples.")
        logger.info(f"State recovered: {n_contaminated_total} overlaps / {total_checked_total} comparisons.")

    max_retries = 10
    retry_count = 0

    while retry_count < max_retries:
        try:
            logger.info("Starting/Resuming training dataset stream...")
            train_data_stream = load_dataset(train_data_name, split="train", streaming=True)

            if processed_examples > 0:
                logger.info(f"Skipping {processed_examples} examples to resume from the last checkpoint...")
                train_data_stream = train_data_stream.skip(processed_examples)

            worker_func = functools.partial(process_batch,
                                            eval_data=eval_data_questions,
                                            string_size=string_size,
                                            n_samples=n_samples,
                                            seed=seed)

            with mp.Pool(processes=NUM_WORKERS) as pool:
                batch_generator = generate_batches(train_data_stream, text_key, BATCH_SIZE)

                for result in pool.imap_unordered(worker_func, batch_generator):
                    new_texts, n_cont, n_checked, batch_len = result

                    for text in new_texts:
                        if text not in contaminated_texts_set:
                            save_contaminated_text(text, output_file)
                            contaminated_texts_set.add(text)

                    n_contaminated_total += n_cont
                    total_checked_total += n_checked
                    processed_examples += batch_len

                    save_progress(progress_file, processed_examples, n_contaminated_total, total_checked_total)
                    logger.info(f"Progress: {processed_examples} training examples processed.")

            logger.info("✅ Stream processing completed successfully!")
            break

        except (RequestException, ProtocolError) as e:
            retry_count += 1
            logger.warning(f"\nNETWORK ERROR DETECTED: {e}")
            if retry_count >= max_retries:
                logger.error("Maximum number of consecutive network failures reached. Aborting.")
                raise e
            logger.warning(f"Attempt {retry_count}/{max_retries}. Waiting 30 seconds before retrying...")
            time.sleep(30)

    # --- FINAL REPORT ---
    final_unique_contaminated = len(contaminated_texts_set)
    frac = 100 * (n_contaminated_total / total_checked_total) if total_checked_total > 0 else 0

    report_data = {
        "check_details": {
            "benchmark": f"{eval_data_name}/{eval_set_key}",
            "training_corpus": train_data_name
        },
        "parameters": {
            "n_samples": n_samples,
            "string_size": string_size,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "seed": seed
        },
        "results": {
            "total_training_examples_processed": processed_examples,
            "unique_contaminated_data_points": final_unique_contaminated,
            "contamination_density_metric": {
                "total_overlaps": n_contaminated_total,
                "total_comparisons": total_checked_total,
                "percentage": f"{frac:.4f}%"
            }
        }
    }

    save_final_report(report_file, report_data)

    logger.info("--- Final Report ---")
    logger.info(json.dumps(report_data, indent=4, ensure_ascii=False))
    logger.info(f"✅ Contamination results saved to '{output_file}'")
    logger.info(f"✅ Run progress saved to '{progress_file}'")
    logger.info(f"✅ Final report saved to '{report_file}'")