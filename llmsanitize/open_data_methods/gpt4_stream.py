"""
This file implements a resilient, multi-processing streaming approach for 
open-data contamination detection based on GPT-4's string-matching methodology.
https://arxiv.org/pdf/2303.08774.pdf
"""

import json
import time
import logging
import multiprocessing as mp
import functools
import numpy as np
from requests.exceptions import RequestException
from urllib3.exceptions import ProtocolError
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar

from llmsanitize.utils.string_utils import build_substrings, overlap_substrings_frequency
from llmsanitize.utils.logger import get_child_logger

logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logger = get_child_logger("gpt4_stream_resilient")


def clean_text_gpt4(text: str) -> str:
    return ''.join(i for i in text if i.isalnum())


def save_contaminated_text(text: str, filename: str):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps({"contaminated_text": text}, ensure_ascii=False) + "\n")


def load_existing_contaminated_texts(filename: str) -> set:
    existing_texts = set()
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                existing_texts.add(json.loads(line)["contaminated_text"])
    except FileNotFoundError:
        pass
    return existing_texts


def save_progress(filename: str, processed_count: int, batch_overlaps_total: int, checked_total: int, contaminated_total: int, contaminated_batches: int):
    state = {
        "processed_examples": processed_count,
        "n_batch_overlaps_total": batch_overlaps_total,
        "total_checked_total": checked_total,
        "n_contaminated_total": contaminated_total,
        "contaminated_batches_total": contaminated_batches
    }
    with open(filename, "w") as f:
        json.dump(state, f)


def load_progress(filename: str) -> tuple:
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return (
                data.get("processed_examples", 0),
                data.get("n_batch_overlaps_total", 0),
                data.get("total_checked_total", 0),
                data.get("n_contaminated_total", 0),
                data.get("contaminated_batches_total", 0)
            )
    except (FileNotFoundError, json.JSONDecodeError):
        return 0, 0, 0, 0, 0


def save_final_report(filename: str, report_data: dict):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)


def process_batch(batch: list, eval_data: list, string_size: int, n_samples: int, seed: int):
    np.random.seed(seed)
    train_substrings = build_substrings(batch, string_size, clean_text_gpt4)
    
    frequencies = overlap_substrings_frequency(
        eval_data, train_substrings, string_size, n_samples, clean_text_gpt4, seed
    )

    newly_contaminated_texts = [eval_data[i] for i, freq in enumerate(frequencies) if freq > 0]
    n_batch_overlaps_in_batch = sum(1 for f in frequencies if f > 0)
    n_contaminated_in_batch = sum(frequencies)
    
    return newly_contaminated_texts, n_batch_overlaps_in_batch, len(frequencies), len(batch), n_contaminated_in_batch


def generate_batches(stream, text_key: str, batch_size: int):
    batch = []
    for example in stream:
        batch.append(example[text_key])
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main_gpt4_stream(
   train_data_name: str = None,
   eval_data: list = [],
   eval_data_name: str = None,
   eval_set_key: str = None,
   train_text_key: str = None,
   eval_text_key: str = None,
   num_proc: int = 1,
   seed: int = 42,
   batch_size: int = 6000
):
    disable_progress_bar()
    
    eval_data_questions = eval_data[eval_text_key]
    string_size = 50
    n_samples = 3

    output_file = f"contaminated_{eval_data_name.replace('/', '_')}_vs_{train_data_name.replace('/', '_')}.jsonl"
    progress_file = f"progress_{eval_data_name.replace('/', '_')}_vs_{train_data_name.replace('/', '_')}.json"
    report_file = f"report_{eval_data_name.replace('/', '_')}_vs_{train_data_name.replace('/', '_')}.json"

    contaminated_texts_set = load_existing_contaminated_texts(output_file)
    processed_examples, n_batch_overlaps_total, total_checked_total, n_contaminated_total, contaminated_batches_total = load_progress(progress_file)

    max_retries = 10
    retry_count = 0

    while retry_count < max_retries:
        try:
            train_data_stream = load_dataset(train_data_name, split="train", streaming=True)
            if processed_examples > 0:
                train_data_stream = train_data_stream.skip(processed_examples)

            worker_func = functools.partial(
                process_batch, eval_data=eval_data_questions, string_size=string_size, n_samples=n_samples, seed=seed
            )

            with mp.Pool(processes=num_proc) as pool:
                batch_generator = generate_batches(train_data_stream, train_text_key, batch_size)
                
                for result in pool.imap_unordered(worker_func, batch_generator):
                    new_texts, n_batch_overlaps, n_checked, batch_len, n_cont_absolute = result

                    for text in new_texts:
                        if text not in contaminated_texts_set:
                            save_contaminated_text(text, output_file)
                            contaminated_texts_set.add(text)

                    n_batch_overlaps_total += n_batch_overlaps
                    total_checked_total += n_checked
                    processed_examples += batch_len
                    n_contaminated_total += n_cont_absolute
                    
                    if n_batch_overlaps > 0:
                        contaminated_batches_total += 1

                    save_progress(progress_file, processed_examples, n_batch_overlaps_total, total_checked_total, n_contaminated_total, contaminated_batches_total)
                    logger.info(f"Progress: {processed_examples} training examples processed.")
            break

        except (RequestException, ProtocolError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise e
            time.sleep(30)

    total_benchmark_questions = len(eval_data_questions)
    total_batches_processed = total_checked_total // total_benchmark_questions if total_benchmark_questions > 0 else 0
    final_unique_contaminated = len(contaminated_texts_set)
    total_instance_comparisons = processed_examples * total_benchmark_questions
    
    blr_percent = 100 * (final_unique_contaminated / total_benchmark_questions) if total_benchmark_questions > 0 else 0
    cdp_percent = 100 * (n_contaminated_total / total_instance_comparisons) if total_instance_comparisons > 0 else 0
    dpl_percent = 100 * (contaminated_batches_total / total_batches_processed) if total_batches_processed > 0 else 0
    dpb_percent = 100 * (n_batch_overlaps_total / total_checked_total) if total_checked_total > 0 else 0

    report_data = {
        "setup": {
            "evaluated_benchmark": f"{eval_data_name}/{eval_set_key}",
            "training_corpus": train_data_name,
            "batch_size": batch_size
        },
        "core_metrics": {
            "Benchmark Leakage Rate (BLR)": f"{blr_percent:.4f}%",
            "Contamination Density Parcial (CDP)": f"{cdp_percent:.6f}%",
            "Dispersion Per Lote (DPL)": f"{dpl_percent:.6f}%",
            "Dispersion Per Benchmark (DPB)": f"{dpb_percent:.6f}%"
        },
        "raw_counters": {
            "total_benchmark_instances": total_benchmark_questions,
            "total_corpus_instances": processed_examples,
            "total_batches_processed": total_batches_processed,
            "unique_leaked_instances": final_unique_contaminated,
            "absolute_contamination_occurrences": n_contaminated_total,
            "contaminated_batches": contaminated_batches_total,
            "item_batch_overlaps": n_batch_overlaps_total
        }
    }

    save_final_report(report_file, report_data)

    logger.info(f"Open-data contamination: checking {eval_data_name}/{eval_set_key} against {train_data_name} (train)")
    logger.info(f"Method: streaming batch dispersion with {string_size}-chars substrings (GPT-4 style)")
    logger.info(json.dumps(report_data, indent=4, ensure_ascii=False))
    logger.info(f"Contamination results saved to '{output_file}'")
    logger.info(f"Run progress saved to '{progress_file}'")
    logger.info(f"Final report saved to '{report_file}'")