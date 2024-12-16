import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import Dict
from functools import partial
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import re


def extract_qa_pairs(text: str) -> list:
    """
    Extracts question-answer pairs from the given text.

    Args:
        text (str): The input text containing questions and answers in a specific format.

    Returns:
        list: A list of tuples, where each tuple is (Question, Answer).
    """
    qa_pattern = r"Question:\s*(.*?)\nAnswer:\s*(.*?)(?=\nQuestion:|$)"  # Regex to match Q&A pairs
    matches = re.findall(qa_pattern, text, re.DOTALL)  # DOTALL allows multiline matching
    return [(q.strip(), a.strip()) for q, a in matches]  # Clean up whitespace


def process(example: Dict, tokenizer: AutoTokenizer)->Dict:
    """
    Tokenize the text and return the tokenized text
    """
    ids = tokenizer(example['text'])
    return dict(ids=ids,len=len(ids))

def write_to_memmap(dset: Dataset, filename: str):
    dtype = np.int32
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = min(1024, len(dset))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
        arr.flush()
from tqdm import tqdm

def create_sft_qa_dataset(ds, tokenizer):
    qa_ds = []
    current_buffer = []
    threshold = 4000
    ds = ds.map(lambda x: {'rephrase_as_qa_list': extract_qa_pairs(x['rephrase_as_qa'])})
    for e in tqdm(ds):
        for qa_list_element in e['rephrase_as_qa_list']:
            current_buffer.append(f"### Question\n{qa_list_element[0]}\n###Answer\n{qa_list_element[1]}")
            tokenized_seq = tokenizer.encode("\n\n".join(current_buffer))
            if len(tokenized_seq) > threshold:
                qa_ds.append({'text': "\n\n".join(current_buffer[:-1])})
                current_buffer = [current_buffer[-1]]
    qa_ds = Dataset.from_list(qa_ds)
    # qa_ds = qa_ds.map(lambda x: {'len': len(tokenizer.encode(x['text']))})
    return qa_ds

def tokenize_and_save(tokenizer: AutoTokenizer):
    """
    After saving the tokenized text, we may read them as
    # >>> import numpy as np
    # >>> arr = np.memmap('data/dataset/bins/ultrachat_test.bin', mode='r', dtype=np.int32)
    # >>> len(arr)
    # 27683545
    # >>> arr[:5]
    # memmap([128000, 128006,   9125, 128007,    271], dtype=int32)
    # >>> from transformers import AutoTokenizer
    # >>> tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', use_fast=True)
    # >>> print(tokenizer.decode(arr[11000000:11000010]))
    # Force, flexible days and times. Contact: Cheryl
    """
    process_map = partial(process, tokenizer=tokenizer)
    # loading dataset
    dataset = load_dataset('json', data_files="/new_data/wenlong/knowledge_sdg/flow_0.1.jsonl", split='train')
    dataset = create_sft_qa_dataset(dataset, tokenizer)#.train_test_split(0.05)
    filename = f'data/dataset/bins/flow_0.1'
    # core tokenization operation happening
    tokenized_train = dataset.map(process_map,
                                           remove_columns=dataset[0].keys(),
                                           desc='Tokenizing training split',
                                           num_proc=16)
    # tokenized_train = dataset['train'].map(process_map,
    #                                        remove_columns=dataset['train'][0].keys(),
    #                                        desc='Tokenizing training split',
    #                                        num_proc=16)
    # tokenized_test = dataset['test'].map(process_map,
    #                                      remove_columns=dataset['train'][0].keys(),
    #                                      desc='Tokenizing test split',
    #                                      num_proc=16)
    print(f"Test size {dataset.num_rows}")
    # concatenate all the ids in each dataset into one large file we can use for training
    write_to_memmap(tokenized_train, f"{filename}.bin")
    # write_to_memmap(tokenized_test, f"{filename}_test.bin")


if __name__ == '__main__':
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', use_fast=True)
    tokenizer.model_max_length=2**20 # this is to hide the token_len>2048 wraning
    # tokenizing the dataset
    tokenize_and_save(tokenizer)