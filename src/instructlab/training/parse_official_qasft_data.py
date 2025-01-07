from datasets import load_dataset
from io_utils import *
import os
from tqdm import tqdm

ds = load_dataset('parquet', data_files="/new_data/wenlong/data/zitongyang-entigraph-qasft/train-00000-of-00001.parquet")

os.makedirs("data/quality_entigraph_gpt-4-turbo", exist_ok=True)
for example in tqdm(ds['train']):
    output = []
    output_path = f'data/quality_qasft_gpt-4-turbo/{example["fileid"]}.json'
    
    # entities = example['entity'].split("<|entityseptoekn|>")
    # entities = [entity.strip() for entity in entities]
    # output.append(entities)
    
    # texts = example['entigraph'].split("<|entigraphseptoekn|>")
    # texts = [text.strip() for text in texts]
    output.extend(example['qa'])
    
    jdump(output, output_path)