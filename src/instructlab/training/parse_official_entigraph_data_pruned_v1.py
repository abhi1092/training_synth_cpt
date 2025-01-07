from datasets import load_dataset
from io_utils import *
import os
from tqdm import tqdm

ds = load_dataset('json', data_files="/new_data/wenlong/data/entigraph-quality-corpus/entigraph-quality-corpus-pruned-v1.1.jsonl")

os.makedirs("data/quality_entigraph_gpt-4-turbo-pruned-v1.1", exist_ok=True)
for example in tqdm(ds['train']):
    output = []
    output_path = f'data/quality_entigraph_gpt-4-turbo-pruned-v1.1/{example["uid"]}.json'
    
    entities = example['entity'].split("<|entityseptoekn|>")
    entities = [entity.strip() for entity in entities]
    output.append(entities)
    
    texts = example['entigraph'].split("<|entigraphseptoekn|>")
    texts = [text.strip() for text in texts]
    output.extend(texts)
    
    jdump(output, output_path)
