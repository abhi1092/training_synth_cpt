from datasets import load_dataset
from io_utils import *
import os
from tqdm import tqdm

ds = load_dataset("zitongyang/entigraph-quality-corpus")

os.makedirs("data/quality_entigraph_gpt-4-turbo", exist_ok=True)
for example in tqdm(ds['train']):
    output = []
    output_path = f'data/quality_entigraph_gpt-4-turbo/{example["uid"]}.json'
    
    entities = example['entity'].split("<|entityseptoekn|>")
    entities = [entity.strip() for entity in entities]
    output.append(entities)
    
    texts = example['entigraph'].split("<|entigraphseptoekn|>")
    texts = [text.strip() for text in texts]
    output.extend(texts)
    
    jdump(output, output_path)