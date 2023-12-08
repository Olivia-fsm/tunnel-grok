import json
from datasets import load_dataset
import os
from tqdm import tqdm
import numpy as np
import tiktoken

DATA_DIR = '/mlodata1/sfan/LLM/curriculum-fsm/datasets/redpajama_subset'
# SRC_FILE_MAPPINGS = {
#     'arxiv': os.path.join(DATA_DIR, 'arxiv_sample.jsonl'),
#     'book': os.path.join(DATA_DIR, 'book_sample.jsonl'),
#     'c4': os.path.join(DATA_DIR, 'c4_sample.jsonl'),
#     'cc': os.path.join(DATA_DIR, 'cc_2019-30_sample.jsonl'),
#     'github': os.path.join(DATA_DIR, 'github_sample.jsonl'),
#     'stackexchange': os.path.join(DATA_DIR, 'stackexchange_sample.jsonl'),
#     'wikipedia': os.path.join(DATA_DIR, 'wikipedia_sample.jsonl'),
# }

FILES_LIST = [os.path.join(DATA_DIR, p) for p in os.listdir(DATA_DIR) if p.startswith('cc_20')]

def process(example):
    ids = tknzr.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(tknzr.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out

def convert_jsonl(src_path, trg_path, add=False):
  if add:
    pattern = 'w+'
  else:
    pattern = 'w'
  with open(trg_path, pattern) as trg:
    with open(src_path, 'r') as src:
        for line in src:
            data = json.loads(line)
            if 'meta' in data.keys():
              data.pop('meta')
            json.dump(data, trg)
            trg.write('\n')
  return trg_path

def write_to_jsonl(file_path, data):
    with open(file_path, 'w') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')


# DATA_DIR = '/content/drive/Shareddrives/EPFL-Research/LLM/datasets/'
CC_SUBSET_PATH = os.path.join(os.path.dirname(__file__), "datasets/cc_subset/")
tknzr = tiktoken.get_encoding("gpt2")
def get_cc_subset(num_proc=40, 
                  test_size=0.0005, total_batches=1024):
  if not os.path.exists(os.path.join(CC_SUBSET_PATH, 'val.bin')):
      os.makedirs(CC_SUBSET_PATH, exist_ok=True)
      trg_datafile = os.path.join(CC_SUBSET_PATH, 'cc_subset_redpajama.jsonl')
      for src_datafile in FILES_LIST:
        data_path = convert_jsonl(src_path=src_datafile,
                                  trg_path=trg_datafile,
                                  add=True)
      dataset = load_dataset('json', data_files=data_path)

      split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=2357, shuffle=True)
      split_dataset['val'] = split_dataset.pop('test')
      
      def process(example):
          ids = tknzr.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
          ids.append(tknzr.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
          # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
          out = {'ids': ids, 'len': len(ids)}
          return out

      # tokenize the dataset
      tokenized = split_dataset.map(
          process,
          remove_columns=['text'],
          desc="tokenizing the splits",
          num_proc=num_proc,
      )

      # concatenate all the ids in each dataset into one large file we can use for training
      for split, dset in tokenized.items():
          arr_len = np.sum(dset['len'])
          filename = os.path.join(CC_SUBSET_PATH, f'{split}.bin')
          dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
          arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
          total_batches = total_batches

          idx = 0
          for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
              # Batch together samples for faster write
              try:
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
              except:
                break
              arr_batch = np.concatenate(batch['ids'])
              # Write into mmap
              arr[idx : idx + len(arr_batch)] = arr_batch
              idx += len(arr_batch)
          arr.flush()

  train_data = np.memmap(os.path.join(CC_SUBSET_PATH, 'train.bin'), dtype=np.uint16, mode='r')
  val_data = np.memmap(os.path.join(CC_SUBSET_PATH, 'val.bin'), dtype=np.uint16, mode='r')
  print(f'Subset: cc_subset || train: {len(train_data)} || val: {len(val_data)}')
  return {'train': train_data, 'val': val_data}
