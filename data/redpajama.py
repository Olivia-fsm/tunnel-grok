import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset 
import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/'
# export HF_DATASETS_CACHE='/scratch/'
REDPAJIMA_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/redpajama/")
tknzr = tiktoken.get_encoding("gpt2")

def get_redpajama(subset='arxiv', num_proc=1):
    """ https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T
    """
    # {
    #     "text": ...,
    #     "meta": {"url": "...", "timestamp": "...", "source": "...", "language": "...", ...},
    #     "red_pajama_subset": "common_crawl" | "c4" | "github" | "books" | "arxiv" | "wikipedia" | "stackexchange"
    # }
    SUBSET_PATH = os.path.join(REDPAJIMA_DATA_PATH, subset)
    if not os.path.exists(os.path.join(REDPAJIMA_DATA_PATH, subset, 'val.bin')):
        os.makedirs(SUBSET_PATH, exist_ok=True)
        if subset=='cc':
            subset_name = 'common_crawl'
        else:
            subset_name = subset
        dataset = load_dataset("togethercomputer/RedPajama-Data-1T", subset_name, )

        try:
            split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        except:
            split_dataset = dataset['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
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
            remove_columns=['text', 'meta'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'])
            filename = os.path.join(SUBSET_PATH, f'{split}.bin')
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 100

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
    train_data = np.memmap(os.path.join(SUBSET_PATH, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(SUBSET_PATH, 'val.bin'), dtype=np.uint16, mode='r')

    return {'train': train_data, 'val': val_data}