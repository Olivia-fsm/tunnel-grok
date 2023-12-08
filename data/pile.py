import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset 


PILE_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/pile/")
tknzr = tiktoken.get_encoding("gpt2")

"""
    Cleaning was performed by removing everything from the Books3, BookCorpus2, OpenSubtitles, YTSubtitles, and OWT2 subsets.
Based on section 7.1 of the original paper, these datasets are the only ones which are not explicitly allowed to be used in AI training.
"""

def get_pile_data(subset, num_proc=40):
    """ https://huggingface.co/datasets/EleutherAI/pile
    """
    SUBSET_PATH = os.path.join(PILE_DATA_PATH, subset)
    subset_name = ' '.join(subset.split('_'))
    if not os.path.exists(os.path.join(SUBSET_PATH, 'val.bin')):
        os.makedirs(SUBSET_PATH, exist_ok=True)
        # split_dataset = load_dataset("monology/pile-uncopyrighted", data_dir="all", split=['train', 'test'])
        split_dataset = load_dataset("monology/pile-uncopyrighted", split=['train', 'test'])
        # split_dataset['test'] = load_dataset("EleutherAI/pile", revision="refs/convert/parquet", data_dir="all", split='test')
        # print(len(split_dataset))
        data_dict = {}
        data_dict['train'] = split_dataset[0].filter(lambda example: example["meta"]['pile_set_name']==subset_name)
        data_dict['val'] = split_dataset[1].filter(lambda example: example["meta"]['pile_set_name']==subset_name)

        # split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        # split_dataset['val'] = split_dataset.pop('test')
        
        def process(example):
            ids = tknzr.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
            ids.append(tknzr.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
            # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
            out = {'ids': ids, 'len': len(ids)}
            return out

        # tokenize the dataset
        tokenized = {}
        tokenized['train'] = data_dict['train'].map(
            process,
            remove_columns=['text', 'meta'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )
        
        tokenized['val'] = data_dict['val'].map(
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
            total_batches = 50

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
