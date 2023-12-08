import numpy as np
from typing import Dict
import random
import itertools

from .shakespeare import get_shakespeare_data
from .wikitext import get_wikitext_data
from .arxiv import get_arxiv_2000, get_arxiv_full
from .openwebtext2 import get_openwebtext2_data
from .redpajama_subset import get_redpajama_subset, SRC_FILE_MAPPINGS
from .redpajama_split_train import get_redpajama_split_train
from .redpajama import get_redpajama
from .slim_redpajama import get_slim_redpajama, get_slim_redpajama_6b, SUBSET2META
from .cc_subset import get_cc_subset
from .pile import get_pile_data
from .textbook import get_textbook

PILE_DOMAINS = [
    'Wikipedia_(en)',
    'PubMed_Abstracts',
    'StackExchange',
    'Github',
    # 'OpenWebText2',
    'USPTO_Backgrounds',
    # 'YoutubeSubtitles',
    'FreeLaw',
    'PubMed_Central',
    'DM_Mathematics',
    'Enron_Emails',
    # 'OpenSubtitles',
    'HackerNews',
    'ArXiv',
    'NIH_ExPorter',
    # 'Books3',
    'Gutenberg_(PG-19)',
    'Ubuntu_IRC',
    # 'BookCorpus2',
    'EuroParl',
    'PhilPapers',
    'Pile-CC',
]

def generate_random_text(corpus, length=1000, ctx_len=1):
    '''
    length: total num of tokens to generate. 
    ctx_len: length of continuous tokens. the longer the more similar to real texts.'''
    if ctx_len==1:
        return np.array([random.choice(corpus) for _ in range(length)], dtype=np.uint16)
    rand_idx = random.randint(0,len(corpus))
    random_text = [corpus[rand_idx: rand_idx+ctx_len] for _ in range(length//ctx_len)]
    random_text = list(itertools.chain.from_iterable(random_text))
    assert len(random_text) == ctx_len * (length//ctx_len)
    # print('random_text', len(random_text))
    return np.array(random_text, dtype=np.uint16)

def generate_mixture(corpus_list:list, 
                     length=1000, 
                     mix_rate=None):
    '''
    length: total num of tokens to generate. 
    mix_rate: rate of mix each corpus in corpus dict. uniform if not provided.'''
    if mix_rate is None:
        mix_rate = np.ones(len(corpus_list)) / len(corpus_list)
    assert len(mix_rate) == len(corpus_list)
    mix_length = np.floor(length * mix_rate)
    print('mix_length', mix_length)
    
    mix_texts = []
    for i, cps in enumerate(corpus_list):
        texts = generate_random_text(cps, length=int(mix_length[i]), ctx_len=1000)
        mix_texts.extend(texts)
    return np.array(mix_texts, dtype=np.uint16)


    
def get_dataset(args, dataset=None) -> Dict[str, np.ndarray]:
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own python file. The expected format at the moment is a dictionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    if dataset is not None:
        trg_dataset = dataset
    else:
        trg_dataset = args.dataset
    print(f"Loading train dataset '{trg_dataset}'")
    if trg_dataset == 'wikitext':
        return get_wikitext_data()
    if trg_dataset == "shakespeare-char":
        return get_shakespeare_data()
    if trg_dataset == "arxiv2000":
        return get_arxiv_2000()
    if trg_dataset == "arxiv":
        return get_arxiv_full()
    if trg_dataset == "arxiv+wiki":
        arxiv_data = get_arxiv_full()
        wiki_data = get_wikitext_data()
        train_data = np.concatenate((arxiv_data['train'], wiki_data['train']))
        val_data = np.concatenate((arxiv_data['val'], wiki_data['val']))
        rst_dict = {}
        rst_dict['train'] = {
            'arxiv': arxiv_data['train'],
            'wiki': wiki_data['train'],
            'all': train_data,
        }
        rst_dict['val'] = {
            'arxiv': arxiv_data['val'],
            'wiki': wiki_data['val'],
            'all': val_data,
        }
        return rst_dict
        # return {'train': train_data, 'val': val_data}
    if trg_dataset == 'openwebtext2':
        return get_openwebtext2_data()
    if trg_dataset =='cc_subset':
        return get_cc_subset()
    if trg_dataset=="arxiv+wiki+cc":
        arxiv_data = get_arxiv_full()
        wiki_data = get_wikitext_data()
        cc_data = get_cc_subset()
        rst_dict = {}
        rst_dict['train'] = {
            'arxiv': arxiv_data['train'],
            'wiki': wiki_data['train'],
            'cc': cc_data['train'],
        }
        rst_dict['val'] = {
            'arxiv': arxiv_data['val'],
            'wiki': wiki_data['val'],
            'cc': cc_data['val'],
        }
        return rst_dict
    if trg_dataset == "synthe_noise":
        arxiv_data = get_arxiv_full()
        wiki_data = get_wikitext_data()
        cc_data = get_cc_subset()
        train_data = np.concatenate((arxiv_data['train'], wiki_data['train'], cc_data['train']))
        val_data = np.concatenate((arxiv_data['val'], wiki_data['val'], cc_data['val']))
        noise_data_train = generate_random_text(corpus=train_data, length=min(len(arxiv_data['train']), len(wiki_data['train']), len(cc_data['train'])), ctx_len=1)
        noise_data_val = generate_random_text(corpus=train_data, length=min(len(arxiv_data['val']), len(wiki_data['val']), len(cc_data['train'])), ctx_len=1)
        rst_dict = {}
        rst_dict['train'] = {
            'arxiv': arxiv_data['train'],
            'wiki': wiki_data['train'],
            'cc': cc_data['train'],
            'noise': noise_data_train,
        }
        rst_dict['val'] = {
            'arxiv': arxiv_data['val'],
            'wiki': wiki_data['val'],
            'cc': cc_data['val'],
            'noise': noise_data_val,
        }
        return rst_dict
    
    if trg_dataset == "synthe_mix":
        arxiv_data = get_arxiv_full()
        wiki_data = get_wikitext_data()
        train_data = np.concatenate((arxiv_data['train'], wiki_data['train']))
        val_data = np.concatenate((arxiv_data['val'], wiki_data['val']))
        mix_data_train = generate_mixture(corpus_list=[arxiv_data['train'], wiki_data['train']], length=min(len(arxiv_data['train']), len(wiki_data['train'])), mix_rate=None)
        mix_data_val = generate_mixture(corpus_list=[arxiv_data['train'], wiki_data['train']], length=min(len(arxiv_data['val']), len(wiki_data['val'])), mix_rate=None)
        rst_dict = {}
        rst_dict['train'] = {
            'arxiv': arxiv_data['train'],
            'wiki': wiki_data['train'],
            'mix': mix_data_train,
        }
        rst_dict['val'] = {
            'arxiv': arxiv_data['val'],
            'wiki': wiki_data['val'],
            'mix': mix_data_val,
        }
        return rst_dict
    
    if trg_dataset=="textbook":
        textbook_data = get_textbook()
        return textbook_data
    
    if 'pile' in trg_dataset:
        subset = trg_dataset.split('-')[1]
        if subset == 'all' or args.eval_all_domains:
            all_train_list, all_val_list = [], []
            rst_dict = {}
            rst_dict['train'] = {}
            rst_dict['val'] = {}
            for k in PILE_DOMAINS:
                subset_data = get_pile_data(subset=k, num_proc=10)
                rst_dict['train'][k] = subset_data['train']
                rst_dict['val'][k] = subset_data['val']
                all_train_list.append(subset_data['train'])
                all_val_list.append(subset_data['val'])
            train_data = np.concatenate(all_train_list)
            val_data = np.concatenate(all_val_list)
            # rst_dict['train']['all'] = train_data
            # rst_dict['val']['all'] = val_data
            
            # print stats #
            # print(f"Num training tokens: {len(rst_dict['train']['all'])}")
            # if 'val' in rst_dict.keys():
            #     print(f"Num validation tokens: {len(rst_dict['val']['all'])}")
            # else:
            #     for k in rst_dict['val'].keys():
            #         print(f"Num validation tokens({k.split('-')}): {len(rst_dict['val'][k])}")
            if subset != 'all':
                rst_dict['train'] = rst_dict['train'][subset]
                if 'all' in rst_dict['val'].keys():
                    rst_dict['val'].pop('all')
            return rst_dict
        return get_pile_data(subset=subset, num_proc=10)
            
        
    if 'full_redpajama' in trg_dataset:
        subset = trg_dataset.split('-')[1]
        if subset == 'all' or args.eval_all_domains:
            all_train_list, all_val_list = [], []
            rst_dict = {}
            rst_dict['train'] = {}
            rst_dict['val'] = {}
            for k in SRC_FILE_MAPPINGS.keys():
                subset_data = get_redpajama(subset=k, num_proc=10)
                rst_dict['train'][k] = subset_data['train']
                rst_dict['val'][k] = subset_data['val']
                all_train_list.append(subset_data['train'])
                all_val_list.append(subset_data['val'])
            train_data = np.concatenate(all_train_list)
            val_data = np.concatenate(all_val_list)
            # rst_dict['train']['all'] = train_data
            # rst_dict['val']['all'] = val_data
            
            # print stats #
            # print(f"Num training tokens: {len(rst_dict['train']['all'])}")
            # if 'val' in rst_dict.keys():
            #     print(f"Num validation tokens: {len(rst_dict['val']['all'])}")
            # else:
            #     for k in rst_dict['val'].keys():
            #         print(f"Num validation tokens({k.split('-')}): {len(rst_dict['val'][k])}")
            if subset != 'all':
                rst_dict['train'] = rst_dict['train'][subset]
                if 'all' in rst_dict['val'].keys():
                    rst_dict['val'].pop('all')
            return rst_dict
        return get_redpajama(subset=subset, num_proc=10)
    elif 'slim_6b' in trg_dataset:
        subset = trg_dataset.split('-')[1]
        if subset == 'all' or args.eval_all_domains:
            all_train_list, all_val_list = [], []
            rst_dict = {}
            rst_dict['train'] = {}
            rst_dict['val'] = {}
            for k in SUBSET2META.keys():
                subset_data = get_slim_redpajama_6b(subset=k, num_proc=10)
                rst_dict['train'][k] = subset_data['train']
                rst_dict['val'][k] = subset_data['val']
                all_train_list.append(subset_data['train'])
                all_val_list.append(subset_data['val'])
            train_data = np.concatenate(all_train_list)
            val_data = np.concatenate(all_val_list)
            rst_dict['train']['all'] = train_data
            rst_dict['val']['all'] = val_data
            
            if subset != 'all':
                rst_dict['train'] = rst_dict['train'][subset]
                if 'all' in rst_dict['val'].keys():
                    rst_dict['val'].pop('all')
            return rst_dict
        return get_slim_redpajama_6b(subset=subset, num_proc=10)
    elif 'slim_full' in trg_dataset:
        subset = trg_dataset.split('-')[1]
        if subset =='all':
            n_items_val = 1000
            all_train_list, all_val_list = [], []
            rst_dict = {}
            rst_dict['train'] = {}
            rst_dict['val'] = {}
            for k in SRC_FILE_MAPPINGS.keys():
                subset_data = get_slim_redpajama(subset=k, num_proc=10)
                rst_dict['train'][k] = subset_data['train']
                rst_dict['val'][k] = subset_data['val'][:n_items_val*args.max_token_length]
                all_train_list.append(subset_data['train'])
                all_val_list.append(subset_data['val'][:n_items_val*args.max_token_length])
            train_data = np.concatenate(all_train_list)
            val_data = np.concatenate(all_val_list)
            rst_dict['train']['all'] = train_data
            rst_dict['val']['all'] = val_data
            
            # print stats #
            # print(f"Num training tokens: {len(rst_dict['train']['all'])}")
            # if 'val' in rst_dict.keys():
            #     print(f"Num validation tokens: {len(rst_dict['val']['all'])}")
            # else:
            #     for k in rst_dict['val'].keys():
            #         print(f"Num validation tokens({k.split('-')}): {len(rst_dict['val'][k])}")
            if subset != 'all':
                rst_dict['train'] = rst_dict['train'][subset]
                if 'all' in rst_dict['val'].keys():
                    rst_dict['val'].pop('all')
            return rst_dict
        elif subset == 'mix':
            # all_train_list, all_val_list = [], []
            rst_dict = {}
            rst_dict['train'] = {}
            rst_dict['val'] = {}
            mix_data_train = []
            mix_data_val = []
            n_items_mix_train = (2000000000//args.max_token_length)//7
            n_items_mix_val = 2000
            
            for k in SRC_FILE_MAPPINGS.keys():
                subset_data = get_slim_redpajama(subset=k, num_proc=10)
                rst_dict['train'][k] = subset_data['train'][:-n_items_mix_train*args.max_token_length]
                rst_dict['val'][k] = subset_data['val'][:n_items_mix_val*args.max_token_length]
                mix_data_train.append(subset_data['train'][-n_items_mix_train*args.max_token_length:])
                mix_data_val.append(subset_data['val'][-n_items_mix_val*args.max_token_length:])
                # all_train_list.append(rst_dict['train'][k])
                # all_val_list.append(rst_dict['val'][k])
            # train_data = np.concatenate(all_train_list)
            # val_data = np.concatenate(all_val_list)
            # rst_dict['train']['all'] = train_data
            # rst_dict['val']['all'] = val_data
            
            mix_train_data = np.concatenate(mix_data_train)
            mix_val_data = np.concatenate(mix_data_val)
            # shuffle
            A = np.arange(0, len(mix_train_data), args.max_token_length)
            np.random.shuffle(A)
            mix_train_data = np.concatenate([mix_train_data[i:i+args.max_token_length] for i in A])
            
            B = np.arange(0, len(mix_val_data), args.max_token_length)
            np.random.shuffle(B)
            mix_val_data = np.concatenate([mix_val_data[i:i+args.max_token_length] for i in B])
            
            rst_dict['train']['mix'] = mix_train_data
            rst_dict['val']['mix'] = mix_val_data
            
            # print stats #
            # print(f"Num training tokens: {len(rst_dict['train']['all'])}")
            # if 'val' in rst_dict.keys():
            #     print(f"Num validation tokens: {len(rst_dict['val']['all'])}")
            # else:
            #     for k in rst_dict['val'].keys():
            #         print(f"Num validation tokens({k.split('-')}): {len(rst_dict['val'][k])}")
            
            # if 'all' in rst_dict['val'].keys():
            #     rst_dict['val'].pop('all')
            return rst_dict
        return get_slim_redpajama(subset=subset, num_proc=10)
    elif 'redpajama' in trg_dataset:
        subset = trg_dataset.split('-')[1]
        if subset == 'train2':
            domain = trg_dataset.split('-')[-1]
            all_train1_list, all_train2_list, all_val_list = [], [], []
            rst_dict = {}
            rst_dict['train1'] = {}
            rst_dict['train2'] = {}
            rst_dict['val'] = {}
            for k in SRC_FILE_MAPPINGS.keys():
                subset_data = get_redpajama_split_train(subset=k, num_proc=10, test_size=0.05, total_batches=100)
                rst_dict['train1'][k] = subset_data['train1']
                rst_dict['train2'][k] = subset_data['train2']
                rst_dict['val'][k] = subset_data['val']
                all_train1_list.append(subset_data['train1'])
                all_train2_list.append(subset_data['train2'])
                all_val_list.append(subset_data['val'])
            train1_data = np.concatenate(all_train1_list)
            train2_data = np.concatenate(all_train2_list)
            val_data = np.concatenate(all_val_list)
            rst_dict['train1']['all'] = train1_data
            rst_dict['train2']['all'] = train2_data
            
            # print stats #
            print(f"Num training set 1 tokens: {len(rst_dict['train1']['all'])}")
            print(f"Num training set 2 tokens: {len(rst_dict['train2']['all'])}")
            if 'all' in rst_dict['val'].keys():
                print(f"Num validation tokens: {len(rst_dict['val']['all'])}")
            else:
                for k in rst_dict['val'].keys():
                    print(f"Num validation tokens({k.split('-')}): {len(rst_dict['val'][k])}")
            if domain!='train2':
                rst_dict['train1'] = rst_dict['train1'][domain]
                rst_dict['train2'] = rst_dict['train2'][domain]
            return rst_dict
            
        if subset == 'all' or args.eval_all_domains:
            all_train_list, all_val_list = [], []
            rst_dict = {}
            rst_dict['train'] = {}
            rst_dict['val'] = {}
            for k in SRC_FILE_MAPPINGS.keys():
                subset_data = get_redpajama_subset(subset=k, num_proc=10, test_size=0.05, total_batches=100)
                rst_dict['train'][k] = subset_data['train']
                rst_dict['val'][k] = subset_data['val']
                all_train_list.append(subset_data['train'])
                all_val_list.append(subset_data['val'])
            train_data = np.concatenate(all_train_list)
            val_data = np.concatenate(all_val_list)
            rst_dict['train']['all'] = train_data
            rst_dict['val']['all'] = val_data
            
            # print stats #
            print(f"Num training tokens: {len(rst_dict['train']['all'])}")
            if 'val' in rst_dict.keys():
                print(f"Num validation tokens: {len(rst_dict['val']['all'])}")
            else:
                for k in rst_dict['val'].keys():
                    print(f"Num validation tokens({k.split('-')}): {len(rst_dict['val'][k])}")
            if subset != 'all':
                rst_dict['train'] = rst_dict['train'][subset]
                rst_dict['val'].pop('all')
            return rst_dict
        return get_redpajama_subset(subset=subset, num_proc=10, test_size=0.05, total_batches=100)
    else:
        raise NotImplementedError(f"Unknow dataset key '{trg_dataset}'")
