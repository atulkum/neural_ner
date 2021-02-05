from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import glob
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
pad_token_id = tokenizer.pad_token_id

class MultiChoiceDataset(Dataset):
    _MAX_LEN = 30
    def __getitem__(self, n):
        return self._features[n]

    def __len__(self):
        return len(self._features)

    def __init__(self, phase, data_dir):
        self._phase = phase
        self.init_dataset(data_dir)

    def init_dataset(self, data_dir):
        if self._phase == 'train':
            prefix = 'wsj.2-21.txt.dep.pp'
        else:
            prefix = 'wsj.23.txt.dep.pp'
        preps = []
        for filename in glob.glob(data_dir + f'/{prefix}.preps.words'):
            with open(filename) as f:
                preps.extend(f.readlines())
        preps = [t.strip() for t in preps]

        children = []
        for filename in glob.glob(data_dir + f'/{prefix}.children.words'):
            with open(filename) as f:
                children.extend(f.readlines())
        children = [t.strip() for t in children]
        heads = []
        for filename in glob.glob(data_dir + f'/{prefix}.heads.words'):
            with open(filename) as f:
                heads.extend(f.readlines())
        heads = [t.strip() for t in heads]

        n_heads = []
        for filename in glob.glob(data_dir + f'/{prefix}.nheads'):
            with open(filename) as f:
                n_heads.extend(f.readlines())
        n_heads = [int(l.strip()) for l in n_heads]

        n_pres = len(preps)
        assert n_pres == len(children)
        assert n_pres == len(heads)
        assert n_pres == len(n_heads)

        if self._phase in {'train', 'dev'}:
            labels = []
            for filename in glob.glob(data_dir + f'/{prefix}.labels'):
                with open(filename) as f:
                    labels.extend(f.readlines())
            labels = [int(l.strip()) for l in labels]

            assert n_pres == len(labels)

        features = []
        for i, h in enumerate(heads):
            assert len(h.split()) == n_heads[i]
            single_feature = []
            for hi, hh in enumerate(h.split()):
                inputs = tokenizer(
                    h,
                    f'{hh} {preps[i]} {children[i]}',
                    add_special_tokens=False,
                    max_length=self._MAX_LEN,
                    padding="max_length",
                    truncation=True,
                    return_overflowing_tokens=False,

                )
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                token_type_ids = inputs["token_type_ids"]
                single_feature.append({
                    'input_ids': input_ids,
                    'attention_mask':attention_mask,
                    'token_type_ids': token_type_ids
                })
            datum = {
                'input_ids': torch.LongTensor([x['input_ids'] for x in single_feature]),
                'attention_mask': torch.LongTensor([x['attention_mask'] for x in single_feature]),
                'token_type_ids': torch.LongTensor([x['token_type_ids'] for x in single_feature]),
            }
            datum['n_heads'] = n_heads[i]
            if self._phase in {'train', 'dev'}:
                datum['labels'] = labels[i] - 1

            features.append(datum)
        self._features = features

def variable_collate_fn(batch):
    batch_features = {}

    batch_features['input_ids'] = pad_sequence([x['input_ids'] for x in batch],
                                               batch_first=True,
                                               padding_value=pad_token_id)
    batch_features['attention_mask'] = pad_sequence([x['attention_mask'] for x in batch],
                                               batch_first=True,
                                               padding_value=pad_token_id)
    batch_features['token_type_ids'] = pad_sequence([x['token_type_ids'] for x in batch],
                                               batch_first=True,
                                               padding_value=pad_token_id)
    batch_features['n_heads'] = torch.LongTensor([x['n_heads'] for x in batch])
    if 'labels' in batch[0]:
        batch_features['labels'] = torch.LongTensor([x['labels'] for x in batch])
    return batch_features
'''
def get_dataloader(dataset, batch_size, is_shuffle, num_worker = 0):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle,
                            num_workers=num_worker, collate_fn=variable_collate_fn)
    return dataloader
'''
