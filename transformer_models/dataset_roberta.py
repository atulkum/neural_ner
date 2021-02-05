
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import pandas as pd
from ast import literal_eval
from torch.nn import CrossEntropyLoss

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
id2tag = ['O', 'B-toxic', 'I-toxic']
tag2id = {v:k for k, v in enumerate(id2tag)}
tag_pad_id = CrossEntropyLoss().ignore_index

def encode_roberta(sentence):
    sentence_tokens = [tokenizer.tokenize(sentence[0])] + \
                      [tokenizer.tokenize(f' {t}') for t in sentence[1:]]
    sentence_ids = [tokenizer.convert_tokens_to_ids(t) for t in sentence_tokens]
    start_idx_mask = []
    all_ids = []
    for subwords in sentence_ids:
        curr_mask = [1]
        if len(subwords) > 1:
            curr_mask += [0] * (len(subwords) - 1)
        start_idx_mask.extend(curr_mask)
        all_ids.extend(subwords)
    special_token_mask = tokenizer.get_special_tokens_mask(all_ids)

    prefix_offset = 0
    while prefix_offset < len(special_token_mask) and special_token_mask[prefix_offset] == 1:
        prefix_offset += 1
    suffix_offset = len(special_token_mask) - len(start_idx_mask) - prefix_offset
    start_idx_mask = [0] * prefix_offset + start_idx_mask + [0] * suffix_offset

    sentence_inputs = tokenizer.prepare_for_model(all_ids, add_special_tokens=True)
    input_ids = sentence_inputs["input_ids"]
    attention_mask = sentence_inputs["attention_mask"]
    #######
    inputs = tokenizer(
        text=' '.join(sentence),
        add_special_tokens=True
    )
    assert inputs["input_ids"] == input_ids
    assert inputs["attention_mask"] == attention_mask
    #######
    return input_ids, attention_mask, start_idx_mask

def get_labels_tokens(orig_sentence, chunks):
    curr = 0
    labels = []
    tokens = []
    for s, e in chunks:
        other_txt = orig_sentence[curr:s].split()
        label_txt = orig_sentence[s:e + 1].split()
        curr = e + 1
        tokens.extend(other_txt)
        labels.extend(['O'] * len(other_txt))

        tokens.append(label_txt[0])
        labels.append('B-toxic')
        for k in range(1, len(label_txt)):
            tokens.append(label_txt[k])
            labels.append('I-toxic')
    if curr < len(orig_sentence):
        other_txt = orig_sentence[curr:].split()
        tokens.extend(other_txt)
        labels.extend(['O'] * len(other_txt))
    return tokens, labels

def get_chunks(span):
    chunks = []
    curr_start = None
    for span_i, t in enumerate(span):
        if span_i == 0 or curr_start is None:
            curr_start = t
        elif t > span[span_i - 1] + 1:
            chunks.append((curr_start, span[span_i - 1]))
            curr_start = t
    if curr_start is not None:
        chunks.append((curr_start, span[-1]))
    return chunks

def get_text_from_ids(input_ids):
    return tokenizer.convert_tokens_to_string(
        [tokenizer._convert_id_to_token(input_id) for input_id in input_ids])

class SpanDataset(Dataset):
    def __getitem__(self, n):
        return self._features[n]

    def __len__(self):
        return len(self._features)

    def __init__(self, phase):
        self._phase = phase
        self.init_dataset()

    def init_dataset(self):
        train = pd.read_csv("tsd_train.csv")
        sentences = train['text']
        if self._phase in {'train', 'dev'}:
            spans = train.spans.apply(literal_eval)
        max_seq_len = -1
        max_token_len = -1
        features = []
        for i, orig_sentence in enumerate(sentences):
            chunks = []
            if self._phase in {'train', 'dev'}:
                chunks = get_chunks(spans[i])

            tokens, labels = get_labels_tokens(orig_sentence, chunks)
            # roberta tokenization
            input_ids, attention_mask, start_idx_mask = encode_roberta(tokens)
            max_seq_len = max(max_seq_len, len(input_ids))
            max_token_len = max(max_token_len, len(labels))
            labels_ids = [tag2id[k] for k in labels]
            padded_labels_ids = labels_ids + [tag_pad_id]*(200 - len(labels_ids))
            datum = {
                'input_ids': torch.LongTensor(input_ids),
                'attention_mask': torch.LongTensor(attention_mask),
                'start_idx_mask': torch.BoolTensor(start_idx_mask),
                'labels': torch.LongTensor(labels_ids),
                'padded_labels': torch.LongTensor(padded_labels_ids)
            }
            features.append(datum)
        print(f'max_seq_len {max_seq_len} max_token_len {max_token_len}')
        self._features = features

def variable_collate_fn(batch):
    batch_features = {}

    batch_features['input_ids'] = pad_sequence([x['input_ids'] for x in batch],
                                               batch_first=True,
                                               padding_value=tokenizer.pad_token_id)
    batch_features['attention_mask'] = pad_sequence([x['attention_mask'] for x in batch],
                                               batch_first=True,
                                               padding_value=0)
    batch_features['start_idx_mask'] = pad_sequence([x['start_idx_mask'] for x in batch],
                                               batch_first=True,
                                               padding_value=0)
    if 'labels' in batch[0]:
        batch_features['labels'] = pad_sequence([x['labels'] for x in batch],
                                               batch_first=True,
                                               padding_value=tag_pad_id)
        batch_features['padded_labels'] = pad_sequence([x['padded_labels'] for x in batch],
                                                batch_first=True,
                                                padding_value=tag_pad_id)
    return batch_features

if __name__ == '__main__':
    data_iter = SpanDataset('dev')
    for d in data_iter:
        print(d)
        break
