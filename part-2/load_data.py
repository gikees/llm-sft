import os, json, random
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt', quiet=True)
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0


def read_schema(schema_path):
    '''Convert the schema JSON into a compact table(col1, col2, ...) string.'''
    with open(schema_path) as f:
        schema = json.load(f)
    parts = []
    for table, cols in schema['ents'].items():
        col_names = ', '.join(cols.keys())
        parts.append(f"{table}({col_names})")
    return ' | '.join(parts)


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

        # Build compact schema string once and reuse for every example
        schema_path = os.path.join(data_folder, 'flight_database.schema')
        self.schema_str = read_schema(schema_path)

        self.encoder_ids = []
        self.decoder_inputs = []
        self.decoder_targets = []
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)

        # Encoder input: task prefix + NL query + schema
        prefixed = [
            f"translate to SQL: {nl} | Schema: {self.schema_str}"
            for nl in nl_lines
        ]
        enc = tokenizer(
            prefixed,
            add_special_tokens=True,
            truncation=True,
            max_length=1024,
        )
        self.encoder_ids = enc['input_ids']

        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)
            dec = tokenizer(
                sql_lines,
                add_special_tokens=True,
                truncation=True,
                max_length=512,
            )
            for sql_ids in dec['input_ids']:
                # decoder_input : [pad] + sql_ids[:-1]  (teacher forcing, EOS removed)
                # decoder_target: sql_ids               (full sequence including EOS)
                self.decoder_inputs.append([tokenizer.pad_token_id] + sql_ids[:-1])
                self.decoder_targets.append(sql_ids)

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        if self.split != 'test':
            return (
                torch.tensor(self.encoder_ids[idx], dtype=torch.long),
                torch.tensor(self.decoder_inputs[idx], dtype=torch.long),
                torch.tensor(self.decoder_targets[idx], dtype=torch.long),
            )
        return torch.tensor(self.encoder_ids[idx], dtype=torch.long)


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Returns:
        encoder_ids:             BxT  — padded encoder input ids
        encoder_mask:            BxT  — 1 for real tokens, 0 for padding
        decoder_inputs:          BxT' — teacher-forcing decoder inputs
        decoder_targets:         BxT' — target token ids (loss computed here)
        initial_decoder_inputs:  Bx1  — start token for autoregressive generation
    '''
    enc = [item[0] for item in batch]
    dec_in = [item[1] for item in batch]
    dec_tgt = [item[2] for item in batch]

    encoder_ids = pad_sequence(enc, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs = pad_sequence(dec_in, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(dec_tgt, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.zeros(len(batch), 1, dtype=torch.long)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Returns:
        encoder_ids:            BxT — padded encoder input ids
        encoder_mask:           BxT — attention mask
        initial_decoder_inputs: Bx1 — start token for generation
    '''
    encoder_ids = pad_sequence(batch, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.zeros(len(batch), 1, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x
