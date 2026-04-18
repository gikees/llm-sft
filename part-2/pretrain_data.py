import json
import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast

PAD_IDX = 0


class PretrainDataset(Dataset):
    def __init__(self, sql_data_path, wiki_data_path=None, max_enc_len=512, max_dec_len=512):
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.data = []

        if sql_data_path and os.path.exists(sql_data_path):
            print(f"Loading SQL data from {sql_data_path}")
            with open(sql_data_path) as f:
                for line in f:
                    try:
                        self.data.append(json.loads(line))
                    except:
                        continue

        if wiki_data_path and os.path.exists(wiki_data_path):
            print(f"Loading wiki data from {wiki_data_path}")
            with open(wiki_data_path) as f:
                for i, line in enumerate(f):
                    if i > 100000:
                        break
                    try:
                        self.data.append(json.loads(line))
                    except:
                        continue

        print(f"Total pretraining examples: {len(self.data)}")
        random.shuffle(self.data)

    def span_corruption(self, text):
        tokens = text.split()
        if len(tokens) < 5:
            return "denoise: " + text, text
        start = random.randint(0, len(tokens) - 3)
        length = random.randint(1, min(5, len(tokens) - start))
        masked_span = " ".join(tokens[start:start + length])
        tokens[start:start + length] = ["<extra_id_0>"]
        input_text = "denoise: " + " ".join(tokens)
        target_text = "<extra_id_0> " + masked_span + " <extra_id_1>"
        return input_text, target_text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        data_type = item.get('type', 'sql')

        if data_type == 'sql':
            nl = item['nl']
            sql = item['sql']
            context = item.get('context', '')
            input_text = f"translate to SQL: {nl}"
            if context:
                input_text += f" | Schema: {context}"
            target_text = sql
        else:
            text = item.get('text', '')[:1000]
            input_text, target_text = self.span_corruption(text)

        enc = self.tokenizer.encode(input_text, add_special_tokens=True, truncation=True, max_length=self.max_enc_len)
        dec = self.tokenizer.encode(target_text, add_special_tokens=True, truncation=True, max_length=self.max_dec_len)

        return torch.tensor(enc, dtype=torch.long), torch.tensor(dec, dtype=torch.long)


def pretrain_collate_fn(batch):
    enc_list = [item[0] for item in batch]
    dec_list = [item[1] for item in batch]

    encoder_ids = pad_sequence(enc_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_padded = pad_sequence(dec_list, batch_first=True, padding_value=PAD_IDX)

    batch_size = decoder_padded.size(0)
    decoder_inputs = torch.cat([
        torch.zeros((batch_size, 1), dtype=torch.long),
        decoder_padded[:, :-1]
    ], dim=1)
    decoder_targets = decoder_padded

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets


def get_pretrain_dataloader(sql_data_path, wiki_data_path, batch_size):
    dset = PretrainDataset(sql_data_path, wiki_data_path)
    return DataLoader(dset, batch_size=batch_size, shuffle=True, collate_fn=pretrain_collate_fn, num_workers=2)
