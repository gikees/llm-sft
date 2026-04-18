import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import T5ForConditionalGeneration, T5Config
import transformers
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from pretrain_data import get_pretrain_dataloader

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sql_data_path', type=str, default='data/external_train.jsonl')
    parser.add_argument('--wiki_data_path', type=str, default='data/external_wiki.jsonl')
    parser.add_argument('--experiment_name', type=str, default='pretrain_v1')
    parser.add_argument('--max_n_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_warmup_epochs', type=int, default=1)
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    return parser.parse_args()


def train_epoch(args, model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    accum = args.gradient_accumulation_steps

    optimizer.zero_grad()
    for step, (enc_ids, enc_mask, dec_in, dec_tgt) in enumerate(tqdm(loader, desc="pretrain")):
        enc_ids = enc_ids.to(DEVICE)
        enc_mask = enc_mask.to(DEVICE)
        dec_in = dec_in.to(DEVICE)
        dec_tgt = dec_tgt.to(DEVICE)

        logits = model(input_ids=enc_ids, attention_mask=enc_mask, decoder_input_ids=dec_in)['logits']
        loss = criterion(logits.reshape(-1, logits.size(-1)), dec_tgt.reshape(-1)) / accum
        loss.backward()

        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        with torch.no_grad():
            n = (dec_tgt != PAD_IDX).sum().item()
            total_loss += loss.item() * accum * n
            total_tokens += n

    return total_loss / total_tokens if total_tokens > 0 else 0


def main():
    args = get_args()
    print(f"Pretraining from scratch | device: {DEVICE}")

    config = T5Config.from_pretrained('google-t5/t5-small')
    model = T5ForConditionalGeneration(config).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    loader = get_pretrain_dataloader(args.sql_data_path, args.wiki_data_path, args.batch_size)

    # Optimizer with weight decay grouping
    decay_params = []
    for name, child in model.named_children():
        for n, p in child.named_parameters():
            if p.requires_grad and not isinstance(child, tuple(ALL_LAYERNORM_LAYERS)) and "bias" not in n:
                decay_params.append(f"{name}.{n}")

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if n in decay_params and p.requires_grad], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n not in decay_params and p.requires_grad], "weight_decay": 0.0},
    ], lr=args.learning_rate)

    steps_per_epoch = len(loader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.max_n_epochs
    warmup_steps = steps_per_epoch * args.num_warmup_epochs
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    checkpoint_dir = os.path.join('checkpoints', 'pretrain', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(args.max_n_epochs):
        loss = train_epoch(args, model, loader, optimizer, scheduler)
        print(f"Epoch {epoch + 1}/{args.max_n_epochs} — loss: {loss:.4f}")
        torch.save({'model_state_dict': model.state_dict(), 'config': model.config},
                   os.path.join(checkpoint_dir, 'last_model.pt'))

    print(f"Pretrained checkpoint saved to {checkpoint_dir}/last_model.pt")


if __name__ == "__main__":
    main()
