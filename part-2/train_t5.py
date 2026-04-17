import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model
    parser.add_argument('--finetune', action='store_true', help="Finetune pretrained T5 (default: train from scratch)")

    # Training
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"])
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"])
    parser.add_argument('--num_warmup_epochs', type=int, default=1)
    parser.add_argument('--max_n_epochs', type=int, default=20)
    parser.add_argument('--patience_epochs', type=int, default=3)

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--experiment_name', type=str, default='baseline')

    # Data
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    # Training stability
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=1, help="Evaluate every N epochs")

    # Generation
    parser.add_argument('--max_gen_length', type=int, default=512)
    parser.add_argument('--num_beams', type=int, default=10)

    return parser.parse_args()


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    gt_sql_path = 'data/dev.sql'
    gt_record_path = 'records/ground_truth_dev.pkl'
    model_sql_path = f'results/t5_{model_type}_{args.experiment_name}_dev.sql'
    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_dev.pkl'

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch + 1}/{args.max_n_epochs} — train loss: {tr_loss:.4f}")

        if (epoch + 1) % args.eval_every != 0:
            continue

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        print(f"  dev loss: {eval_loss:.4f}  F1: {record_f1:.4f}  EM: {record_em:.4f}  SQL-EM: {sql_em:.4f}  err: {error_rate*100:.1f}%")

        if args.use_wandb:
            import wandb
            wandb.log({
                'train/loss': tr_loss, 'dev/loss': eval_loss,
                'dev/record_f1': record_f1, 'dev/record_em': record_em,
                'dev/sql_em': sql_em, 'dev/error_rate': error_rate,
            }, step=epoch)

        save_model(checkpoint_dir, model, best=False)
        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
            save_model(checkpoint_dir, model, best=True)
            print(f"  ✓ new best F1: {best_f1:.4f}")
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= args.patience_epochs:
            print(f"Early stopping after {epoch + 1} epochs (best F1: {best_f1:.4f})")
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    accum_steps = args.gradient_accumulation_steps

    optimizer.zero_grad()
    for step, (encoder_input, encoder_mask, decoder_input, decoder_targets, _) in enumerate(tqdm(train_loader, desc="train")):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_targets.reshape(-1))
        loss = loss / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        with torch.no_grad():
            num_tokens = (decoder_targets != PAD_IDX).sum().item()
            total_loss += loss.item() * accum_steps * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens if total_tokens > 0 else 0


def eval_epoch(args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path):
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    total_loss = 0
    total_tokens = 0
    all_sql_queries = []

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader, desc="eval"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            # Loss
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']
            loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_targets.reshape(-1))
            num_tokens = (decoder_targets != PAD_IDX).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Generation
            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=args.max_gen_length,
                num_beams=args.num_beams,
                early_stopping=True,
            )
            for seq in generated:
                all_sql_queries.append(tokenizer.decode(seq, skip_special_tokens=True))

    save_queries_and_records(all_sql_queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    error_rate = sum(1 for m in error_msgs if m) / len(error_msgs) if error_msgs else 0
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0

    return avg_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    all_sql_queries = []

    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader, desc="test inference"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=args.max_gen_length,
                num_beams=args.num_beams,
                early_stopping=True,
            )
            for seq in generated:
                all_sql_queries.append(tokenizer.decode(seq, skip_special_tokens=True))

    os.makedirs(os.path.dirname(model_sql_path) or '.', exist_ok=True)
    save_queries_and_records(all_sql_queries, model_sql_path, model_record_path)


def main():
    args = get_args()
    print(f"Mode: {'fine-tuning' if args.finetune else 'from scratch'}  |  device: {DEVICE}")

    if args.use_wandb:
        setup_wandb(args)

    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Final evaluation with best checkpoint
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = 'data/dev.sql'
    gt_record_path = 'records/ground_truth_dev.pkl'
    model_sql_path = f'results/t5_{model_type}_{args.experiment_name}_dev.sql'
    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_dev.pkl'

    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    print(f"\nFinal dev — Loss: {dev_loss:.4f}  F1: {dev_record_f1:.4f}  EM: {dev_record_em:.4f}  SQL-EM: {dev_sql_em:.4f}  err: {dev_error_rate*100:.1f}%")

    # Test inference
    model_sql_path = f'results/t5_{model_type}_{args.experiment_name}_test.sql'
    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_test.pkl'
    test_inference(args, model, test_loader, model_sql_path, model_record_path)
    print(f"Test predictions saved to {model_sql_path}")


if __name__ == "__main__":
    main()
