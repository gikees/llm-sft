"""Download external data for pretraining T5 from scratch."""
import json
import os

def download_sql_data():
    from datasets import load_dataset
    print("Downloading synthetic_text_to_sql...")
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    out_path = "data/external_train.jsonl"
    with open(out_path, "w") as f:
        for row in ds:
            record = {
                "nl": row.get("sql_prompt", ""),
                "sql": row.get("sql", ""),
                "context": row.get("sql_context", ""),
                "type": "sql",
            }
            if record["nl"] and record["sql"]:
                f.write(json.dumps(record) + "\n")
    print(f"Saved {sum(1 for _ in open(out_path))} SQL examples to {out_path}")


def download_wiki_data():
    from datasets import load_dataset
    print("Downloading wikitext-103...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    out_path = "data/external_wiki.jsonl"
    count = 0
    with open(out_path, "w") as f:
        for row in ds:
            text = row["text"].strip()
            if len(text) > 50:
                f.write(json.dumps({"text": text, "type": "general"}) + "\n")
                count += 1
                if count >= 100000:
                    break
    print(f"Saved {count} wiki examples to {out_path}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download_sql_data()
    download_wiki_data()
