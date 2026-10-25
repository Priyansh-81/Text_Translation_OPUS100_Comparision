import argparse
from datasets import load_dataset
from src.common import DATA_DIR, seed_everything, write_jsonl

def extract_pairs(split, limit, seed):
    # OPUS-100 en-hi examples typically expose translation dictionaries.
    ds = split.shuffle(seed=seed)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = []
    for item in ds:
        trans = item.get("translation", {})
        en, hi = trans.get("en", ""), trans.get("hi", "")
        if en and hi:
            rows.append({"source": en.strip(), "reference": hi.strip()})
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Helsinki-NLP/opus-100")
    p.add_argument("--config", default="en-hi")
    p.add_argument("--max-train", type=int, default=2000)
    p.add_argument("--max-val", type=int, default=300)
    p.add_argument("--max-test", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_everything(args.seed)
    ds = load_dataset(args.dataset, args.config)
    # OPUS-100 commonly names validation split "validation"; verify actual splits on first run.
    split_names = ds.keys()
    print("Available splits:", list(split_names))
    val_name = "validation" if "validation" in split_names else ("dev" if "dev" in split_names else None)
    if val_name is None or "train" not in split_names or "test" not in split_names:
        raise RuntimeError(f"Unexpected dataset splits {list(split_names)}. Update split mapping after inspecting dataset card.")
    train = extract_pairs(ds["train"], args.max_train, args.seed)
    val = extract_pairs(ds[val_name], args.max_val, args.seed)
    test = extract_pairs(ds["test"], args.max_test, args.seed)
    for name, rows in [("train", train), ("validation", val), ("test", test)]:
        path = DATA_DIR / f"{name}.jsonl"
        write_jsonl(path, rows)
        print(f"{name}: {len(rows)} examples -> {path}")
    print("Example:", train[0] if train else "No valid examples")

if __name__ == "__main__":
    main()
