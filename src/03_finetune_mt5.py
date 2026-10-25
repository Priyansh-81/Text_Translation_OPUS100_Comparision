import argparse, time, os
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback)
from src.common import ROOT, DATA_DIR, OUTPUT_DIR, read_jsonl, append_metric, seed_everything, device_name, score_translation, write_jsonl

MODEL_ID = "google/mt5-small"
OUT = ROOT / "checkpoints" / "mt5-small-en-hi"

class MPSMemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

def main():
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-source-len", type=int, default=64)
    p.add_argument("--max-target-len", type=int, default=64)
    p.add_argument("--max-test", type=int, default=100)
    args = p.parse_args()
    seed_everything()
    train_rows, val_rows, test_rows = [read_jsonl(DATA_DIR / f"{s}.jsonl") for s in ("train","validation","test")]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    def preprocess(batch):
        x = tok(["translate English to Hindi: " + s for s in batch["source"]], max_length=args.max_source_len, truncation=True)
        y = tok(text_target=batch["reference"], max_length=args.max_target_len, truncation=True)
        x["labels"] = y["input_ids"]
        return x
    tr = Dataset.from_list(train_rows).map(preprocess, batched=True, remove_columns=["source","reference"])
    va = Dataset.from_list(val_rows).map(preprocess, batched=True, remove_columns=["source","reference"])
    collator = DataCollatorForSeq2Seq(tok, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUT), learning_rate=5e-5, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01, predict_with_generate=False, eval_strategy="epoch",
        save_strategy="no", logging_steps=25, report_to="none",
        fp16=torch.cuda.is_available(), gradient_accumulation_steps=4
    )
    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=tr, eval_dataset=va,
                             data_collator=collator, processing_class=tok,
                             callbacks=[MPSMemoryCallback()])
    t0=time.perf_counter(); trainer.train(); train_seconds=time.perf_counter()-t0

    OUT.mkdir(parents=True, exist_ok=True); trainer.save_model(str(OUT)); tok.save_pretrained(str(OUT))
    test_rows = test_rows[:args.max_test]
    refs=[]; hyps=[]; infer=[]
    model.eval()
    device = next(model.parameters()).device
    for row in test_rows:
        inp = tok("translate English to Hindi: " + row["source"], return_tensors="pt",
                  max_length=args.max_source_len, truncation=True).to(device)
        t0=time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=args.max_target_len, num_beams=4)
        infer.append(time.perf_counter()-t0)
        hyps.append(tok.decode(out[0], skip_special_tokens=True).strip()); refs.append(row["reference"])
    write_jsonl(OUTPUT_DIR/"mt5_translations.jsonl",
                [{"source":r["source"],"reference":r["reference"],"hypothesis":h} for r,h in zip(test_rows,hyps)])
    bleu,chrf=score_translation(refs,hyps)
    total=sum(p.numel() for p in model.parameters()); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    append_metric({"system":"mT5-small","architecture":"Encoder-decoder","adaptation":"Fine-tuned",
                   "bleu":bleu,"chrf":chrf,"training_seconds":round(train_seconds,2),
                   "avg_inference_seconds":round(sum(infer)/max(1,len(infer)),4),
                   "total_parameters":total,"trainable_parameters":trainable,"device":device_name(),
                   "notes":"Fine-tuned on selected OPUS-100 English-Hindi subset."})
    print("Saved model, translations, and metrics.")

if __name__ == "__main__":
    main()
