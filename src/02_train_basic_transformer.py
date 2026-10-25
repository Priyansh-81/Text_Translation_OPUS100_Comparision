import argparse, time, math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from src.common import ROOT, DATA_DIR, OUTPUT_DIR, RESULTS_DIR, read_jsonl, append_metric, seed_everything, device_name

TOKENIZER_ID = "google/mt5-small"  # Used only as a shared subword tokenizer; the model weights are NOT used.
CKPT_DIR = ROOT / "checkpoints" / "basic_transformer"

class PairDataset(Dataset):
    def __init__(self, rows, tok, max_len=96):
        self.rows, self.tok, self.max_len = rows, tok, max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        src = self.tok(r["source"], max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        tgt = self.tok(r["reference"], max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        return src.input_ids.squeeze(0), src.attention_mask.squeeze(0), tgt.input_ids.squeeze(0)

class ScratchTransformer(nn.Module):
    def __init__(self, vocab, pad, d_model=256, heads=4, layers=3, ff=512, max_len=128):
        super().__init__()
        self.pad = pad
        self.src_emb = nn.Embedding(vocab, d_model, padding_idx=pad)
        self.tgt_emb = nn.Embedding(vocab, d_model, padding_idx=pad)
        self.pos = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=heads, dim_feedforward=ff, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.out = nn.Linear(d_model, vocab)
        self.scale = math.sqrt(d_model)

    def encode(self, src):
        sl = src.shape[1]
        spos = torch.arange(sl, device=src.device).unsqueeze(0)
        s = self.src_emb(src) * self.scale + self.pos(spos)
        return self.encoder(s, src_key_padding_mask=(src == self.pad))

    def decode(self, tgt_in, memory, src_pad_mask=None):
        tl = tgt_in.shape[1]
        tpos = torch.arange(tl, device=tgt_in.device).unsqueeze(0)
        t = self.tgt_emb(tgt_in) * self.scale + self.pos(tpos)
        causal = torch.triu(torch.ones(tl, tl, device=tgt_in.device, dtype=torch.bool), diagonal=1)
        return self.decoder(t, memory, tgt_mask=causal,
                            tgt_key_padding_mask=(tgt_in == self.pad),
                            memory_key_padding_mask=src_pad_mask)

    def forward(self, src, tgt_in):
        src_mask = (src == self.pad)
        memory = self.encode(src)
        out = self.decode(tgt_in, memory, src_pad_mask=src_mask)
        return self.out(out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=96)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-test", type=int, default=100)
    args = p.parse_args()
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    train_rows, val_rows, test_rows = [read_jsonl(DATA_DIR / f"{s}.jsonl") for s in ("train","validation","test")]
    train_dl = DataLoader(PairDataset(train_rows, tok, args.max_len), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(PairDataset(val_rows, tok, args.max_len), batch_size=args.batch_size)
    model = ScratchTransformer(len(tok), tok.pad_token_id, d_model=args.d_model, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); total = 0
        for src, mask, tgt in train_dl:
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad()
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item()
        model.eval(); vloss = 0
        with torch.no_grad():
            for src, mask, tgt in val_dl:
                src, tgt = src.to(device), tgt.to(device)
                logits = model(src, tgt[:, :-1])
                vloss += loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1)).item()
        print(f"epoch={epoch+1} train_loss={total/max(1,len(train_dl)):.4f} val_loss={vloss/max(1,len(val_dl)):.4f}")
    train_seconds = time.perf_counter() - start
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": vars(args)}, CKPT_DIR / "model.pt")
    tok.save_pretrained(CKPT_DIR / "tokenizer")

    # Greedy autoregressive decoding
    test_eval_rows = test_rows[:args.max_test]
    model.eval(); hypotheses=[]; references=[]; infer_times=[]
    for row in test_eval_rows:
        src = tok(row["source"], max_length=args.max_len, truncation=True, return_tensors="pt").input_ids.to(device)
        generated = torch.tensor([[tok.pad_token_id]], device=device)
        if tok.eos_token_id is not None: generated[0,0] = tok.eos_token_id
        t0 = time.perf_counter()
        with torch.no_grad():
            mem = model.encode(src)
            src_mask = (src == model.pad)
            for _ in range(args.max_new_tokens):
                dec_out = model.decode(generated, mem, src_mask)
                nxt = model.out(dec_out[:, -1:, :]).argmax(-1)
                generated = torch.cat([generated, nxt], dim=1)
                if tok.eos_token_id is not None and nxt.item() == tok.eos_token_id: break
        infer_times.append(time.perf_counter()-t0)
        hypotheses.append(tok.decode(generated[0,1:], skip_special_tokens=True).strip())
        references.append(row["reference"])
    from src.common import write_jsonl, score_translation
    write_jsonl(OUTPUT_DIR / "basic_transformer_translations.jsonl",
                [{"source": r["source"], "reference": r["reference"], "hypothesis": h} for r,h in zip(test_eval_rows,hypotheses)])
    bleu, chrf = score_translation(references, hypotheses)
    append_metric({"system":"Basic Transformer","architecture":"Encoder-decoder","adaptation":"From scratch",
                   "bleu":bleu,"chrf":chrf,"training_seconds":round(train_seconds,2),
                   "avg_inference_seconds":round(sum(infer_times)/max(1,len(infer_times)),4),
                   "total_parameters":sum(x.numel() for x in model.parameters()),"trainable_parameters":sum(x.numel() for x in model.parameters()),
                   "device":device_name(),"notes":"Shared mT5 tokenizer only; Transformer weights initialized from scratch."})
    print("Saved checkpoint, translations, and metrics.")

if __name__ == "__main__":
    main()

