import argparse, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.common import ROOT, DATA_DIR, OUTPUT_DIR, read_jsonl, append_metric, seed_everything, device_name, score_translation, write_jsonl

MODELS = {
    "llama": ["meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"],
    "gemma": ["google/gemma-2-2b-it", "unsloth/gemma-2-2b-it"]
}

def load_tok_model(name):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    for cid in MODELS[name]:
        try:
            print(f"Attempting to load {cid} on {device} ({dtype})...")
            tok = AutoTokenizer.from_pretrained(cid)
            if tok.pad_token_id is None: tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(cid, torch_dtype=dtype).to(device)
            return tok, model, cid, device
        except Exception as e:
            print(f"Could not load {cid}: {e}")
    raise RuntimeError(f"Could not load any candidate for {name}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model", choices=["llama","gemma"], required=True)
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    args=p.parse_args()
    seed_everything()
    tok, model, model_id, device = load_tok_model(args.model)
    rows=read_jsonl(DATA_DIR/"test.jsonl")[:args.max_samples]
    refs=[]; hyps=[]; infer=[]; outrows=[]
    model.eval()
    print(f"Running inference for {args.model} ({model_id}) on {len(rows)} samples...")
    for idx, row in enumerate(rows):
        messages=[{"role":"user","content":
            "Translate the following English text into natural Hindi. Output only the Hindi translation; do not explain or add information.\n\nEnglish: "+row["source"]}]
        prompt=tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp=tok(prompt, return_tensors="pt").to(device)
        gen_kwargs={"max_new_tokens":args.max_new_tokens,"do_sample":args.temperature>0}
        if args.temperature>0: gen_kwargs["temperature"]=args.temperature
        t0=time.perf_counter()
        with torch.no_grad():
            out=model.generate(**inp, **gen_kwargs, pad_token_id=tok.pad_token_id)
        infer.append(time.perf_counter()-t0)
        new_tokens=out[0][inp["input_ids"].shape[1]:]
        hyp=tok.decode(new_tokens, skip_special_tokens=True).strip()
        refs.append(row["reference"]); hyps.append(hyp)
        outrows.append({"source":row["source"],"reference":row["reference"],"hypothesis":hyp})
        if (idx + 1) % 25 == 0 or (idx + 1) == len(rows):
            print(f"[{idx+1}/{len(rows)}] processed in {sum(infer):.1f}s")
    filename=f"{args.model}_translations.jsonl"
    write_jsonl(OUTPUT_DIR/filename,outrows)
    bleu,chrf=score_translation(refs,hyps)
    total=sum(p.numel() for p in model.parameters())
    append_metric({"system":"LLaMA 3.2 1B Instruct" if args.model=="llama" else "Gemma 2 2B IT",
                   "architecture":"Decoder-only","adaptation":"Prompt-only","bleu":bleu,"chrf":chrf,
                   "training_seconds":0,"avg_inference_seconds":round(sum(infer)/max(1,len(infer)),4),
                   "total_parameters":total,"trainable_parameters":0,"device":device_name(),
                   "notes":f"Prompt-only; model_id={model_id}; temperature={args.temperature}; tested on same {len(rows)} test set."})
    print(f"Saved {filename} and metrics. BLEU: {bleu}, chrF: {chrf}")

if __name__=="__main__":
    main()

