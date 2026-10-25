import argparse, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.common import ROOT, DATA_DIR, RESULTS_DIR, read_jsonl, score_translation, device_name
import pandas as pd

MODELS={
    "llama": ["meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"],
    "gemma": ["google/gemma-2-2b-it", "unsloth/gemma-2-2b-it"]
}

def load_tok_model(name):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    for cid in MODELS[name]:
        try:
            print(f"Loading {cid} on {device} ({dtype})...")
            tok = AutoTokenizer.from_pretrained(cid)
            if tok.pad_token_id is None: tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(cid, torch_dtype=dtype).to(device)
            return tok, model, cid, device
        except Exception as e:
            print(f"Could not load {cid}: {e}")
    raise RuntimeError(f"Could not load any candidate for {name}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=["llama","gemma"],required=True)
    p.add_argument("--samples",type=int,default=30)
    p.add_argument("--temperatures",type=float,nargs="+",default=[0.0,0.3,0.7,1.0])
    args=p.parse_args()
    tok, model, mid, device = load_tok_model(args.model)
    model.eval()

    rows=read_jsonl(DATA_DIR/"test.jsonl")[:args.samples]
    results=[]
    for temp in args.temperatures:
        refs=[]; hyps=[]; elapsed=[]
        for row in rows:
            messages=[{"role":"user","content":"Translate English to Hindi. Return only the translation.\nEnglish: "+row["source"]}]
            prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inp=tok(prompt,return_tensors="pt").to(device)
            args_gen={"max_new_tokens":96,"do_sample":temp>0}
            if temp>0: args_gen["temperature"]=temp
            t0=time.perf_counter()
            with torch.no_grad(): out=model.generate(**inp,**args_gen,pad_token_id=tok.pad_token_id)
            elapsed.append(time.perf_counter()-t0)
            hyps.append(tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip())
            refs.append(row["reference"])
        bleu,chrf=score_translation(refs,hyps)
        results.append({"model":args.model,"temperature":temp,"bleu":bleu,"chrf":chrf,
                        "avg_inference_seconds":sum(elapsed)/max(1,len(elapsed)),
                        "samples":len(rows),"device":device_name()})
        print(results[-1])
    out=RESULTS_DIR/f"temperature_{args.model}.csv"
    pd.DataFrame(results).to_csv(out,index=False)
    print("Saved",out)
if __name__=="__main__": main()
