from __future__ import annotations

"""
Gemma4-12B-it 服务器 QLoRA 训练脚本。
来源：根据原项目 training/lora_train_gemma4.py 的数据格式和文本 LoRA 目标规则改造。
用途：A100 40GB 上训练 4bit QLoRA；不用于 Windows 本地。
"""

import argparse
import gc
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class JsonlDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], processor, max_length: int):
        self.rows, self.processor, self.max_length = rows, processor, max_length

    def __len__(self): return len(self.rows)

    def build_text(self, row: dict[str, Any]) -> str:
        instruction = str(row.get("instruction") or row.get("question") or row.get("prompt") or "").strip()
        context = str(row.get("context") or row.get("input") or "").strip()
        output = str(row.get("output") or row.get("answer") or row.get("response") or "").strip()
        agent_type = str(row.get("agent_type") or "").strip()
        if context: instruction += f"\n\n参考上下文：\n{context}"
        if agent_type: instruction = f"任务类型：{agent_type}\n\n{instruction}"
        if not instruction or not output:
            return ""
        messages = [
            {"role":"user","content":[{"type":"text","text":instruction}]},
            {"role":"assistant","content":[{"type":"text","text":output}]},
        ]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def __getitem__(self, index: int):
        text = self.build_text(self.rows[index])
        if not text:
            text = "<start_of_turn>user\n请解释什么是 RAG。<end_of_turn>\n<start_of_turn>model\nRAG 是检索增强生成。<end_of_turn>"
        item = self.processor(text=[text], return_tensors="pt", truncation=True, max_length=self.max_length, padding=False)
        ids = item["input_ids"][0]
        return {"input_ids":ids, "attention_mask":item["attention_mask"][0], "labels":ids.clone()}


def collate(batch, pad_token_id: int):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    out = {"input_ids":[], "attention_mask":[], "labels":[]}
    for x in batch:
        n = max_len - x["input_ids"].shape[0]
        ids = torch.cat([x["input_ids"], torch.full((n,),pad_token_id,dtype=x["input_ids"].dtype)])
        mask = torch.cat([x["attention_mask"], torch.zeros((n,),dtype=x["attention_mask"].dtype)])
        labels = torch.cat([x["labels"], torch.full((n,),-100,dtype=x["labels"].dtype)])
        labels = labels.masked_fill(ids == pad_token_id, -100)
        out["input_ids"].append(ids); out["attention_mask"].append(mask); out["labels"].append(labels)
    return {k:torch.stack(v) for k,v in out.items()}


def load_rows(path: str):
    rows=[]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row=json.loads(line)
            q=str(row.get("instruction") or row.get("question") or row.get("prompt") or "").strip()
            a=str(row.get("output") or row.get("answer") or row.get("response") or "").strip()
            if q and a: rows.append(row)
    if len(rows)<2: raise ValueError("至少需要 2 条含 instruction/question 与 output/answer 的样本。")
    return rows


def find_targets(model, target: str):
    suffixes=["q_proj","k_proj","v_proj","o_proj"]
    if target=="all": suffixes += ["gate_proj","up_proj","down_proj"]
    exclude=("vision","image","audio","tower","clip","projector","visual","sound","speech")
    names=[]
    for name,module in model.named_modules():
        low=name.lower()
        if not isinstance(module,torch.nn.Linear) or any(x in low for x in exclude): continue
        if any(name.endswith(s) for s in suffixes) and ("language_model" in low or ".layers." in low):
            names.append(name)
    names=list(dict.fromkeys(names))
    if not names: raise RuntimeError("未找到 Gemma4 language_model 文本分支 LoRA 层。请运行原项目的 debug 脚本查看模块名。")
    print("LoRA targets:", len(names)); print("\n".join(names[:30]))
    return names


def metric(path: Path, row: dict):
    row={"timestamp":datetime.now().isoformat(),**row}
    with path.open("a",encoding="utf-8") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")


def args():
    p=argparse.ArgumentParser()
    p.add_argument("--model_path",required=True)
    p.add_argument("--dataset",required=True)
    p.add_argument("--output_dir",required=True)
    p.add_argument("--epochs",type=float,default=1)
    p.add_argument("--eval_size",type=int,default=10)
    p.add_argument("--max_length",type=int,default=512)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--grad_accum",type=int,default=8)
    p.add_argument("--r",type=int,default=8)
    p.add_argument("--alpha",type=int,default=16)
    p.add_argument("--target",choices=["attention","all"],default="attention")
    p.add_argument("--resume_adapter",default="")
    return p.parse_args()


def main():
    a=args()
    if not torch.cuda.is_available(): raise RuntimeError("服务器未识别到 CUDA GPU。")
    print("GPU:",torch.cuda.get_device_name(0))
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"train_config.json").write_text(json.dumps(vars(a),ensure_ascii=False,indent=2),encoding="utf-8")

    processor=AutoProcessor.from_pretrained(a.model_path,trust_remote_code=True,use_fast=False)
    tokenizer=getattr(processor,"tokenizer",None)
    pad_id = tokenizer.pad_token_id if tokenizer and tokenizer.pad_token_id is not None else 0
    if tokenizer and tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token; pad_id=tokenizer.pad_token_id

    rows=load_rows(a.dataset); random.Random(42).shuffle(rows)
    ev=min(max(1,a.eval_size),max(1,len(rows)//10))
    train_rows,eval_rows=rows[ev:],rows[:ev]
    print(f"rows={len(rows)}, train={len(train_rows)}, eval={len(eval_rows)}")

    bnb=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model=AutoModelForImageTextToText.from_pretrained(
        a.model_path, quantization_config=bnb, device_map="auto",
        dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa",
    )
    model.config.use_cache=False
    model=prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    targets=find_targets(model,a.target)
    cfg=LoraConfig(r=a.r,lora_alpha=a.alpha,target_modules=targets,lora_dropout=.05,bias="none",task_type="CAUSAL_LM")
    model=get_peft_model(model,cfg)
    model.print_trainable_parameters()
    model.train()

    train=DataLoader(JsonlDataset(train_rows,processor,a.max_length),batch_size=a.batch,shuffle=True,collate_fn=lambda b:collate(b,pad_id))
    valid=DataLoader(JsonlDataset(eval_rows,processor,a.max_length),batch_size=1,shuffle=False,collate_fn=lambda b:collate(b,pad_id))
    first=next(p for p in model.parameters() if p.requires_grad).device
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr)
    log=out/"metrics_log.jsonl"; optim.zero_grad(set_to_none=True)
    update=0; micro=0; running=0.0

    for epoch in range(math.ceil(a.epochs)):
        for batch in train:
            batch={k:v.to(first) for k,v in batch.items()}
            loss=model(**batch).loss
            if not loss.requires_grad: raise RuntimeError("loss 未连接 LoRA 参数，请检查 target_modules。")
            (loss/a.grad_accum).backward()
            running+=float(loss.detach().cpu()); micro+=1
            if micro % a.grad_accum==0:
                optim.step(); optim.zero_grad(set_to_none=True); update+=1
                tr=running/micro
                metric(log,{"step":update,"split":"train","loss":tr,"epoch":epoch+1,"lr":optim.param_groups[0]["lr"]})
                model.eval(); vals=[]
                with torch.no_grad():
                    for vb in valid:
                        vb={k:v.to(first) for k,v in vb.items()}
                        vals.append(float(model(**vb).loss.detach().cpu()))
                model.train()
                metric(log,{"step":update,"split":"eval","eval_loss":sum(vals)/len(vals),"epoch":epoch+1})
                print({"step":update,"loss":tr,"eval_loss":sum(vals)/len(vals)})
                gc.collect(); torch.cuda.empty_cache()

    adapter=out/"adapter"; model.save_pretrained(adapter)
    try: processor.save_pretrained(adapter)
    except Exception: pass
    print("saved:",adapter)


if __name__=="__main__":
    main()
