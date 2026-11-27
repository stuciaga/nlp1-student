# Import relevant packages
import torch
import pickle
import argparse
import os
import json
import random
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
from tqdm import tqdm

from prompts import (build_instruction, format_demos, invalid_rate,
                     parse_generated_string as _parse)

class Generation:
    def __init__(self, model_name: str, device: str, 
                max_new_tokens: int = 8, use_bf16: bool = True, adapter_path: str = None,
                max_review_tokens: int = None):
        
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.adapter_path = adapter_path
        self.max_review_tokens = max_review_tokens
        self.n_truncated = 0  # reviews cut to max_review_tokens
        self.n_seen = 0
        self.tokenizer, self.model = self._load_tokenizer_and_model(
            model_name, use_bf16=use_bf16 and self.device == "cuda", adapter_path=adapter_path
        )
        self.model.eval()

    def _load_tokenizer_and_model(self, model_name: str, use_bf16: bool, adapter_path: str):
        kwargs = {}
        if use_bf16 and torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.bfloat16
        
        if model_name == "llama-3b-instruct":
            mid = "meta-llama/Llama-3.2-3B-Instruct"
        elif model_name == "qwen-0.5b-instruct":
            mid = "Qwen/Qwen2.5-0.5B-Instruct"
        elif model_name == "qwen-0.5b-instruct-ft-lora":
            mid = "Qwen/Qwen2.5-0.5B-Instruct"
        else:
            raise ValueError(f"Model {model_name} not found")

        if "ft-lora" in model_name:
            adapter_path = self.adapter_path
            base_model = AutoModelForCausalLM.from_pretrained(mid, **kwargs)
            model = PeftModel.from_pretrained(base_model, adapter_path)
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        else:       
            tokenizer = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid, **kwargs)

        # Left-padding is required for correct batched generation with decoder-only LMs
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        model.to(self.device)
        return tokenizer, model

    def _truncate(self, question: str) -> str:
        self.n_seen += 1
        if self.max_review_tokens is None:
            return question
        ids = self.tokenizer(question, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.max_review_tokens:
            return question
        self.n_truncated += 1
        return self.tokenizer.decode(ids[: self.max_review_tokens])

    def _prompt_ids(self, question: str, instruction: str) -> list:
        question = self._truncate(question)
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Input: {question}\nAnswer:"},
        ]
        t = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        if isinstance(t, torch.Tensor):
            t = t.tolist()
        if isinstance(t, dict):
            t = t["input_ids"]
        # apply_chat_template may return a nested batch of size 1
        if len(t) > 0 and isinstance(t[0], list):
            t = t[0]
        return t

    def _inputs(self, question: str, instruction: str) -> torch.Tensor:
        return torch.tensor([self._prompt_ids(question, instruction)], device=self.device)

    def _batch_inputs(self, questions: list, instruction: str):
        input_ids_list = [self._prompt_ids(q, instruction) for q in questions]
        padded = self.tokenizer.pad(
            {"input_ids": input_ids_list},
            padding=True,
            return_tensors="pt",
        )
        return (
            padded["input_ids"].to(self.device),
            padded["attention_mask"].to(self.device),
        )

    def _gen_kw(self):
        return {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }

    def _decode_new_tokens(self, outputs: torch.Tensor, input_len: int):
        generated_tokens = outputs[:, input_len:]
        generated_strings = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        return generated_tokens, generated_strings

    def generate_greedy(self, question: str, instruction: str):
        generated_tokens, generated_strings = self.generate_greedy_batch(
            [question], instruction
        )
        return generated_tokens, generated_strings[0]

    def generate_greedy_batch(self, questions: list, instruction: str):
        input_ids, attention_mask = self._batch_inputs(questions, instruction)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                temperature=0.0,
                top_k=0,
                **self._gen_kw(),
            )
        return self._decode_new_tokens(outputs, input_ids.shape[-1])

    def generate_samples(self, question: str, instruction: str, num_return_sequences: int = 1, seed: int = 42, 
                         temperature: float = 1.0, top_p: float = 1.0):
        generated_tokens, generated_strings = self.generate_samples_batch(
            [question],
            instruction,
            num_return_sequences=num_return_sequences,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
        )
        return generated_tokens, generated_strings

    def generate_samples_batch(
        self,
        questions: list,
        instruction: str,
        num_return_sequences: int = 1,
        seed: int = 42,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        torch.manual_seed(seed)
        set_seed(seed)

        input_ids, attention_mask = self._batch_inputs(questions, instruction)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                **self._gen_kw(),
            )

        generated_tokens, generated_strings = self._decode_new_tokens(
            outputs, input_ids.shape[-1]
        )
        # Flatten to one string per prompt when num_return_sequences == 1
        if num_return_sequences == 1:
            return generated_tokens, generated_strings

        # Group strings per prompt: [s0_r0, s0_r1, ..., s1_r0, ...]
        grouped = []
        for i in range(len(questions)):
            start = i * num_return_sequences
            grouped.append(generated_strings[start : start + num_return_sequences])
        return generated_tokens, grouped

def parse_generated_string(generated_string: str, order=("negative", "positive")) -> int:
    """Binary parser (0 negative / 1 positive / -1 invalid). See prompts.py."""
    return _parse(generated_string, order)


def sample_demos(train, k_per_class: int, rng: random.Random):
    """k balanced demos per class, shuffled (demo choice/order ablation uses --seed)."""
    demos = []
    for label in (0, 1):
        pool = [x for x in train if x[1] == label]
        demos += rng.sample(pool, k_per_class)
    rng.shuffle(demos)
    return demos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--cv_run", type=int, default=0, help="index into pickle list = which CV seed (0,3,5)")
    ap.add_argument("--instruction_type", type=str, default="0-shot", choices=["0-shot", "few-shot"])
    ap.add_argument("--instruction_id", type=int, default=0, help="wording variant, see prompts.INSTRUCTIONS")
    ap.add_argument("--label_order", type=str, default="negative,positive",
                    help="which label gets letter A first; 'positive,negative' = permutation stress test")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_name", type=str, default="qwen-0.5b-instruct",
                    choices=["qwen-0.5b-instruct", "llama-3b-instruct", "qwen-0.5b-instruct-ft-lora"])
    ap.add_argument("--input_path", type=str, default="part3/data/cross_validation_sets.pkl",
                    help="CV pickle, or a robustness pickle holding a list of (text, label)")
    ap.add_argument("--demo_path", type=str, default="part3/data/cross_validation_sets.pkl",
                    help="CV pickle that few-shot demos are drawn from (fold --fold, default 0)")
    ap.add_argument("--output_path", type=str, default="part3/predictions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapter_path", type=str, default=None)
    ap.add_argument("--num_examples_per_class", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; >0 = sampling")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--subsample", type=int, default=None, help="fixed subsample of N reviews per fold")
    ap.add_argument("--max_review_tokens", type=int, default=512, help="truncate reviews; count is logged")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    order = tuple(args.label_order.split(","))
    base_instruction = build_instruction(args.instruction_id, order)

    with open(args.input_path, "rb") as f:
        data_in = pickle.load(f)
    robustness = not (isinstance(data_in, list) and data_in and isinstance(data_in[0], dict))
    original_model_name = args.model_name.split("-ft")[0]
    os.makedirs(args.output_path, exist_ok=True)

    def load_cv(path):
        with open(path, "rb") as f:
            return pickle.load(f)[args.cv_run]

    def make_instruction(train):
        if args.instruction_type == "0-shot":
            return base_instruction
        demos = sample_demos(train, args.num_examples_per_class, rng)
        return base_instruction + "\n" + format_demos(demos, order)

    def resolve_adapter_path(fold):
        if "ft-lora" not in args.model_name:
            return None
        if args.adapter_path is None:
            raise ValueError("--adapter_path is required when using a fine-tuned LoRA model")
        return f"{args.adapter_path}/{original_model_name}-fold-{fold}/lora-adapter"

    def new_gen(fold):
        return Generation(args.model_name, device=args.device, adapter_path=resolve_adapter_path(fold),
                          max_review_tokens=args.max_review_tokens)

    def free_generation(gen):
        if gen is None:
            return
        del gen.model
        del gen.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def decode(gen, sentences, instruction):
        if args.temperature > 0:
            return gen.generate_samples_batch(sentences, instruction, seed=args.seed,
                                              temperature=args.temperature, top_p=args.top_p)[1]
        return gen.generate_greedy_batch(sentences, instruction)[1]

    def generate_predictions(gen, data, name, instruction):
        data = list(data)
        if args.subsample is not None:  # fixed subsample: same reviews for every config
            data = random.Random(1234).sample(data, min(args.subsample, len(data)))
        n, res = len(data), {}
        gen.n_truncated = gen.n_seen = 0
        t0 = time.time()
        for start in tqdm(range(0, n, args.batch_size), desc=str(name)):
            batch = data[start:start + args.batch_size]
            outs = decode(gen, [s for s, _ in batch], instruction)
            for off, ((sentence, label), out) in enumerate(zip(batch, outs)):
                res[start + off] = {"sentence": sentence, "generated_string": out,
                                    "predicted_label": parse_generated_string(out, order), "label": label}
        # Retry invalid outputs with sampling (kept from the tutorial); first-pass rate is logged
        first_invalid = invalid_rate([r["predicted_label"] for r in res.values()])
        for t in range(3):
            bad = [i for i in range(n) if res[i]["predicted_label"] == -1]
            if not bad:
                break
            for start in range(0, len(bad), args.batch_size):
                idx = bad[start:start + args.batch_size]
                outs = gen.generate_samples_batch([res[i]["sentence"] for i in idx], instruction,
                                                  seed=args.seed + t)[1]
                for i, out in zip(idx, outs):
                    res[i]["generated_string"] = out
                    res[i]["predicted_label"] = parse_generated_string(out, order)
        final_invalid = invalid_rate([r["predicted_label"] for r in res.values()])
        acc = sum(r["predicted_label"] == r["label"] for r in res.values()) / n
        stats = {"name": name, "n": n, "accuracy": acc, "invalid_first_pass": first_invalid,
                 "invalid_final": final_invalid, "truncated": gen.n_truncated, "seen": gen.n_seen,
                 "gpu_minutes": (time.time() - t0) / 60,
                 "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0}
        print(json.dumps(stats))
        tag = f"{args.model_name}_{args.instruction_type}_{name}"
        with open(os.path.join(args.output_path, f"{tag}.json"), "w") as f:
            json.dump({str(k): v for k, v in res.items()}, f, indent=4)
        with open(os.path.join(args.output_path, f"{tag}.stats.json"), "w") as f:
            json.dump({**stats, "args": vars(args)}, f, indent=4)

    if robustness:
        # Model-free datasets (OOD, paraphrase, stress): list of (text, label); named after the file
        demo_fold = args.fold if args.fold is not None else 0
        instruction = make_instruction(load_cv(args.demo_path)[demo_fold]["training"]) \
            if args.instruction_type == "few-shot" else base_instruction
        gen = new_gen(demo_fold)
        generate_predictions(gen, data_in, os.path.splitext(os.path.basename(args.input_path))[0], instruction)
    else:
        cv = data_in[args.cv_run]
        folds = list(cv) if args.fold is None else [args.fold]
        gen = None
        for fold in tqdm(folds):
            if gen is None or "ft-lora" in args.model_name:
                free_generation(gen)
                gen = new_gen(fold)
            generate_predictions(gen, cv[fold]["validation"], fold, make_instruction(cv[fold]["training"]))
        free_generation(gen)
