#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_num(s: str) -> str:
    s = s.strip().replace(",", "")
    if not s:
        return ""
    try:
        # normalize floats like 15.00 -> 15
        f = float(s)
        if f.is_integer():
            return str(int(f))
        # trim trailing zeros
        out = f"{f}".rstrip("0").rstrip(".")
        return out
    except ValueError:
        return s


def gold_answer(ans: str) -> str:
    # GSM8K uses "#### <final>" at the end
    return normalize_num(ans.split("####")[-1].strip())


def pred_answer(text: str) -> str:
    # try tags first, then fall back to last number in whole output
    t = text
    for tag in ["####", "Final answer", "Final Answer", "Answer:", "Ans:", "A:"]:
        if tag in t:
            t = t.rsplit(tag, 1)[-1]
            break
    m = NUM_RE.findall(t.replace(",", ""))
    if m:
        return normalize_num(m[-1])
    # fallback: any number in full output
    m = NUM_RE.findall(text.replace(",", ""))
    return normalize_num(m[-1]) if m else ""


def run_one(
    llama_bin: Path,
    model: Path,
    dataset: Path,
    limit: int,
    args_list,
    progress_every: int,
    n_predict: int,
    dump_failures: int,
    dump_path: Optional[Path],
):
    correct = 0
    total = 0
    start = time.time()

    failures = []
    with dataset.open("r") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item = json.loads(line)
            q = item["question"]
            gold = gold_answer(item["answer"])

            prompt = (
                "Solve the problem. You may show brief reasoning, but finish with a single final number.\n\n"
                f"Question: {q}\nAnswer:"
            )

            cmd = [
                str(llama_bin),
                "-m", str(model),
                "-p", prompt,
                "--single-turn",
                "--no-show-timings",
                "--no-display-prompt",
                "--temp", "0",
                "--top-k", "0",
                "--top-p", "1",
                "--seed", "1",
                "--n-predict", str(n_predict),
                "--ctx-size", "2048",
            ] + args_list

            proc = subprocess.run(cmd, capture_output=True)
            out = proc.stdout.decode("utf-8", errors="replace")
            pred = pred_answer(out)
            if pred == gold:
                correct += 1
            else:
                if dump_failures and len(failures) < dump_failures:
                    failures.append({
                        "index": i,
                        "question": q,
                        "gold": gold,
                        "pred": pred,
                        "output_tail": out[-1200:],
                    })
            total += 1

            if progress_every and total % progress_every == 0:
                acc = correct / total if total else 0.0
                elapsed = time.time() - start
                print(f"{model.name}: {total} done | acc={acc:.4f} | {elapsed/60:.1f} min", flush=True)

    acc = correct / total if total else 0.0

    if dump_failures and failures and dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w") as df:
            for f in failures:
                df.write(json.dumps(f, ensure_ascii=False) + "\n")

    return total, correct, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-bin", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", required=True, nargs="+")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--dump-failures", type=int, default=0)
    ap.add_argument("--dump-path", default="")
    ap.add_argument("--extra-args", default="")
    args = ap.parse_args()

    llama_bin = Path(args.llama_bin)
    dataset = Path(args.dataset)
    models = [Path(m) for m in args.models]
    extra_args = [a for a in args.extra_args.split() if a]

    if not llama_bin.exists():
        print(f"llama bin not found: {llama_bin}", file=sys.stderr)
        return 2
    if not dataset.exists():
        print(f"dataset not found: {dataset}", file=sys.stderr)
        return 2

    print(f"Dataset: {dataset}")
    print(f"Extra args: {' '.join(extra_args) if extra_args else '(none)'}")

    results = []
    dump_path = Path(args.dump_path) if args.dump_path else None
    for model in models:
        if not model.exists():
            print(f"model not found: {model}", file=sys.stderr)
            return 2
        total, correct, acc = run_one(
            llama_bin,
            model,
            dataset,
            args.limit,
            extra_args,
            args.progress_every,
            args.n_predict,
            args.dump_failures,
            dump_path,
        )
        results.append((model.name, total, correct, acc))

    print("\nSummary:")
    for name, total, correct, acc in results:
        print(f"{name}: {correct}/{total} = {acc:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
