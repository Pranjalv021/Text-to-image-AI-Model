"""Batch evaluation for your report (writes a CSV + prints a summary).

    python evaluate.py --num 5
    python evaluate.py --models stabilityai/sd-turbo segmind/tiny-sd --num 5
    python evaluate.py --source dataset --num 5     (adds real-vs-AI similarity)

All models are tested on the SAME prompts/seeds, so the comparison is fair.
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import core
import dataset_utils
import evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[core.DEFAULT_MODEL])
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--source", choices=["prompts", "dataset"], default="prompts")
    a = ap.parse_args()

    if a.source == "dataset":
        items = [(cap, real) for real, cap in
                 dataset_utils.load_image_caption_samples(a.num, seed=42)]
    else:
        items = [(p, None) for p in dataset_utils.sample_prompts(a.num, seed=42)]

    rows = []
    for model in a.models:
        for i, (prompt, real) in enumerate(items):
            print(f"[{model}] {i + 1}/{len(items)}: {prompt[:60]}")
            img, seed, _, secs = core.generate(prompt, seed=i, model_id=model)
            row = {"model": model, "prompt": prompt, "seed": seed,
                   "seconds": round(secs, 2),
                   "clip_score": round(evaluation.clip_score(img, prompt), 3)}
            if real is not None:
                row["real_similarity"] = round(evaluation.image_similarity(real, img), 3)
            rows.append(row)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"evaluation_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    fields = ["model", "prompt", "seed", "seconds", "clip_score"]
    if a.source == "dataset":
        fields.append("real_similarity")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n=== Summary (mean over prompts) ===")
    for model in a.models:
        r = [x for x in rows if x["model"] == model]
        line = (f"{model}: time={statistics.mean(x['seconds'] for x in r):.1f}s  "
                f"CLIP={statistics.mean(x['clip_score'] for x in r):.2f}")
        if a.source == "dataset":
            line += f"  real-sim={statistics.mean(x['real_similarity'] for x in r):.3f}"
        print(line)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
