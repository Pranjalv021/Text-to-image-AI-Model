"""Command-line use:
    python generate.py --prompt "a red fox in the snow"
    python generate.py --random-prompt --num 3 --model segmind/tiny-sd
"""
import argparse

import core
import dataset_utils


def main():
    ap = argparse.ArgumentParser(description="Text-to-image with Hugging Face diffusers")
    ap.add_argument("--prompt")
    ap.add_argument("--random-prompt", action="store_true",
                    help="pick a prompt from the Hugging Face dataset")
    ap.add_argument("--model", default=core.DEFAULT_MODEL)
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--num", type=int, default=1)
    a = ap.parse_args()

    if a.random_prompt or not a.prompt:
        a.prompt = dataset_utils.get_random_prompt()
    print(f"Model : {a.model}\nPrompt: {a.prompt}")

    for i in range(a.num):
        seed = None if a.seed is None else a.seed + i
        _, used, path, secs = core.generate(a.prompt, a.negative, a.steps, a.guidance,
                                            seed, a.size, a.size, model_id=a.model)
        print(f"[{i + 1}/{a.num}] seed={used} time={secs:.1f}s -> {path}")


if __name__ == "__main__":
    main()
