"""Free Hugging Face datasets used by the project.

1) PROMPT_DATASET  : Gustavosta/Stable-Diffusion-Prompts  (~80k text prompts)
2) IMAGE_DATASET   : lambdalabs/pokemon-blip-captions     (image + caption pairs,
                     streamed, so only the few samples you use are downloaded)

Override the image dataset with the env var IMAGE_DATASET_ID (it must have one
image column and one text column).
"""
import os
import random

from PIL import Image

PROMPT_DATASET = "Gustavosta/Stable-Diffusion-Prompts"
IMAGE_DATASET = os.environ.get("IMAGE_DATASET_ID", "lambdalabs/pokemon-blip-captions")

FALLBACK_PROMPTS = [
    "a cozy cabin in a snowy forest at night, warm lights, highly detailed",
    "a futuristic city skyline at sunset, neon reflections, cinematic",
    "a cute robot watering plants in a greenhouse, soft lighting",
    "an astronaut riding a horse on mars, digital art",
    "a peaceful japanese garden with a red bridge, watercolor style",
    "a majestic dragon flying over mountains, fantasy illustration",
    "a bowl of ramen with steam rising, macro photography",
    "an old library with floating books and glowing dust, magical atmosphere",
]

_prompts = None


def _load_prompts():
    global _prompts
    if _prompts is not None:
        return _prompts
    try:
        from datasets import load_dataset
        ds = load_dataset(PROMPT_DATASET, split="train")
        col = "Prompt" if "Prompt" in ds.column_names else ds.column_names[0]
        _prompts = [p for p in ds[col] if isinstance(p, str) and p.strip()]
        print(f"[info] Loaded {len(_prompts)} prompts from {PROMPT_DATASET}")
    except Exception as e:
        print(f"[warn] Could not load prompt dataset ({e}); using built-in prompts.")
        _prompts = FALLBACK_PROMPTS
    return _prompts


def get_random_prompt():
    return random.choice(_load_prompts())


def sample_prompts(n, seed=42):
    prompts = _load_prompts()
    return random.Random(seed).sample(prompts, min(n, len(prompts)))


def load_image_caption_samples(n=4, seed=0):
    """Stream n (real PIL image, caption) pairs from the image-caption dataset."""
    from datasets import load_dataset
    ds = load_dataset(IMAGE_DATASET, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=200)
    samples = []
    for row in ds:
        img, txt = None, None
        for value in row.values():
            if img is None and isinstance(value, Image.Image):
                img = value
            elif txt is None and isinstance(value, str):
                txt = value
        if img is not None and txt:
            samples.append((img.convert("RGB"), txt))
        if len(samples) >= n:
            break
    if not samples:
        raise RuntimeError(f"No image/caption pairs found in {IMAGE_DATASET}")
    return samples
