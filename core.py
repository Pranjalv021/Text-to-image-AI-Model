"""Core text-to-image engine (Hugging Face diffusers).

Free models (open weights, no login needed):
  * stabilityai/sd-turbo  - distilled Stable Diffusion, 1-4 steps, fastest on CPU
  * segmind/tiny-sd       - smaller distilled Stable Diffusion 1.5, lighter download
Choose the default with the env var T2I_MODEL, or pick one in the web UI.
"""
import gc
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from diffusers import AutoPipelineForText2Image

MODELS = {
    "stabilityai/sd-turbo": {"steps": 2, "guidance": 0.0, "size": 512},
    "segmind/tiny-sd": {"steps": 25, "guidance": 7.5, "size": 512},
}
DEFAULT_MODEL = os.environ.get("T2I_MODEL", "stabilityai/sd-turbo")
OUTPUT_DIR = Path(__file__).parent / "outputs"

_pipe = None
_pipe_id = None


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def settings_for(model_id):
    """Recommended steps / guidance / size for a model."""
    if model_id in MODELS:
        return dict(MODELS[model_id])
    if "turbo" in model_id.lower():
        return {"steps": 2, "guidance": 0.0, "size": 512}
    return {"steps": 25, "guidance": 7.5, "size": 512}


def load_pipeline(model_id=None):
    """Load a model (cached). The first call downloads it from Hugging Face."""
    global _pipe, _pipe_id
    model_id = model_id or DEFAULT_MODEL
    if _pipe is not None and _pipe_id == model_id:
        return _pipe

    _pipe, _pipe_id = None, None          # free the previous model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = get_device()
    use_fp16 = device == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32
    print(f"[info] Loading '{model_id}' on {device} (first run downloads it)...")

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=dtype, variant="fp16" if use_fp16 else None)
    except Exception:
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)

    pipe = pipe.to(device)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    _pipe, _pipe_id = pipe, model_id
    print("[info] Model ready.")
    return pipe


def generate(prompt, negative_prompt="", steps=None, guidance=None, seed=None,
             width=512, height=512, model_id=None, save=True):
    """Generate one image -> (PIL image, seed used, saved path or None, seconds)."""
    if not prompt or not prompt.strip():
        raise ValueError("Please enter a prompt.")
    model_id = model_id or DEFAULT_MODEL
    d = settings_for(model_id)
    steps = int(steps if steps is not None else d["steps"])
    guidance = float(guidance if guidance is not None else d["guidance"])
    width, height = int(width) // 8 * 8, int(height) // 8 * 8

    if seed is None or int(seed) < 0:
        seed = random.randint(0, 2**31 - 1)
    seed = int(seed)

    pipe = load_pipeline(model_id)
    gen_device = "cuda" if get_device() == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)

    kwargs = dict(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
                  width=width, height=height, generator=generator)
    if negative_prompt and negative_prompt.strip() and guidance > 1.0:
        kwargs["negative_prompt"] = negative_prompt

    t0 = time.perf_counter()
    with torch.inference_mode():
        image = pipe(**kwargs).images[0]
    secs = time.perf_counter() - t0

    path = None
    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower())[:40].strip("-") or "image"
        name = model_id.split("/")[-1]
        path = OUTPUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}_{seed}_{slug}.png"
        image.save(path)
    return image, seed, path, secs
