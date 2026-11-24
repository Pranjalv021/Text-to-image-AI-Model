# Text-to-Image AI Project (Hugging Face, 100% free)

Generate images from text with free open models, test them on free Hugging Face
datasets, and measure quality with CLIP. No API key, no login, no payment.

| Part | What is used |
|---|---|
| Models | `stabilityai/sd-turbo` (default), `segmind/tiny-sd` (lighter) |
| Datasets | `Gustavosta/Stable-Diffusion-Prompts`, `lambdalabs/pokemon-blip-captions` |
| Metric | CLIP score with `openai/clip-vit-base-patch32` |
| Libraries | diffusers, transformers, datasets, torch, gradio |

## Setup on Windows + VS Code
1. Install Python 3.10 - 3.12 from python.org (tick "Add python.exe to PATH").
2. Unzip this folder; in VS Code use File > Open Folder.
3. Open the terminal (Ctrl+`) and run:

       python -m venv venv
       venv\Scripts\activate.bat          (PowerShell: venv\Scripts\Activate.ps1)
       pip install -r requirements.txt
       python app.py

   Or double-click `run.bat`.
4. Your browser opens at http://127.0.0.1:7860

Internet is needed on the first run only (model ~5 GB on CPU / ~2.5 GB on NVIDIA
GPU, CLIP ~600 MB). Models are cached afterwards. Images go to `outputs/`.

## What you can do in the app
* **Generate** tab - choose a model, type a prompt (or click "Random prompt" to take
  one from the dataset), get an image, time and CLIP score.
* **Real vs AI (dataset)** tab - streams real image+caption pairs, generates an AI
  image for each caption, shows both side by side with scores.
* **About** tab - project summary.

## Command line (for report results)
    python generate.py --prompt "a red fox in the snow"
    python evaluate.py --models stabilityai/sd-turbo segmind/tiny-sd --num 5
    python evaluate.py --source dataset --num 5
`evaluate.py` saves a CSV in `results/` and prints mean time and CLIP score per
model - paste these numbers into the Results section of REPORT.md.

## Faster with an NVIDIA GPU (optional)
    pip uninstall -y torch
    pip install torch --index-url https://download.pytorch.org/whl/cu121

## Troubleshooting
* torch / pygame-style build errors: use Python 3.11 or 3.12.
* Slow on CPU (about 20-60 s per image for sd-turbo): lower image size to 384, or
  keep steps at 1-2. tiny-sd needs ~25 steps, so it is slower on CPU.
* Out of memory: image size 256-384.
* Dataset tab fails: check internet; the first load streams data from Hugging Face.
* C: drive full: set `HF_HOME=D:\hf_cache` before running to move the model cache.

## Files
    app.py            web UI (3 tabs)          core.py         model loading + generation
    generate.py       command-line generator   dataset_utils.py HF dataset loading
    evaluate.py       batch evaluation -> CSV  evaluation.py   CLIP metrics
    REPORT.md         report outline + viva questions
