"""Web UI (opens in your browser):  python app.py"""
import gradio as gr

import core
import dataset_utils
import evaluation

MODEL_CHOICES = list(core.MODELS.keys())
if core.DEFAULT_MODEL not in MODEL_CHOICES:
    MODEL_CHOICES.insert(0, core.DEFAULT_MODEL)
D = core.settings_for(core.DEFAULT_MODEL)


def safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:           # CLIP download/offline problems etc.
        print(f"[warn] {fn.__name__} failed: {e}")
        return None


def fmt(x, digits=2):
    return "n/a" if x is None else f"{x:.{digits}f}"


def on_model_change(model_id):
    s = core.settings_for(model_id)
    return s["steps"], s["guidance"]


def on_generate(model_id, prompt, negative, steps, guidance, seed, size, want_clip):
    try:
        image, used_seed, path, secs = core.generate(
            prompt, negative, steps, guidance, seed, size, size, model_id=model_id)
    except Exception as e:
        raise gr.Error(str(e))
    msg = f"**Time:** {secs:.1f}s &nbsp; **Seed:** {used_seed}"
    if want_clip:
        msg += f" &nbsp; **CLIP score:** {fmt(safe(evaluation.clip_score, image, prompt))}"
    msg += f"  \n**Saved to:** `{path}`"
    return image, msg


def on_compare(model_id, n, progress=gr.Progress()):
    n = int(n)
    try:
        samples = dataset_utils.load_image_caption_samples(n, seed=7)
    except Exception as e:
        raise gr.Error(f"Could not load dataset (check internet): {e}")

    real_items, gen_items, rows = [], [], []
    for i, (real, cap) in enumerate(samples):
        progress(i / n, desc=f"Generating {i + 1}/{n}")
        try:
            img, _, _, secs = core.generate(cap, seed=i, model_id=model_id)
        except Exception as e:
            raise gr.Error(str(e))
        score = safe(evaluation.clip_score, img, cap)
        sim = safe(evaluation.image_similarity, real, img)
        real_items.append((real, cap))
        gen_items.append((img, f"CLIP {fmt(score, 1)} | similarity {fmt(sim)}"))
        rows.append((cap, secs, score, sim))

    table = "| Caption | Time (s) | CLIP score | Real-vs-AI similarity |\n|---|---|---|---|\n"
    for cap, secs, score, sim in rows:
        table += f"| {cap} | {secs:.1f} | {fmt(score)} | {fmt(sim)} |\n"
    scores = [r[2] for r in rows if r[2] is not None]
    if scores:
        table += f"\n**Mean CLIP score:** {sum(scores) / len(scores):.2f}"
    return real_items, gen_items, table


ABOUT = f"""
### Text-to-Image AI Project
* **Task:** generate images from text (text-to-image synthesis).
* **Models (Hugging Face, free):** `stabilityai/sd-turbo`, `segmind/tiny-sd` (latent diffusion, via `diffusers`).
* **Datasets (Hugging Face, free):** `{dataset_utils.PROMPT_DATASET}` (prompts) and
  `{dataset_utils.IMAGE_DATASET}` (real image + caption pairs).
* **Evaluation:** CLIP score (prompt-image match) and CLIP image similarity (real vs generated).
* **Device in use:** `{core.get_device()}`
"""

with gr.Blocks(title="Text-to-Image AI Project") as demo:
    gr.Markdown("# Text-to-Image AI Project (Hugging Face)")
    with gr.Tabs():
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column():
                    model = gr.Dropdown(MODEL_CHOICES, value=core.DEFAULT_MODEL, label="Model")
                    prompt = gr.Textbox(label="Prompt", lines=3,
                                        placeholder="a cozy cabin in a snowy forest at night")
                    rand_btn = gr.Button("Random prompt (from dataset)")
                    negative = gr.Textbox(label="Negative prompt (ignored by Turbo)")
                    steps = gr.Slider(1, 50, value=D["steps"], step=1, label="Steps")
                    guidance = gr.Slider(0, 15, value=D["guidance"], step=0.5,
                                         label="Guidance scale")
                    size = gr.Slider(256, 768, value=D["size"], step=64, label="Image size")
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    want_clip = gr.Checkbox(value=True, label="Compute CLIP score")
                    gen_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    out = gr.Image(label="Result", type="pil")
                    info = gr.Markdown()
            model.change(on_model_change, model, [steps, guidance])
            rand_btn.click(dataset_utils.get_random_prompt, outputs=prompt)
            gen_btn.click(on_generate,
                          [model, prompt, negative, steps, guidance, seed, size, want_clip],
                          [out, info])

        with gr.Tab("Real vs AI (dataset)"):
            gr.Markdown("Streams real image-caption pairs from the Hugging Face dataset, "
                        "generates an AI image for each caption, and scores them.")
            with gr.Row():
                cmp_model = gr.Dropdown(MODEL_CHOICES, value=core.DEFAULT_MODEL, label="Model")
                cmp_n = gr.Slider(1, 6, value=3, step=1, label="Number of samples")
            cmp_btn = gr.Button("Run comparison", variant="primary")
            with gr.Row():
                real_g = gr.Gallery(label="Real images (dataset)", columns=3, height="auto")
                gen_g = gr.Gallery(label="AI-generated", columns=3, height="auto")
            cmp_md = gr.Markdown()
            cmp_btn.click(on_compare, [cmp_model, cmp_n], [real_g, gen_g, cmp_md])

        with gr.Tab("About"):
            gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
