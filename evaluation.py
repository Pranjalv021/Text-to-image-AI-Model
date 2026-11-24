"""Quality metrics using CLIP (openai/clip-vit-base-patch32, free, ~600 MB).

clip_score(image, text)      : 100 * max(cosine(image_emb, text_emb), 0)
                               -> how well the picture matches the prompt
image_similarity(img1, img2) : cosine similarity of CLIP image embeddings
                               -> how close a generated image is to a real one
"""
import torch
import torch.nn.functional as F

CLIP_ID = "openai/clip-vit-base-patch32"
_model = None
_proc = None


def _load():
    global _model, _proc
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor
        print(f"[info] Loading CLIP '{CLIP_ID}' for evaluation...")
        _model = CLIPModel.from_pretrained(CLIP_ID).eval()
        _proc = CLIPProcessor.from_pretrained(CLIP_ID)
    return _model, _proc


def _embed(images, texts):
    model, proc = _load()
    inputs = proc(text=texts, images=images, return_tensors="pt",
                  padding=True, truncation=True)
    with torch.no_grad():
        out = model(**inputs)
    return F.normalize(out.image_embeds, dim=-1), F.normalize(out.text_embeds, dim=-1)


def clip_score(image, text):
    img_e, txt_e = _embed([image], [text])
    cos = float((img_e[0] * txt_e[0]).sum())
    return 100.0 * max(cos, 0.0)


def image_similarity(image_a, image_b):
    img_e, _ = _embed([image_a, image_b], ["image"])
    return float((img_e[0] * img_e[1]).sum())
