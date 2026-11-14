"""Bind measured reliability settings to the matching Flask checkpoint."""

import base64
import hashlib
import io
import json
from pathlib import Path

import torch
from PIL import Image

from .reliability import attribution_images, normalized_crop_to_image, patch_attribution, probabilities, score_arrays


ROOT = Path(__file__).resolve().parent
DEFAULT_SETTINGS = ROOT / 'results_v2' / 'demo_settings.json'
PREVIOUS_SETTINGS = ROOT / 'results' / 'demo_settings.json'
RELEASE_SETTINGS = ROOT / 'demo_settings.json'


def resolve_settings_path(explicit=None):
    if explicit is not None:
        return Path(explicit)
    metrics_path = DEFAULT_SETTINGS.with_name('metrics.json')
    if DEFAULT_SETTINGS.is_file() and metrics_path.is_file():
        if json.loads(metrics_path.read_text(encoding='utf-8')).get('status') == 'complete':
            return DEFAULT_SETTINGS
    return PREVIOUS_SETTINGS if PREVIOUS_SETTINGS.is_file() else RELEASE_SETTINGS


def load_settings(checkpoint, settings_path=None):
    path = resolve_settings_path(settings_path)
    if settings_path is None and not path.is_file():
        return None
    settings = json.loads(path.read_text(encoding='utf-8'))
    with checkpoint.open('rb') as source:
        fingerprint = hashlib.file_digest(source, 'sha256').hexdigest()
    if fingerprint != settings['checkpoint_sha256']:
        if settings_path is None:
            return None
        raise ValueError('Reliability settings belong to a different checkpoint. Use settings fitted for the selected checkpoint.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return settings


def interpret(logits, settings):
    values = logits.detach().cpu().numpy()
    temperature = settings['temperature']
    confidence = probabilities(values, temperature)[0]
    score = float(score_arrays(values, temperature)[settings['score']][0])
    return torch.from_numpy(confidence), score, score > settings['threshold']


def render_attribution(model, inputs, normalization, target=None, *, comparison=False):
    with torch.inference_mode():
        targets = model(inputs).argmax(dim=-1) if target is None else torch.tensor([target], device=inputs.device)
    grid = patch_attribution(model, inputs, targets.clone())[0]
    crop = normalized_crop_to_image(inputs[0], normalization)
    if comparison:
        # Give the browser enough pixels for the desktop panels while bounding the response size.
        crop = crop.resize((448, 448), Image.Resampling.BILINEAR)
    original, heatmap, overlay = attribution_images(crop, grid)
    if comparison:
        return {key: 'data:image/png;base64,' + base64.b64encode(png_bytes(image).getvalue()).decode('ascii')
                for key, image in [('original', original), ('heatmap', heatmap), ('overlay', overlay)]}
    return png_bytes(overlay)


def png_bytes(image):
    payload = io.BytesIO()
    image.save(payload, format='PNG')
    payload.seek(0)
    return payload
