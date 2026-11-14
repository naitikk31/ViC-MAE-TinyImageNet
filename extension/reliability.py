"""Metrics and inference shared by the study and its Flask UI."""

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF


PATCH_SIZE = 16


def probabilities(logits, temperature=1.0):
    values = np.asarray(logits, dtype=np.float64) / temperature
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def score_arrays(logits, temperature):
    scores = {}
    for suffix, scale in [('raw', 1.0), ('calibrated', temperature)]:
        values = np.asarray(logits, dtype=np.float64) / scale
        maximum = values.max(axis=1)
        scores[f'msp_{suffix}'] = 1 - probabilities(logits, scale).max(axis=1)
        scores[f'energy_{suffix}'] = -scale * (maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1)))
    return scores


def threshold_for(scores, acceptance=0.95):
    return float(np.sort(scores)[int(np.ceil(acceptance * len(scores))) - 1])


def roc_metrics(known, unknown):
    scores = np.concatenate((known, unknown))
    positive = np.arange(len(scores)) >= len(known)
    order = np.argsort(-scores, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
    tp = np.cumsum(positive[order])[ends]
    fp = ends + 1 - tp
    tpr = np.r_[0., tp / len(unknown)]
    fpr = np.r_[0., fp / len(known)]
    precision = tp / (tp + fp)
    metrics = {'auroc': float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)),
               'aupr': float(np.sum(np.diff(tpr) * precision)),
               'fpr_at_95_tpr': float(fpr[np.searchsorted(tpr, 0.95)])}
    return metrics, (fpr, tpr)


def fit_temperature(logits, targets):
    values = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(targets, dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], max_iter=100, line_search_fn='strong_wolfe',
                                  tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(values / log_t.exp(), labels)
        loss.backward()
        return loss

    before = float(F.cross_entropy(values, labels))
    optimizer.step(closure)
    temperature = float(log_t.detach().exp())
    if not np.isfinite(temperature) or temperature <= 0 or float(F.cross_entropy(values / temperature, labels)) > before + 1e-10:
        raise ValueError('Temperature fitting failed to produce a finite NLL improvement on the fit half.')
    return temperature


def calibration_metrics(probs, targets, bins=15):
    rows = np.arange(len(targets))
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == targets
    membership = np.minimum((confidence * bins).astype(int), bins - 1)
    stats = []
    for index in range(bins):
        mask = membership == index
        count = int(mask.sum())
        stats.append({'lower': index / bins, 'upper': (index + 1) / bins, 'count': count,
                      'accuracy': float(correct[mask].mean()) if count else None,
                      'confidence': float(confidence[mask].mean()) if count else None})
    return {'ece': sum(row['count'] * abs(row['accuracy'] - row['confidence']) for row in stats if row['count']) / len(targets),
            'nll': float(-np.log(probs[rows, targets]).mean()),
            'brier': float((np.square(probs).sum(axis=1) - 2 * probs[rows, targets] + 1).mean()),
            'bins': stats}


def patch_attribution(model, inputs, targets):
    """Differentiate the class logit through tokens before the final attention operation."""
    if model.global_pool:
        raise ValueError('This attribution implementation expects the demo\'s CLS-token classifier.')
    captured = []
    hook = model.blocks[-1].norm1.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    try:
        with torch.enable_grad():
            logits = model(inputs.detach().requires_grad_(True))
            activation, = captured
            gradients, = torch.autograd.grad(logits.gather(1, targets[:, None]).sum(), activation)
            weights = gradients[:, 1:].mean(dim=1, keepdim=True)
            relevance = (activation[:, 1:] * weights).sum(dim=-1).relu().detach()
    finally:
        hook.remove()
    grid = inputs.shape[-1] // PATCH_SIZE
    if relevance.shape[1] != grid * grid or not torch.isfinite(relevance).all():
        raise ValueError('Invalid attribution grid.')
    return relevance.reshape(-1, grid, grid)


def normalize_attribution(grid):
    """Scale the patch scores for one image, including flat or empty maps."""
    scores = grid.detach().float().cpu()
    minimum, maximum = scores.aminmax()
    return (scores - minimum) / (maximum - minimum) if maximum > minimum else torch.zeros_like(scores)


def attribution_images(image, grid):
    original = image.convert('RGB')
    normalized = normalize_attribution(grid)
    resized = F.interpolate(normalized[None, None], size=(image.height, image.width),
                            mode='bilinear', align_corners=False)[0, 0].numpy()
    # Jet has blue, cyan, yellow and red bands at distinct score levels.
    channels = [np.clip(1.5 - np.abs(4 * resized - offset), 0, 1) for offset in (3, 2, 1)]
    heatmap = Image.fromarray(np.rint(255 * np.stack(channels, axis=-1)).astype(np.uint8))
    overlay = Image.blend(original, heatmap, 0.5) if normalized.any() else original.copy()
    return original, heatmap, overlay


def overlay(image, grid):
    return attribution_images(image, grid)[2]


def normalized_crop_to_image(tensor, normalization):
    return TF.to_pil_image(tensor.cpu().mul(torch.tensor(normalization.std)[:, None, None])
                           .add(torch.tensor(normalization.mean)[:, None, None]).clamp(0, 1))
