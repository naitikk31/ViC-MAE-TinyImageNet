"""One deterministic inference path for Flask and the reliability study."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from .reliability import probabilities


VIEW_IDS = {'original': (0,), 'flip': (1,), 'original_flip': (0, 1), 'original_flip_corners': tuple(range(6))}
METHODS = ('msp_raw', 'msp_calibrated', 'energy_raw', 'energy_calibrated', 'mahalanobis', 'knn50', 'knn10')
BATCH_SIZE = 16
COVARIANCE_RIDGE = 1e-4


class Preprocessing:
    def __init__(self):
        from util.datasets import build_transform
        self.center_transform = build_transform(False, argparse.Namespace(input_size=224))
        self.resize, self.center, self.to_tensor, self.normalize = self.center_transform.transforms

    def views(self, image, recipe):
        resized = self.resize(image.convert('RGB'))
        center = self.center(resized)
        size = center.height
        right, bottom = resized.width - size, resized.height - size
        crops = [center, TF.hflip(center)]
        crops.extend(TF.crop(resized, y, x, size, size) for y, x in ((0, 0), (0, right), (bottom, 0), (bottom, right)))
        return torch.stack([self.normalize(self.to_tensor(crops[i])) for i in VIEW_IDS[recipe]])


def forward_views(model, inputs):
    """Return head logits and normalized CLS features for batched image views."""
    count, views = inputs.shape[:2]
    logits, features = [], []
    with torch.inference_mode():
        for batch in inputs.flatten(0, 1).split(BATCH_SIZE):
            pooled = model.forward_features(batch)
            logits.append(model.head(pooled))
            features.append(pooled)
    return (torch.cat(logits).reshape(count, views, -1).cpu().numpy(),
            torch.cat(features).reshape(count, views, -1).cpu().numpy())


def averaged_log_probabilities(logits):
    count, views, classes = logits.shape
    probs = probabilities(logits.reshape(-1, classes)).reshape(count, views, classes)
    return np.log(probs.mean(axis=1).clip(min=np.finfo(np.float64).tiny))


def outside_classes_score(view_logits):
    """Apply raw energy to the mean unnormalized logits of the TTA views."""
    mean_logits = np.asarray(view_logits, dtype=np.float64).mean(axis=1)
    return -np.logaddexp.reduce(mean_logits, axis=1)


class FeatureDetectors:
    def __init__(self, means, whitening, reference, device):
        self.means = torch.as_tensor(means, dtype=torch.float32, device=device)
        self.whitening = torch.as_tensor(whitening, dtype=torch.float32, device=device)
        self.means_white = self.means @ self.whitening
        self.reference = F.normalize(torch.as_tensor(reference, dtype=torch.float32, device=device), dim=1)

    @classmethod
    def fit(cls, features, labels, classes, output, device):
        values = features.astype(np.float64)
        means = np.stack([values[labels == target].mean(axis=0) for target in range(classes)])
        centered = values - means[labels]
        covariance = centered.T @ centered / (len(values) - classes)
        ridge = COVARIANCE_RIDGE * np.trace(covariance) / covariance.shape[0]
        factor = np.linalg.cholesky(covariance + ridge * np.eye(covariance.shape[0]))
        whitening = np.linalg.solve(factor, np.eye(len(factor))).T
        np.savez(output, means=means, whitening=whitening, reference=features, ridge=ridge)
        return cls(means, whitening, features, device), float(ridge)

    @classmethod
    def load(cls, path, device):
        with np.load(path, allow_pickle=False) as data:
            return cls(data['means'], data['whitening'], data['reference'], device)

    def scores(self, features, methods):
        feature_methods = tuple(name for name in methods if name in ('mahalanobis', 'knn50', 'knn10'))
        result = {name: [] for name in feature_methods}
        neighbors = [k for k in (50, 10) if f'knn{k}' in methods]
        with torch.inference_mode():
            for offset in range(0, len(features), 512):
                values = torch.as_tensor(features[offset:offset+512], dtype=torch.float32, device=self.means.device)
                if 'mahalanobis' in methods:
                    distances = torch.cdist(values @ self.whitening, self.means_white)
                    result['mahalanobis'].append(-distances.min(dim=1).values.cpu().numpy())
                if neighbors:
                    similarity = F.normalize(values, dim=1) @ self.reference.T
                    closest = similarity.topk(max(neighbors), dim=1, sorted=True).values
                    for k in neighbors:
                        result[f'knn{k}'].append((1 - closest[:, :k].mean(dim=1)).clamp(0, 2).cpu().numpy())
        return {name: np.concatenate(parts) for name, parts in result.items()}


def unknown_score(method, native_score):
    return -native_score if method == 'mahalanobis' else native_score


def detector_scores(logits, center_features, temperature, detectors, methods=METHODS):
    results = {}
    for suffix, scale in (('raw', 1.0), ('calibrated', temperature)):
        msp, energy = f'msp_{suffix}', f'energy_{suffix}'
        if msp in methods:
            results[msp] = 1 - probabilities(averaged_log_probabilities(logits), scale).max(axis=1)
        if energy in methods:
            values = logits.astype(np.float64) / scale
            peaks = values.max(axis=-1)
            per_view = -scale * (peaks + np.log(np.exp(values - peaks[..., None]).sum(axis=-1)))
            results[energy] = per_view.mean(axis=1)
    if any(name in methods for name in ('mahalanobis', 'knn50', 'knn10')):
        results.update(detectors.scores(center_features, methods))
    return results


class Pipeline:
    def __init__(self, model, device, settings, detectors=None):
        self.model, self.device, self.settings, self.detectors = model, device, settings, detectors
        self.preprocess = Preprocessing()

    @classmethod
    def load(cls, model, device, checkpoint, settings_path):
        path = Path(settings_path)
        settings = json.loads(path.read_text(encoding='utf-8'))
        with Path(checkpoint).open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != settings['checkpoint_sha256']:
            raise ValueError('Reliability settings belong to a different checkpoint.')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        detectors = None
        if settings['score'] in ('mahalanobis', 'knn50', 'knn10'):
            path = path.parent / settings['feature_detectors']
            with path.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != settings['feature_detectors_sha256']:
                    raise ValueError('The saved feature detector bank does not match the settings.')
            detectors = FeatureDetectors.load(path, device)
        return cls(model, device, settings, detectors)

    def from_outputs(self, logits, center_features, methods=None):
        method = self.settings['score']
        temperature = self.settings['temperature']
        scores = detector_scores(logits, center_features, temperature, self.detectors, methods or (method,))
        if method in scores:
            unsure = unknown_score(method, scores[method]) > self.settings['threshold']
        else:
            unsure = None
        return probabilities(averaged_log_probabilities(logits), temperature), scores, unsure

    def predict(self, image):
        return self.predict_with_flags(image)[:3]

    def predict_with_flags(self, image):
        recipe = self.settings['recipe']
        views = self.preprocess.views(image, recipe)
        logits, features = forward_views(self.model, views.unsqueeze(0).to(self.device))
        center = features[:, 0]
        if recipe == 'flip' and self.settings['score'] in ('mahalanobis', 'knn50', 'knn10'):
            _, features = forward_views(self.model, self.preprocess.views(image, 'original').unsqueeze(0).to(self.device))
            center = features[:, 0]
        probs, scores, unsure = self.from_outputs(logits, center)
        cutoff = self.settings.get('outside_classes_threshold')
        energy = float(outside_classes_score(logits)[0]) if cutoff is not None else None
        return probs[0], float(scores[self.settings['score']][0]), bool(unsure[0]), energy, energy is not None and energy > cutoff
