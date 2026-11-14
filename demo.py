"""Run the local ViC-MAE Tiny-ImageNet classifier at http://127.0.0.1:5000."""

import argparse
import base64
import io
import pickle
import sys
from pathlib import Path
from threading import Lock, Timer

import torch
from flask import Flask, Request, jsonify, render_template, request, send_file
from PIL import Image, JpegImagePlugin
from torchvision.datasets.folder import find_classes
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from extension.demo_integration import interpret, load_settings, render_attribution, resolve_settings_path
from extension.pipeline import Pipeline

from classifier_checkpoints import (
    DATA_DIR,
    INPUT_SIZE,
    NUM_CLASSES,
    PROJECT_DIR,
    UPSTREAM_DIR,
    checkpoint_globals,
    measured_accuracy,
    newest_classifier_checkpoint,
)

sys.path.insert(0, str(UPSTREAM_DIR))
import models_vit
from util.datasets import build_transform

LOW_CONFIDENCE_THRESHOLD = 0.30
HIGH_CONFIDENCE_THRESHOLD = 0.70
BYTES_PER_MIB = 1024 * 1024
MAX_UPLOAD_BYTES = 8 * BYTES_PER_MIB
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
MAX_IMAGE_PIXELS = 20_000_000
UI_DIR = PROJECT_DIR / "demo_ui"
VALIDATION_EXAMPLES = {
    "n01443537": "val_1230.JPEG",
    "n02099601": "val_1321.JPEG",
    "n04146614": "val_1097.JPEG",
}


def build_class_labels(data_path: Path):
    class_ids, _ = find_classes(data_path / "train")
    expected_ids = (data_path / "wnids.txt").read_text(encoding="utf-8").split()
    if len(class_ids) != NUM_CLASSES or class_ids != sorted(expected_ids):
        raise ValueError(f"{data_path / 'train'} must contain the {NUM_CLASSES} Tiny-ImageNet class folders.")

    words_file = data_path / "words.txt"
    wnid_to_name = {}
    with words_file.open(encoding="utf-8") as words:
        for line in words:
            wnid, separator, names = line.rstrip("\r\n").partition("\t")
            if separator and names.strip():
                wnid_to_name[wnid] = names.split(",", 1)[0].strip()
    missing = [wnid for wnid in class_ids if not wnid_to_name.get(wnid)]
    if missing:
        raise ValueError(f"{words_file} has no readable names for: {', '.join(missing)}")
    return [(wnid, wnid_to_name[wnid]) for wnid in class_ids]


def load_model(checkpoint_path: Path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")
    try:
        with checkpoint_globals(checkpoint_path, allow_saved_args=True):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not load classifier checkpoint {checkpoint_path}: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise ValueError(f"{checkpoint_path} must contain a classifier state dictionary.")
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict) or "head.weight" not in state_dict:
        raise ValueError(f"{checkpoint_path} is missing the trained Tiny-ImageNet classifier head.")
    model = models_vit.vit_base_patch16(num_classes=NUM_CLASSES, global_pool=False)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"{checkpoint_path} is incompatible with the Tiny-ImageNet ViT-B/16: {exc}") from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Loaded classifier from {checkpoint_path} on {device}")
    return model, device


class MemoryUploadRequest(Request):
    """Keep multipart files in memory instead of Werkzeug's temporary-file spool."""

    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


def confidence_level(probability):
    if probability < LOW_CONFIDENCE_THRESHOLD:
        return "low"
    if probability >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    return "medium"


def describe_bytes(size):
    return f"{size / BYTES_PER_MIB:.2f} MiB ({size:,} bytes)"


def pixel_limit_message(file_bytes, width, height):
    pixels = width * height
    return (
        f"Your image is {pixels / 1_000_000:.2f} MP "
        f"({width:,} × {height:,}, {pixels:,} pixels) and {describe_bytes(file_bytes)}. "
        f"The pixel limit is {MAX_IMAGE_PIXELS / 1_000_000:g} MP. "
        f"The file limit is {describe_bytes(MAX_UPLOAD_BYTES)}."
    )


def read_image():
    uploaded = request.files.get("image")
    if not uploaded or not uploaded.filename:
        raise BadRequest("Choose an image to classify.")
    file_bytes = uploaded.stream.seek(0, io.SEEK_END)
    if file_bytes > MAX_UPLOAD_BYTES:
        raise RequestEntityTooLarge(
            f"Your image file is {describe_bytes(file_bytes)}. "
            f"The file limit is {describe_bytes(MAX_UPLOAD_BYTES)}."
        )
    uploaded.stream.seek(0)
    try:
        with Image.open(uploaded.stream) as source:
            width, height = source.size
            if width * height > MAX_IMAGE_PIXELS:
                raise RequestEntityTooLarge(pixel_limit_message(file_bytes, width, height))
            return source.convert("RGB")
    except Image.DecompressionBombError as exc:
        # Pillow can reject the JPEG header before Image.open returns its dimensions.
        # Read only the header for the error message. This image is always rejected.
        uploaded.stream.seek(0)
        try:
            with JpegImagePlugin.JpegImageFile(uploaded.stream) as header:
                message = pixel_limit_message(file_bytes, *header.size)
        except (OSError, ValueError, SyntaxError):
            message = (
                f"Your image file is {describe_bytes(file_bytes)}. "
                f"Choose an image up to {MAX_IMAGE_PIXELS / 1_000_000:g} MP."
            )
        raise RequestEntityTooLarge(f"{message} The image decoder's safety limit was exceeded.") from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise BadRequest("Could not read that image. Try a JPG, PNG, or WebP file.") from exc


def create_app(checkpoint_path=None, data_path=DATA_DIR, reliability_settings=None, *, classifier=None):
    checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path is not None else newest_classifier_checkpoint()
    data_path = Path(data_path).resolve()
    model, device = load_model(checkpoint_path) if classifier is None else classifier
    settings = load_settings(checkpoint_path, reliability_settings)
    pipeline = None
    if settings and settings.get('schema_version') == 2:
        pipeline = Pipeline.load(model, device, checkpoint_path, resolve_settings_path(reliability_settings))
    class_labels = build_class_labels(data_path)
    preprocess = build_transform(is_train=False, args=argparse.Namespace(input_size=INPUT_SIZE))
    accuracy = measured_accuracy(checkpoint_path)
    labels = dict(class_labels)
    examples = {
        wnid: path for wnid, filename in VALIDATION_EXAMPLES.items()
        if (path := data_path / "val" / wnid / filename).is_file() and wnid in labels
    }

    app = Flask(__name__, template_folder=str(UI_DIR / "templates"), static_folder=str(UI_DIR / "static"))
    app.request_class = MemoryUploadRequest
    app.config.update(MAX_CONTENT_LENGTH=MAX_REQUEST_BYTES, MAX_FORM_PARTS=4)
    inference_lock = Lock()

    def respond(results=(), image_data=None, error=None, status=200, *, unsure=False, rejection_score=None,
                probabilities=None, outside_classes=False, outside_classes_score=None):
        low_confidence = bool(results and results[0]["confidence"] == "low")
        if request.accept_mimetypes.best_match(("text/html", "application/json")) == "application/json":
            response = jsonify(results=results, low_confidence=low_confidence, error=error,
                               calibrated=settings is not None, unsure=unsure, rejection_score=rejection_score,
                               probabilities=probabilities, outside_classes=outside_classes,
                               outside_classes_score=outside_classes_score)
        else:
            response = app.make_response(render_template(
                "index.html", checkpoint_name=checkpoint_path.name, accuracy=accuracy,
                results=results, image_data=image_data, error=error, low_confidence=low_confidence,
                calibrated=settings is not None, unsure=unsure, outside_classes=outside_classes,
                classes=sorted(class_labels, key=lambda item: item[1].casefold()),
                examples=[(wnid, labels[wnid]) for wnid in examples],
                upload_limit=MAX_UPLOAD_BYTES,
                threshold=round(LOW_CONFIDENCE_THRESHOLD * 100), num_classes=NUM_CLASSES,
            ))
        response.status_code = status
        response.cache_control.no_store = True
        return response

    @app.errorhandler(BadRequest)
    def invalid_upload(error):
        return respond(error=error.description, status=400)

    @app.errorhandler(RequestEntityTooLarge)
    def large_upload(error):
        if error.description != RequestEntityTooLarge.description:
            message = error.description
        elif request.content_length is not None and request.content_length > MAX_REQUEST_BYTES:
            message = (
                f"The upload request is {describe_bytes(request.content_length)} including form overhead. "
                f"The request limit is {describe_bytes(MAX_REQUEST_BYTES)}. "
                f"The image file limit is {describe_bytes(MAX_UPLOAD_BYTES)}."
            )
        else:
            message = (
                "The upload exceeded a server form limit. "
                f"Choose a single image file up to {describe_bytes(MAX_UPLOAD_BYTES)} and try again."
            )
        return respond(error=message, status=413)

    @app.get("/examples/<wnid>")
    def example_image(wnid):
        if wnid not in examples:
            return "", 404
        return send_file(examples[wnid], mimetype="image/jpeg")

    @app.post("/heatmap")
    def heatmap():
        if settings is None:
            return "", 404
        img = read_image()
        comparison = request.accept_mimetypes.best_match(("image/png", "application/json")) == "application/json"
        with inference_lock:
            transform = pipeline.preprocess.center_transform if pipeline else preprocess
            target = int(pipeline.predict(img)[0].argmax()) if pipeline else None
            inputs = transform(img).unsqueeze(0).to(device)
            payload = render_attribution(model, inputs, transform.transforms[-1], target=target, comparison=comparison)
        response = jsonify(payload) if comparison else send_file(payload, mimetype="image/png")
        response.cache_control.no_store = True
        return response

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "GET":
            return respond()

        img = read_image()
        rejection_score = None
        unsure = False
        outside_classes_score = None
        outside_classes = False
        with inference_lock, torch.inference_mode():
            if pipeline:
                prediction, rejection_score, unsure, outside_classes_score, outside_classes = pipeline.predict_with_flags(img)
                probs = torch.from_numpy(prediction)
            else:
                logits = model(preprocess(img).unsqueeze(0).to(device))
                if settings is not None:
                    probs, rejection_score, unsure = interpret(logits, settings)
                else:
                    probs = logits.softmax(dim=-1)[0]
            top5_probs, top5_indices = probs.topk(5)
            probabilities = top5_probs.tolist()
            indices = top5_indices.tolist()

        results = []
        for prob, idx in zip(probabilities, indices):
            wnid, label = class_labels[idx]
            results.append({"wnid": wnid, "label": label, "pct": round(prob * 100, 1),
                            "confidence": confidence_level(prob)})

        image_data = None
        if request.accept_mimetypes.best_match(("text/html", "application/json")) != "application/json":
            # Resize only the fallback preview. Inference uses the original RGB image.
            img.thumbnail((960, 960), Image.Resampling.LANCZOS)
            preview = io.BytesIO()
            img.save(preview, format="JPEG")
            image_data = base64.b64encode(preview.getvalue()).decode("ascii")
        return respond(results=results, image_data=image_data, unsure=unsure, rejection_score=rejection_score,
                       probabilities=probs.tolist() if pipeline else None, outside_classes=outside_classes,
                       outside_classes_score=outside_classes_score)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Tiny-ImageNet classifier demo.")
    parser.add_argument("--checkpoint", type=Path, help="Local classifier file. Defaults to the newest available finetuning checkpoint.")
    parser.add_argument("--data-path", type=Path, default=DATA_DIR)
    parser.add_argument("--reliability-settings", type=Path, help="Measured settings for this exact checkpoint.")
    parser.add_argument("--port", type=int, default=5000, help="Local port for the demo (default: 5000).")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    try:
        app = create_app(args.checkpoint, args.data_path, args.reliability_settings)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Unable to start classifier: {exc}\n")

    url = f"http://127.0.0.1:{args.port}/"
    print(f"Open {url} to classify an image.")
    import webbrowser

    browser = Timer(0.5, webbrowser.open, args=(url,))
    browser.daemon = True
    browser.start()
    app.run(host="127.0.0.1", port=args.port, debug=False)
