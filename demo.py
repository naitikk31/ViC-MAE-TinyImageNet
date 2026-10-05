"""
ViC-MAE Demo — Upload an image and classify it with the finetuned ViT-B/16.

Usage:
    python demo.py

Opens a web UI at http://127.0.0.1:5000
"""
import os, sys, io, base64, json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from flask import Flask, request, render_template_string

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR / "ViC-MAE"
CKPT_PATH = PROJECT_DIR / "output" / "20260918-044542-vit_base_patch16-224" / "checkpoint-20.pth"
WORDS_FILE = PROJECT_DIR / "tiny-imagenet-200" / "words.txt"
DATA_DIR = PROJECT_DIR / "tiny-imagenet-200"

sys.path.insert(0, str(REPO_DIR))
import models_vit

# ── Build class label mapping ──────────────────────────────────────────
def build_class_labels():
    train_dir = DATA_DIR / "train"
    class_ids = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    wnid_to_name = {}
    with open(WORDS_FILE) as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                wnid_to_name[parts[0]] = parts[1].split(',')[0].strip()
    return {idx: wnid_to_name.get(wnid, wnid) for idx, wnid in enumerate(class_ids)}

# ── Load model ─────────────────────────────────────────────────────────
def load_model():
    print("Loading ViT-B/16 model...")
    model = models_vit.vit_base_patch16(num_classes=200, global_pool=False)
    checkpoint = torch.load(str(CKPT_PATH), map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model loaded on {device}")
    return model, device

# ── Image preprocessing ────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── HTML Template ──────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ViC-MAE · Darkroom</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        /* ── Reset ─────────────────────────────────────── */
        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

        /* ── Tokens ────────────────────────────────────── */
        :root {
            --bg:        #17140F;
            --surface:   #211D18;
            --accent:    #C1443B;
            --accent-hover: #A33830;
            --amber:     #E8A33D;
            --border:    #3A332B;
            --text:      #EDE6DA;
            --muted:     #A89F94;
            --sans:      'Space Grotesk', system-ui, sans-serif;
            --mono:      'JetBrains Mono', 'Consolas', monospace;
        }

        /* ── Page ──────────────────────────────────────── */
        body {
            font-family: var(--sans);
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 60px 20px;
        }
        .container { max-width: 640px; width: 100%; }

        /* ── Header ────────────────────────────────────── */
        .header { text-align: center; margin-bottom: 36px; }
        h1 {
            font-size: 1.75em;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text);
            margin-bottom: 6px;
        }
        h1 .icon { font-size: 0.85em; margin-right: 4px; }
        .subtitle {
            color: var(--muted);
            font-size: 0.85em;
            margin-bottom: 16px;
        }

        /* ── Stat badges (camera-readout style) ────────── */
        .badges { display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; }
        .badge {
            font-family: var(--mono);
            font-size: 0.75em;
            font-weight: 600;
            color: var(--muted);
            border: 1px solid var(--border);
            padding: 4px 12px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .badge .val { color: var(--text); }

        /* ── Dropzone with crop marks ──────────────────── */
        .dropzone-wrapper { position: relative; margin: 32px 0 24px; }
        .dropzone {
            background: var(--surface);
            min-height: 220px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: background 0.2s;
            position: relative;
        }
        .dropzone:hover { background: #2A251E; }
        .dropzone.has-image { min-height: unset; }
        .dropzone img {
            max-width: 100%;
            max-height: 340px;
            display: block;
        }
        .dropzone-placeholder {
            text-align: center;
            color: var(--muted);
            font-size: 0.9em;
            line-height: 1.6;
        }
        .dropzone-placeholder .arrow { font-size: 1.8em; margin-bottom: 8px; display: block; color: var(--border); }

        /* ── Crop / register marks ─────────────────────── */
        .crop-mark {
            position: absolute;
            width: 20px;
            height: 20px;
            pointer-events: none;
        }
        .crop-mark::before, .crop-mark::after {
            content: '';
            position: absolute;
            background: var(--border);
        }
        /* horizontal stroke */
        .crop-mark::before { width: 20px; height: 1px; }
        /* vertical stroke */
        .crop-mark::after  { width: 1px; height: 20px; }

        .crop-mark.tl { top: -6px; left: -6px; }
        .crop-mark.tl::before { top: 0; left: 0; }
        .crop-mark.tl::after  { top: 0; left: 0; }

        .crop-mark.tr { top: -6px; right: -6px; }
        .crop-mark.tr::before { top: 0; right: 0; }
        .crop-mark.tr::after  { top: 0; right: 0; }

        .crop-mark.bl { bottom: -6px; left: -6px; }
        .crop-mark.bl::before { bottom: 0; left: 0; }
        .crop-mark.bl::after  { bottom: 0; left: 0; }

        .crop-mark.br { bottom: -6px; right: -6px; }
        .crop-mark.br::before { bottom: 0; right: 0; }
        .crop-mark.br::after  { bottom: 0; right: 0; }

        input[type="file"] { display: none; }

        /* ── Action buttons ────────────────────────────── */
        .actions {
            display: flex;
            gap: 12px;
            width: 100%;
        }
        .btn {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            padding: 14px 20px;
            background: var(--accent);
            color: var(--text);
            border: none;
            font-family: var(--sans);
            font-size: 0.95em;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            cursor: pointer;
            text-decoration: none;
            transition: background 0.15s, border-color 0.15s, color 0.15s;
        }
        .btn:hover { background: var(--accent-hover); }
        .btn-clear {
            width: auto;
            min-width: 110px;
            background: transparent;
            border: 1px solid var(--border);
            color: var(--muted);
            font-family: var(--mono);
            font-size: 0.85em;
        }
        .btn-clear:hover {
            background: var(--surface);
            border-color: var(--muted);
            color: var(--text);
        }

        /* ── Dropzone hover overlay ─────────────────────── */
        .dropzone-overlay {
            position: absolute;
            inset: 0;
            background: rgba(23, 20, 15, 0.78);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.2s;
            color: var(--text);
            font-family: var(--mono);
            font-size: 0.8em;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .dropzone:hover .dropzone-overlay { opacity: 1; }

        /* ── Results ───────────────────────────────────── */
        .results {
            margin-top: 32px;
            border-top: 1px solid var(--border);
            padding-top: 24px;
        }
        .results-title {
            font-size: 0.7em;
            font-family: var(--mono);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
            margin-bottom: 16px;
        }

        .result-row {
            display: flex;
            align-items: center;
            padding: 10px 0;
            border-left: 3px solid transparent;
            padding-left: 12px;
            margin-bottom: 4px;
        }
        .result-row.top-1 {
            border-left-color: var(--accent);
            background: rgba(193, 68, 59, 0.06);
        }

        .rank {
            font-family: var(--mono);
            font-weight: 600;
            font-size: 0.8em;
            color: var(--muted);
            width: 24px;
            flex-shrink: 0;
        }
        .result-row.top-1 .rank { color: var(--accent); }

        .result-label {
            font-size: 0.9em;
            font-weight: 600;
            color: var(--text);
            width: 160px;
            flex-shrink: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .bar-track {
            flex: 1;
            height: 22px;
            background: var(--surface);
            margin: 0 14px;
            overflow: hidden;
            position: relative;
        }
        .bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--amber));
            transition: width 0.6s ease-out;
        }
        .result-row:not(.top-1) .bar-fill {
            background: linear-gradient(90deg, #3A332B, #4A4238);
        }

        .result-pct {
            font-family: var(--mono);
            font-weight: 600;
            font-size: 0.85em;
            color: var(--muted);
            width: 55px;
            text-align: right;
            flex-shrink: 0;
        }
        .result-row.top-1 .result-pct { color: var(--text); }

        /* ── Footer ────────────────────────────────────── */
        .footer {
            margin-top: 40px;
            text-align: center;
            font-size: 0.7em;
            font-family: var(--mono);
            color: var(--border);
            letter-spacing: 0.06em;
        }
    </style>
</head>
<body>
<div class="container">

    <!-- Header -->
    <div class="header">
        <h1><span class="icon">◉</span> ViC-MAE Classifier</h1>
        <p class="subtitle">ViT-Base/16 · finetuned on Tiny-ImageNet-200</p>
        <div class="badges">
            <div class="badge"><span class="val">80.77%</span> TOP-1</div>
            <div class="badge"><span class="val">94.39%</span> TOP-5</div>
            <div class="badge"><span class="val">200</span> CLASSES</div>
        </div>
    </div>

    <!-- Upload form -->
    <form method="POST" enctype="multipart/form-data">
        <div class="dropzone-wrapper">
            <div class="crop-mark tl"></div>
            <div class="crop-mark tr"></div>
            <div class="crop-mark bl"></div>
            <div class="crop-mark br"></div>
            <div class="dropzone {{ 'has-image' if image_data else '' }}"
                 id="dropzone"
                 onclick="document.getElementById('file').click()">
                {% if image_data %}
                    <img src="data:image/jpeg;base64,{{ image_data }}" alt="Uploaded image">
                    <div class="dropzone-overlay">
                        <span style="font-size:1.8em; margin-bottom:6px;">⟲</span>
                        Click to change image
                    </div>
                {% else %}
                    <div class="dropzone-placeholder">
                        <span class="arrow">⤓</span>
                        Drop image here or <b>click to browse</b><br>
                        <span style="font-size:0.8em; color: var(--border);">JPG · PNG · WEBP</span>
                    </div>
                {% endif %}
            </div>
        </div>
        <input type="file" name="image" id="file" accept="image/*"
               onchange="previewAndSubmit(this)">
        {% if results %}
        <div class="actions">
            <button type="button" class="btn" onclick="document.getElementById('file').click()">⤓ TEST ANOTHER IMAGE</button>
            <a href="/" class="btn btn-clear">↺ RESET</a>
        </div>
        {% else %}
        <button type="submit" class="btn">▸ CLASSIFY</button>
        {% endif %}
    </form>

    <!-- Results -->
    {% if results %}
    <div class="results">
        <div class="results-title">Top-5 predictions</div>
        {% for item in results %}
        <div class="result-row {{ 'top-1' if loop.index == 1 else '' }}">
            <div class="rank">{{ loop.index }}</div>
            <div class="result-label" title="{{ item.label }}">{{ item.label }}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width: {{ item.pct }}%"></div>
            </div>
            <div class="result-pct">{{ item.pct }}%</div>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <div class="footer">ECCV 2024 · HERNANDEZ, VILLEGAS, ORDONEZ</div>
</div>

<script>
    const dz = document.getElementById('dropzone');
    ['dragenter','dragover'].forEach(e =>
        dz.addEventListener(e, ev => { ev.preventDefault(); dz.style.background = '#2A251E'; }));
    ['dragleave','drop'].forEach(e =>
        dz.addEventListener(e, ev => { ev.preventDefault(); dz.style.background = ''; }));
    dz.addEventListener('drop', ev => {
        if (ev.dataTransfer.files[0]) {
            document.getElementById('file').files = ev.dataTransfer.files;
            previewAndSubmit(document.getElementById('file'));
        }
    });
    function previewAndSubmit(input) {
        if (input.files && input.files[0]) {
            const r = new FileReader();
            r.onload = e => {
                dz.classList.add('has-image');
                dz.innerHTML = '<img src="'+e.target.result+'" alt="Preview">';
                input.form.submit();
            };
            r.readAsDataURL(input.files[0]);
        }
    }
</script>
</body>
</html>
"""

# ── Flask App ──────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    image_data = None

    if request.method == "POST" and "image" in request.files:
        file = request.files["image"]
        if file.filename:
            img_bytes = file.read()
            image_data = base64.b64encode(img_bytes).decode("utf-8")

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(img_tensor)
                probs = F.softmax(logits, dim=-1)[0]

            top5_probs, top5_indices = probs.topk(5)
            results = []
            for prob, idx in zip(top5_probs, top5_indices):
                label = class_labels.get(idx.item(), f"Class {idx.item()}")
                pct = round(float(prob) * 100, 1)
                results.append({"label": label, "pct": pct})

    return render_template_string(HTML, results=results, image_data=image_data)


if __name__ == "__main__":
    if not CKPT_PATH.exists():
        print(f"ERROR: Checkpoint not found at {CKPT_PATH}")
        sys.exit(1)

    class_labels = build_class_labels()
    model, device = load_model()

    print("\n" + "="*50)
    print("  ViC-MAE Demo is running!")
    print("  Open: http://127.0.0.1:5000")
    print("="*50 + "\n")

    import webbrowser
    webbrowser.open("http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
