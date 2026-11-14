"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("upload-form");
const fileInput = byId("image");
const dropZone = byId("drop-zone");
const preview = byId("preview");
const previewWrap = byId("preview-wrap");
const resultsRegion = byId("results-region");
const emptyResults = byId("empty-results");
const loadingResults = byId("loading-results");
const predictions = byId("prediction-list");
const predictionTemplate = byId("prediction-template");
const lowWarning = byId("low-warning");
const outsideClassesWarning = byId("outside-classes-warning");
const heatmapControls = byId("heatmap-controls");
const heatmapToggle = byId("heatmap-toggle");
const heatmapStatus = byId("heatmap-status");
const heatmapComparison = byId("heatmap-comparison");
const heatmapPanels = {original: byId("heatmap-original"), heatmap: byId("heatmap-only"), overlay: byId("heatmap-overlay")};
let heatmapFile = null;
let heatmapRequest = null;
let heatmapReady = false;
const status = byId("status");
const resetButton = byId("reset");
const error = byId("upload-error");
const errorText = byId("error-text");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const maxBytes = Number(form.dataset.maxBytes);
const bytesPerMiB = 1024 * 1024;
let activeRequest = null;
let objectUrl = null;
let animationFrame = 0;

const themeToggle = byId("theme-toggle");
const themeAnnouncement = byId("theme-announcement");
const themeKey = "vicmae-demo-theme";
const WAVE_DURATION = 1360;
const WAVE_TRAVEL_TIME = 1180;
const THREAD_COUNT = 64;
const TAU = Math.PI * 2;
let themeChanging = false;

function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    themeToggle.setAttribute("aria-pressed", String(theme === "light"));
}

function easeWave(progress) {
    return (1 - Math.cos(Math.PI * Math.min(1, Math.max(0, progress)))) / 2;
}

function makeWavePainter(canvas, context, root, theme, revealPage) {
    const direction = theme === "dark" ? 1 : -1;
    const accent = direction === 1 ? "180, 163, 255" : "89, 66, 183";
    const highlight = direction === 1 ? "237, 224, 255" : "136, 100, 227";
    const threads = Array.from({length: THREAD_COUNT}, (_, i) => ({
        position: (i + .18 + ((i * 37) % 19) / 30) / THREAD_COUNT,
        phase: ((i * 41) % 67) / 67,
        frequency: .55 + ((i * 23) % 37) / 31,
        amplitude: .45 + ((i * 13) % 31) / 33,
        speed: .42 + ((i * 17) % 29) / 32,
        length: .13 + ((i * 29) % 37) / 145,
        thickness: .55 + ((i * 11) % 17) / 14,
        opacity: .26 + ((i * 19) % 23) / 48,
    }));
    let width = 0;
    let height = 0;
    let ratio = 0;
    let samples = [];

    return function paintWave(elapsed) {
        const nextWidth = Math.max(1, root.clientWidth);
        const nextHeight = Math.max(1, root.clientHeight);
        // Keep native pixel density where practical while bounding large canvas allocations.
        const nextRatio = Math.min(window.devicePixelRatio || 1, Math.sqrt(6_000_000 / (nextWidth * nextHeight)));
        if (nextWidth !== width || nextHeight !== height || nextRatio !== ratio) {
            width = nextWidth;
            height = nextHeight;
            ratio = nextRatio;
            canvas.width = Math.ceil(width * ratio);
            canvas.height = Math.ceil(height * ratio);
            context.setTransform(canvas.width / width, 0, 0, canvas.height / height, 0, 0);
            const segments = Math.max(72, Math.min(160, Math.ceil(width / 13)));
            samples = Array.from({length: segments + 1}, (_, i) => {
                const x = -1 + (width + 2) * i / segments;
                return {x, clipX: x.toFixed(2) + "px"};
            });
        }
        context.clearRect(0, 0, width, height);

        const progress = elapsed / WAVE_DURATION;
        const travel = easeWave(elapsed / WAVE_TRAVEL_TIME);
        const amplitude = Math.min(height * .045, 32);
        const margin = amplitude + 3;
        const from = direction === 1 ? -margin : height + margin;
        const base = from + direction * (height + 2 * margin) * travel;
        const outerY = direction === 1 ? -2 * margin : height + 2 * margin;
        function edgeAt(x) {
            const fraction = x / width;
            return base + amplitude * (
                .76 * Math.sin(TAU * (2.6 * fraction + .54 * progress))
                + .24 * Math.sin(TAU * (3.8 * fraction - .31 * progress + .18))
            );
        }

        const points = samples.map(({x, clipX}) => ({x, clipX, y: edgeAt(x)}));
        if (revealPage) {
            const edge = points.map(({clipX, y}) => clipX + " " + y.toFixed(2) + "px").join(", ");
            root.style.setProperty("--theme-wave-clip",
                "polygon(-1px " + outerY + "px, " + edge + ", " + (width + 1) + "px " + outerY + "px)");
        }

        const fade = easeWave(elapsed / 160) * (1 - easeWave((elapsed - 980) / (WAVE_DURATION - 980)));
        if (!fade) return;
        context.beginPath();
        context.moveTo(-1, outerY);
        for (const {x, y} of points) context.lineTo(x, y);
        context.lineTo(width + 1, outerY);
        context.closePath();
        if (!revealPage) {
            const tint = direction === 1 ? "70, 47, 124" : "201, 184, 253";
            const wash = easeWave(elapsed / 210) * (1 - easeWave((elapsed - WAVE_TRAVEL_TIME) / (WAVE_DURATION - WAVE_TRAVEL_TIME)));
            context.fillStyle = "rgba(" + tint + ", " + (.2 * wash) + ")";
            context.fill();
        }

        context.save();
        context.clip();
        context.globalAlpha = fade;
        context.lineCap = "round";
        const swayScale = Math.max(4, Math.min(16, width * .013));
        for (const strand of threads) {
            const phase = TAU * (strand.phase + strand.speed * progress);
            const tipX = strand.position * width + swayScale * .55 * Math.sin(phase);
            const tipY = edgeAt(tipX) - direction * (9 + 7 * Math.sin(phase * 1.4));
            const length = height * strand.length * (.87 + .13 * Math.cos(phase));
            const tailY = tipY - direction * length;
            if (Math.max(tipY, tailY) < 0 || Math.min(tipY, tailY) > height) continue;

            const gradient = context.createLinearGradient(0, tipY, 0, tailY);
            gradient.addColorStop(0, "rgba(" + highlight + ", " + strand.opacity + ")");
            gradient.addColorStop(.27, "rgba(" + accent + ", " + strand.opacity + ")");
            gradient.addColorStop(1, "rgba(" + accent + ", 0)");
            context.lineWidth = strand.thickness;
            context.strokeStyle = gradient;
            context.beginPath();
            context.moveTo(tipX, tipY);
            for (let step = 1; step <= 16; step++) {
                const t = step / 16;
                const wave = phase + TAU * strand.frequency * t;
                const x = tipX + swayScale * strand.amplitude * (Math.sin(wave) - Math.sin(phase)) * Math.sqrt(t);
                const y = tipY - direction * length * t + 8 * Math.sin(wave) * Math.sin(Math.PI * t);
                context.lineTo(x, y);
            }
            context.stroke();
        }
        context.restore();

        context.beginPath();
        points.forEach(({x, y}, index) => {
            if (index === 0) context.moveTo(x, y);
            else context.lineTo(x, y);
        });
        context.lineWidth = 1.15;
        context.strokeStyle = "rgba(" + accent + ", " + (.58 * fade) + ")";
        context.stroke();
    };
}

function animateThemeWave(paint, signal) {
    return new Promise((resolve, reject) => {
        const started = performance.now();
        let frameId;
        const finish = (error) => {
            cancelAnimationFrame(frameId);
            signal.removeEventListener("abort", stop);
            if (error) reject(error);
            else resolve();
        };
        const stop = () => finish();
        function drawThemeFrame(now) {
            const elapsed = Math.min(WAVE_DURATION, Math.max(0, now - started));
            try {
                paint(elapsed);
            } catch (error) {
                finish(error);
                return;
            }
            if (elapsed < WAVE_DURATION) frameId = requestAnimationFrame(drawThemeFrame);
            else finish();
        }
        if (signal.aborted) resolve();
        else {
            signal.addEventListener("abort", stop, {once: true});
            frameId = requestAnimationFrame(drawThemeFrame);
        }
    });
}

async function changeTheme(theme) {
    const root = document.documentElement;
    const commit = () => {
        applyTheme(theme);
        themeAnnouncement.textContent = theme === "light" ? "Light theme enabled." : "Dark theme enabled.";
    };
    try {
        localStorage.setItem(themeKey, theme);
    } catch {
        // The theme still works when browser storage is unavailable.
    }

    if (reducedMotion.matches || typeof window.requestAnimationFrame !== "function") {
        commit();
        return;
    }
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    if (!context) {
        commit();
        return;
    }
    canvas.className = "theme-wave-canvas";
    canvas.setAttribute("aria-hidden", "true");
    const revealPage = typeof document.startViewTransition === "function"
        && window.CSS?.supports?.("clip-path", "polygon(0 0, 100% 0, 0 100%)");
    const paint = makeWavePainter(canvas, context, root, theme, revealPage);
    const controller = new AbortController();
    const stopForReducedMotion = (event) => { if (event.matches) controller.abort(); };
    reducedMotion.addEventListener?.("change", stopForReducedMotion);
    themeChanging = true;
    themeToggle.setAttribute("aria-disabled", "true");
    root.dataset.themeTransition = revealPage ? "view" : "fallback";

    let transition;
    try {
        paint(0);
        if (revealPage) {
            transition = document.startViewTransition(() => {
                if (controller.signal.aborted) return;
                commit();
                document.body.append(canvas);
            });
            // The browser may release the viewport snapshots early after a resize.
            const stop = () => controller.abort();
            transition.finished.then(stop, stop);
            transition.updateCallbackDone.catch(stop);
            await transition.ready;
            await animateThemeWave(paint, controller.signal);
        } else {
            document.body.append(canvas);
            await animateThemeWave((elapsed) => {
                paint(elapsed);
                if (elapsed >= WAVE_TRAVEL_TIME && root.dataset.theme !== theme) commit();
            }, controller.signal);
        }
    } catch {
        // A skipped snapshot or unavailable graphics context can finish directly in the chosen theme.
    } finally {
        controller.abort();
        transition?.skipTransition();
        canvas.remove();
        delete root.dataset.themeTransition;
        root.style.removeProperty("--theme-wave-clip");
        reducedMotion.removeEventListener?.("change", stopForReducedMotion);
        themeToggle.removeAttribute("aria-disabled");
        themeChanging = false;
        if (root.dataset.theme !== theme) commit();
    }
}

try {
    if (localStorage.getItem(themeKey) === "light") applyTheme("light");
} catch {
    // The toggle still works for this page if browser storage is unavailable.
}
themeToggle.hidden = false;
themeToggle.addEventListener("click", () => {
    if (themeChanging) return;
    const theme = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    void changeTheme(theme);
});

function clearPreview() {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
    preview.removeAttribute("src");
    previewWrap.hidden = true;
    dropZone.classList.remove("has-preview");
}

function showPreview(source, isObjectUrl) {
    clearPreview();
    if (isObjectUrl) objectUrl = source;
    preview.src = source;
    preview.alt = "Image selected for classification";
    previewWrap.hidden = false;
    dropZone.classList.add("has-preview");
}

function setLoading(loading) {
    status.classList.toggle("is-busy", loading);
    status.textContent = loading ? "Classifying…" : "";
    loadingResults.hidden = !loading;
    dropZone.classList.toggle("is-scanning", loading);
    resultsRegion.setAttribute("aria-busy", String(loading));
}

function clearResults() {
    window.cancelAnimationFrame(animationFrame);
    predictions.replaceChildren();
    predictions.hidden = true;
    lowWarning.hidden = true;
    outsideClassesWarning.hidden = true;
    emptyResults.hidden = false;
    if (heatmapControls) {
        heatmapRequest?.abort();
        heatmapRequest = null;
        heatmapReady = false;
        heatmapFile = null;
        heatmapComparison.hidden = true;
        for (const panel of Object.values(heatmapPanels)) panel.removeAttribute("src");
        heatmapControls.hidden = true;
        heatmapToggle.disabled = false;
        heatmapToggle.textContent = "Show heatmap";
        heatmapToggle.setAttribute("aria-pressed", "false");
        heatmapStatus.textContent = "";
    }
}

function clearError() {
    error.hidden = true;
    errorText.textContent = "";
}

function showError(message) {
    setLoading(false);
    clearPreview();
    clearResults();
    errorText.textContent = message;
    error.hidden = false;
    resetButton.hidden = false;
}

function showResults(data) {
    const animate = !reducedMotion.matches;
    const fragment = document.createDocumentFragment();
    const rows = data.results.map((result, index) => {
        const row = predictionTemplate.content.firstElementChild.cloneNode(true);
        row.classList.toggle("top-prediction", index === 0);
        row.classList.toggle("is-new", animate);
        row.dataset.wnid = result.wnid;
        row.dataset.confidence = result.confidence;
        row.querySelector(".rank").textContent = index + 1;
        row.querySelector(".prediction-name").textContent = result.label;
        row.querySelector(".prediction-meta span").textContent = result.confidence + " confidence";
        const score = row.querySelector(".score-count");
        const fill = row.querySelector(".score-fill");
        const finalText = result.pct.toFixed(1) + "%";
        row.querySelector(".score-value").textContent = finalText;
        score.textContent = animate ? "0.0%" : finalText;
        if (!animate) fill.style.setProperty("--score", finalText);
        fragment.append(row);
        return {score, fill, value: result.pct, finalText};
    });
    predictions.replaceChildren(fragment);
    predictions.hidden = false;
    emptyResults.hidden = true;
    lowWarning.hidden = !(data.calibrated ? data.unsure : data.low_confidence);
    outsideClassesWarning.hidden = !data.outside_classes;
    status.textContent = "Complete";

    if (animate) {
        animationFrame = window.requestAnimationFrame((start) => {
            function count(timestamp) {
                const fraction = reducedMotion.matches ? 1 : Math.min(1, (timestamp - start) / 570);
                const ease = 1 - (1 - fraction) ** 3;
                for (const {score, fill, value, finalText} of rows) {
                    fill.style.setProperty("--score", finalText);
                    score.textContent = (value * ease).toFixed(1) + "%";
                }
                if (fraction < 1) animationFrame = window.requestAnimationFrame(count);
            }
            animationFrame = window.requestAnimationFrame(count);
        });
    }
}

async function classify(file, exampleUrl) {
    activeRequest?.abort();
    const controller = new AbortController();
    activeRequest = controller;
    clearError();
    clearResults();
    resetButton.hidden = false;
    if (file && file.size > maxBytes) {
        activeRequest = null;
        const size = (file.size / bytesPerMiB).toFixed(2);
        const limit = maxBytes / bytesPerMiB;
        showError(
            "Your image file is " + size + " MiB (" + file.size.toLocaleString() + " bytes). "
            + "The file limit is " + limit + " MiB (" + maxBytes.toLocaleString() + " bytes)."
        );
        return;
    }

    if (exampleUrl) showPreview(exampleUrl, false);
    else showPreview(URL.createObjectURL(file), true);
    emptyResults.hidden = true;
    setLoading(true);
    try {
        if (exampleUrl) {
            const example = await fetch(exampleUrl, {signal: controller.signal});
            if (!example.ok) throw new Error("Example image unavailable");
            file = await example.blob();
        }
        const body = new FormData();
        body.set("image", file, exampleUrl ? "validation-example.jpeg" : file.name);
        const response = await fetch(form.action, {
            method: "POST",
            headers: {Accept: "application/json"},
            body,
            signal: controller.signal,
        });
        const data = await response.json();
        if (activeRequest !== controller) return;
        if (!response.ok || data.error) {
            showError(data.error || "Could not classify this image. Try another image.");
            return;
        }
        setLoading(false);
        showResults(data);
        if (data.calibrated && heatmapControls) {
            heatmapFile = file;
            heatmapControls.hidden = false;
        }
    } catch {
        if (activeRequest === controller) showError("The local demo could not process this image. Try again.");
    } finally {
        if (activeRequest === controller) activeRequest = null;
    }
}

heatmapToggle?.addEventListener("click", async () => {
    if (!heatmapFile || heatmapRequest) return;
    const showing = heatmapToggle.getAttribute("aria-pressed") === "true";
    if (showing) {
        heatmapComparison.hidden = true;
        heatmapToggle.textContent = "Show heatmap";
        heatmapToggle.setAttribute("aria-pressed", "false");
        return;
    }

    const controller = new AbortController();
    if (!heatmapReady) {
        heatmapRequest = controller;
        heatmapToggle.disabled = true;
        heatmapStatus.textContent = "Generating heatmap…";
        try {
            const body = new FormData();
            body.set("image", heatmapFile, "heatmap-input");
            const response = await fetch(heatmapToggle.dataset.url, {
                method: "POST", body, signal: controller.signal, headers: {Accept: "application/json"},
            });
            if (!response.ok) throw new Error("Heatmap unavailable");
            const images = await response.json();
            if (heatmapRequest !== controller) return;
            for (const [key, panel] of Object.entries(heatmapPanels)) panel.src = images[key];
            heatmapReady = true;
            heatmapStatus.textContent = "";
        } catch {
            if (heatmapRequest === controller) heatmapStatus.textContent = "Could not generate the heatmap. Try again.";
            return;
        } finally {
            if (heatmapRequest === controller) {
                heatmapRequest = null;
                heatmapToggle.disabled = false;
            }
        }
    }
    heatmapComparison.hidden = false;
    heatmapToggle.textContent = "Hide heatmap";
    heatmapToggle.setAttribute("aria-pressed", "true");
    heatmapComparison.scrollIntoView({block: "nearest", behavior: reducedMotion.matches ? "instant" : "smooth"});
});

function classifySelectedFile() {
    const file = fileInput.files[0];
    if (!file) return;
    fileInput.value = "";
    classify(file);
}

fileInput.addEventListener("change", classifySelectedFile);
form.addEventListener("submit", (event) => {
    event.preventDefault();
    classifySelectedFile();
});

let dragDepth = 0;
dropZone.addEventListener("dragenter", (event) => {
    event.preventDefault();
    dragDepth++;
    dropZone.classList.add("is-dragging");
});
dropZone.addEventListener("dragover", (event) => event.preventDefault());
dropZone.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) dropZone.classList.remove("is-dragging");
});
dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    dropZone.classList.remove("is-dragging");
    if (event.dataTransfer.files[0]) classify(event.dataTransfer.files[0]);
});

for (const button of document.querySelectorAll("[data-example]")) {
    button.addEventListener("click", () => classify(null, button.dataset.example));
}
resetButton.hidden = previewWrap.hidden && error.hidden;
resetButton.addEventListener("click", () => {
    activeRequest?.abort();
    activeRequest = null;
    form.reset();
    setLoading(false);
    clearPreview();
    clearResults();
    clearError();
    resetButton.hidden = true;
    fileInput.focus();
});

const classPanel = byId("class-list");
const classSearch = byId("class-search");
const classEntries = Array.from(byId("class-grid").children, (item) => ({
    item, searchable: item.querySelector("strong").textContent.toLowerCase() + " " + item.dataset.wnid,
}));
byId("open-classes").addEventListener("click", (event) => {
    event.preventDefault();
    classPanel.open = true;
    classPanel.scrollIntoView({block: "start"});
    classSearch.focus({preventScroll: true});
});
classSearch.addEventListener("input", () => {
    const query = classSearch.value.trim().toLowerCase();
    let visible = 0;
    for (const {item, searchable} of classEntries) {
        item.hidden = !searchable.includes(query);
        if (!item.hidden) visible++;
    }
    byId("class-count").textContent = visible + (visible === 1 ? " class" : " classes");
    byId("no-classes").hidden = visible !== 0;
});
