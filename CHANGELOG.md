# Changelog

## Classification demo redesign

- Theme changes use a wavy edge and vertical silk strands. Dark sweeps down and light sweeps up, with an offline canvas fallback and instant switching for reduced motion.

- Upload errors now distinguish file size, pixel count, and multipart limits. Selecting a file clears the previous error and allows retries of the same file.

- Added a responsive offline interface with theme preferences, local validation examples, ranked results, and searchable Tiny-ImageNet classes.
- The demo selects the latest epoch in a standard finetuning directory, supports explicit checkpoint paths, and displays accuracy from matching validation records.
- Added a low-confidence warning, bounded in-memory upload handling, and visible file errors. Inference uses the existing model and preprocessing.
