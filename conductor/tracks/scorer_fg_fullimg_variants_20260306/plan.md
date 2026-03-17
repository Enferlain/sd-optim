# Scorer FG Logic Hardening + Full-Image Variants (2026-03-06)

- [x] Task: Harden foreground `textureclean` logic (mask coverage handling, starved ratio normalization, and anti-over-smoothing penalties) while preserving FG-focused behavior.
- [x] Task: Harden foreground `hybridnoise` logic for failure cases and small/unstable background masks while preserving FG-focused behavior.
- [x] Task: Add full-image variants for texture and hybrid-noise scorers that do not use rembg.
- [x] Task: Register new scorer identifiers in scorer loader/factory wiring and validate via manual smoke run.
