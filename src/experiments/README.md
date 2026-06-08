# TactileGen Benchmark Experiments

This directory contains scripts for reproducing the benchmark tables from the TactileGen paper. The scripts write outputs under each benchmark directory's `output/<run-id>/` folder.

## Setup

Install TactileGen following the main project README and make sure the required checkpoints and API keys are configured. Most benchmark scripts use the project dependencies. The patch self-similarity metric additionally uses LPIPS:

```bash
uv pip install lpips
```

**Run the benchmark commands from the repository root so checkpoint and project-relative paths resolve consistently.**

## Text-to-Image Geometry

This benchmark reports CLIP score, average plate height, and plate height variation.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m experiments.t2i.01_generate_images --run-id paper_t2i
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m experiments.t2i.02_estimate_geometry --run-id paper_t2i
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m experiments.t2i.03_evaluate_clip --run-id paper_t2i
uv run --frozen python -m experiments.t2i.04_aggregate_metrics --run-id paper_t2i
```

## Text-to-Texture

This benchmark reports CLIP score, patch self-similarity, and HF ratio.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m experiments.text2texture.01_generate_textures --run-id paper_t2t
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m experiments.text2texture.02_estimate_normals --run-id paper_t2t
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m experiments.text2texture.03_evaluate_clip --run-id paper_t2t
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m experiments.text2texture.04_evaluate_patch_self_similarity --run-id paper_t2t
uv run --frozen python -m experiments.text2texture.05_evaluate_hf_ratio --run-id paper_t2t
uv run --frozen python -m experiments.text2texture.06_aggregate_metrics --run-id paper_t2t
```

## Tiling

Tiling consumes normal maps from a Text-to-Texture run. Pass `--t2t-run-id` to point at the source run.

```bash
uv run --frozen python -m experiments.tiling.01_generate_tiled --run-id paper_tiling --t2t-run-id paper_t2t --baselines resize center_crop highpass
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m experiments.tiling.01_generate_tiled --run-id paper_tiling --t2t-run-id paper_t2t --baselines tiled_diffusion_sdxl ours_inter ours_intra
uv run --frozen python -m experiments.tiling.02_evaluate_tileability --run-id paper_tiling
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m experiments.tiling.04_measure_timing --run-id paper_tiling --t2t-run-id paper_t2t --method tiled_diffusion_sdxl
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m experiments.tiling.04_measure_timing --run-id paper_tiling --t2t-run-id paper_t2t --method ours_inter
CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m experiments.tiling.04_measure_timing --run-id paper_tiling --t2t-run-id paper_t2t --method ours_intra
uv run --frozen python -m experiments.tiling.03_aggregate_metrics --run-id paper_tiling
```

## Partial Runs

All generation scripts support `--prompts START END` for sharding by prompt range. Some scripts also support `--baselines` for running a subset of methods. Reuse the same `--run-id` when combining shards, then run the evaluation and aggregation steps after all expected outputs are present.

Aggregation scripts fail if required metric files are missing for observed samples. If aggregation fails, rerun the missing evaluation step before reporting numbers.