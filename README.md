# TactileGen

**Text-based Tactile Graphics Generation for the Visually Impaired**  
[Ruihan Gao\*](https://ruihangao.github.io/), [Joonghyuk Shin\*](https://joonghyuk.com/), [Ava Pun](https://avapun.com/), [Jaesik Park](https://jaesik.info/), [Wenzhen Yuan](https://siebelschool.illinois.edu/about/people/all-faculty/yuanwz), and [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)  
Carnegie Mellon University · Seoul National University · University of Illinois Urbana-Champaign  

![arXiv](https://img.shields.io/badge/arXiv-2606.00000-b31b1b.svg)
[![Project Page](https://img.shields.io/badge/Project_Page-Website-blue)](https://ruihangao.github.io/TactileGen/)
[![Checkpoints](https://img.shields.io/badge/Hugging_Face-Checkpoints-yellow)](https://huggingface.co/alex4727/tactilegen_ckpt)
[![Dataset](https://img.shields.io/badge/Hugging_Face-Dataset-yellow)](https://huggingface.co/datasets/alex4727/tactilegen_data)

## System requirements

This project has been tested on Linux x86_64 with the following stack:

- 8× NVIDIA A100-SXM4-80GB, driver 580.x (CUDA 13)
- Python 3.12.13
- PyTorch 2.9.1 + CUDA 12.8 wheels

## Installation

This repo uses the Python project manager [uv](https://docs.astral.sh/uv/).

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Run `uv sync` to create a Python virtual environment with all dependencies installed.
3. Download TactileGen checkpoints from Hugging Face Hub:
    ```bash
    uv run --frozen hf download alex4727/tactilegen_ckpt \
      --repo-type model \
      --local-dir ckpt
    ```
   Place the `ckpt` folder in `~/.cache/tactilegen/`, or specify its location via the environment variable
   `TACTILEGEN_CKPT_DIR`. If you keep it in the project root, point the runtime config at that directory:
    ```bash
    export TACTILEGEN_CKPT_DIR="$PWD/ckpt"
    ```

## Usage

### Gradio demo

Run the end-to-end Gradio demo with:

```bash
uv run gradio src/tactilegen/ui/app.py
```

### Environment variables

Set the following environment variables as needed. If they are missing at app startup, you will be prompted on the
terminal.

| Variable                    | Purpose                                                     | Default if unset                         | When required                   |
|-----------------------------|-------------------------------------------------------------|------------------------------------------|---------------------------------|
| `HF_TOKEN`                  | HuggingFace Hub access (gated weights, higher rate limits)  | None                                     | Always, when downloading models |
| `GEMINI_API_KEY`            | Google Gemini API                                           | None                                     | Only when using Nano Banana     |
| `TACTILEGEN_CKPT_DIR`       | Override default location for TactileGen custom checkpoints | `~/.cache/tactilegen/ckpt`               | Optional                        |
| `HF_HOME`                   | Override default location for Hugging Face model weights    | `~/.cache/huggingface`                   | Optional                        |
| `DIFFSYNTH_MODEL_BASE_PATH` | Override default location for DiffSynth model weights       | `./models` relative to the current shell | Optional                        |

## Development

### Project structure

- `src/tactilegen/`: Main source code.
    - `assets/`: Image assets used during generation, and example assets for the Gradio app.
    - `generation/`: Image generation, texture generation, and segmentation.
    - `geometry/`: Mesh and braille creation.
    - `ui/`: Gradio interface.
- `tests/`: Testing code.

### Code formatting and linting

This project uses [ruff](https://docs.astral.sh/ruff/) for formatting and linting:

```bash
uv run --frozen ruff format src/
uv run --frozen ruff check --fix src/
```

### Testing

This project uses [pytest](https://docs.pytest.org/) for tests. Run core tests with:

```bash
uv run --frozen pytest -q tests
```

#### Slow tests

End-to-end tests that run inference on a CUDA GPU are marked `@pytest.mark/slow` and **skipped by default**. To run
them, use:

```bash
uv run --frozen pytest -q tests -m slow
```

#### Regression tests

Regression tests pin outputs against snapshot files committed under `tests/<package>/<test_module_stem>/`. When a
snapshot *intentionally* changes (e.g. due to an algorithm change), you can update these snapshots with:

```bash
uv run --frozen pytest -q tests --force-regen
```

After regenerating, commit the updated snapshot files alongside the code change.

#### Debugging test outputs

Some tests save additional outputs to `/tmp/pytest-of-<username>/pytest-<number>/` to assist with visual debugging. You
can change this output directory with

```bash
uv run --frozen pytest -q tests --basetemp <output_directory>
```

## Text-to-Texture Model Training

We delegate Qwen-Image LoRA training to [DiffSynth-Studio](https://github.com/modelscope/diffsynth-studio). The training data is released on HuggingFace in the CSV format expected by DiffSynth:

```bash
export TACTILEGEN_TEXTURE_DATA=/path/to/tactilegen_data

uv run --frozen hf download alex4727/tactilegen_data \
  --repo-type dataset \
  --local-dir "$TACTILEGEN_TEXTURE_DATA"
```

The downloaded dataset should have this layout:

```text
$TACTILEGEN_TEXTURE_DATA/
  tactile_data.csv
  images/
    nb_000000.png
    nbp_000000.png
    real_000000.jpg
```

Then run training from the DiffSynth-Studio repository. These instructions are checked against DiffSynth-Studio commit `83eece4faf52ab392ca707ad643ab62ca2f58773`:

```bash
accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path "$TACTILEGEN_TEXTURE_DATA" \
  --dataset_metadata_path "$TACTILEGEN_TEXTURE_DATA/tactile_data.csv" \
  --data_file_keys image \
  --max_pixels 1048576 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 100 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path /path/to/output/tactile_qwen_lora \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --save_steps 100 \
  --enable_wandb_log \
  --gradient_accumulation_steps 4
```

Configure `accelerate` for your local hardware before launching (e.g., # of gpus/processes). Our released texture LoRA was trained on 8x A100 80GB GPUs with per-gpu batch size of 1 and gradient accumulation 4, giving an effective batch size of 32. We stopped at 3,000 steps after validation; you can stop earlier or later based on your own validation samples. 

## Citation

If you find this work useful, please cite:

```bibtex
@article{gao2026tactilegen,
  title={Text-based Tactile Graphics Generation for the Visually Impaired},
  author={Gao, Ruihan and Shin, Joonghyuk and Pun, Ava and Park, Jaesik and Yuan, Wenzhen and Zhu, Jun-Yan},
  journal={arXiv preprint arXiv:2606.00000},
  year={2026}
}
```

## Acknowledgments

This codebase is released with a clean Git history. All students (Ruihan Gao, Joonghyuk Shin, and Ava Pun) made substantial contributions to both the research project and code development.
<!-- Add other Acks here -->