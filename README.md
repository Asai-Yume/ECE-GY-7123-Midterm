# Text-to-SVG Generation via QLoRA Fine-Tuning

**NYU Deep Learning Spring 2026 — Midterm Kaggle Competition**

Fine-tuning small language models (Qwen3.5-2B) to generate Scalable Vector Graphics (SVG) from text descriptions using QLoRA with 4-bit quantization.

**Best Score: 14.35** (V8, completion-only SFT) · **Best SFT: 14.20** (V7) · **SFT+DPO: 14.19**

---

## Results

| Version | Model | Method | Valid SVGs | Kaggle Score | Key Change |
|---------|-------|--------|-----------|-------------|------------|
| V1 | Qwen2.5-Coder-1.5B | SFT | 22/1000 | 11.23 | Baseline |
| V2 | Qwen2.5-Coder-1.5B | SFT | 244/1000 | 11.83 | Fixed inference pipeline |
| V3 | Qwen3.5-2B | SFT | 320/1000 | 10.96 | Adapter broken* |
| V4 | Qwen3.5-2B | SFT | 357/1000 | 11.07 | xmlns bug + adapter broken* |
| V5 | Qwen3.5-2B | SFT | 906/1000 | 10.21 | Greedy decoding, adapter broken* |
| DPO (v6) | Qwen3.5-2B | SFT+DPO | 197/1000 | 14.19 | Preference optimization |
| V7 | Qwen3.5-2B | SFT | 867/1000 | **14.20** | Compressed data + adapter fix |
| V8 | Qwen3.5-2B | SFT | 748/1000 | **14.35** | Completion-only training |

\*V3–V5 ran on the unfine-tuned base model due to a silent adapter key mismatch between Unsloth and PEFT (see [Key Findings](#key-findings)).

### Ablation: LoRA Target Modules

Trained on V7 configuration, varying only `target_modules`:

| Variant | Valid SVGs | Fallback Rate | Color Match | Top viewBox | Mean Paths |
|---------|-----------|---------------|-------------|-------------|------------|
| Both (baseline) | 867 | 13.3% | 59.5% | 200×200 (800) | 2.2 |
| Attention-only | 692 | 30.8% | 30.6% | 32×32 (432) | 4.1 |
| MLP-only | 866 | 13.4% | 59.8% | 200×200 (795) | 2.2 |

**Finding:** MLP layers alone account for nearly all SVG generation capability. Attention-only fine-tuning produces a qualitatively different output distribution (icon-style 32×32 SVGs).

---

## Key Findings

1. **Adapter Key Mismatch (Critical Bug):** Unsloth saves adapter weights with keys like `...language_model.layers.X...lora_A.weight`, but PEFT expects `...layers.X...lora_A.default.weight`. All V3–V5 submissions silently ran on the base model. Fixed via same-session inference.

2. **Data Compression (+3 points):** Training SVGs contained 15-digit floats (e.g., `151.14999389648438`). Truncating to 1 decimal reduced mean SVG length from 2,524 → 1,076 chars (58%) with zero visual change. This was the single largest score improvement.

3. **MLP Dominance:** MLP-only LoRA fine-tuning (gate/up/down_proj) replicates 99.9% of full model performance. Attention-only learns a completely different output distribution.

4. **DPO Quality–Reliability Trade-off:** DPO scores 14.19 with only 197 valid SVGs vs. SFT's 14.20 with 867, implying ~4× higher per-SVG quality but 80% fallback rate.

---

## Repository Structure

```
├── notebooks/
│   ├── training_v1_v2.ipynb      # Qwen2.5-Coder-1.5B baseline
│   ├── training_v3.ipynb         # First Qwen3.5-2B attempt (adapter broken)
│   ├── training_v5.ipynb         # Greedy decoding experiment (adapter broken)
│   ├── training_v7.ipynb         # Best SFT: compressed data + same-session inference
│   ├── training_v8.ipynb         # Completion-only SFT variant
│   ├── training_dpo.ipynb        # SFT + DPO pipeline
│   └── visualizations.ipynb            # All report figures
├── scripts/
│   ├── fix_adapter_keys.py             # Remap Unsloth adapter keys → PEFT format
├── submissions/
│   ├── submission_v7.csv               # SFT both modules (score: 14.20)
│   ├── submission_v7_attn.csv          # Ablation: attention-only
│   ├── submission_v7_mlp.csv           # Ablation: MLP-only
│   ├── submission_v8.csv               # Completion-only SFT (score: 14.35)
│   └── submission_final.csv            # SFT+DPO (score: 14.19)
├── figures/
│   ├── score_progression.png
│   ├── compression_analysis.png
│   ├── loss_curves.png
│   ├── ablation_comparison.png
│   ├── ablation_viewbox.png
│   └── output_length_dist.png
├── report/
│   ├── acl_latex.tex                   # ACL-format report
│   └── custom.bib                      # Bibliography
└── README.md
```

---

## Reproduction

### Requirements

- Google Colab with GPU (T4 minimum for inference, RTX PRO 6000 / A100 used for training)
- Google Drive for checkpoint storage
- Python 3.10+

### Quick Start (Inference Only)

1. Download the V7 adapter from [TODO: link or instructions]
2. Open `notebooks/inference_adapter_fixed.ipynb` in Colab
3. Update `ADAPTER_PATH` to point to your adapter directory
4. Run all cells — generates 1,000 SVGs in ~5 hours on T4

### Full Training (V7)

1. Upload `train.csv` and `test.csv` to Google Drive
2. Open `notebooks/training_v7_colab.ipynb` in Colab
3. Update `PROJECT_DIR` to your Drive path
4. Run all cells — trains in ~16 hours on RTX PRO 6000, then runs inference in the same session

### Ablation Experiments

To reproduce the LoRA target module ablation:

1. Copy `training_v7_colab.ipynb` twice
2. In the attention-only copy, change `target_modules` to `['q_proj','k_proj','v_proj','o_proj']`
3. In the MLP-only copy, change `target_modules` to `['gate_proj','up_proj','down_proj']`
4. Update `adapter_save_dir` and `SUBMISSION_PATH` in each copy
5. Run each notebook end-to-end

### DPO Training

1. Open `notebooks/training_dpo_colab.ipynb` in Colab
2. Set `RUN_SFT = True` for first run (or `False` if reusing a saved SFT adapter)
3. Set `RUN_BUILD_DPO = True` and `RUN_DPO = True`
4. Mining + DPO training takes ~4 hours on top of SFT

### Generating Report Figures

1. Place all submission CSVs, `train.csv`, and `test.csv` in the working directory
2. Run `notebooks/visualizations.ipynb`
3. Figures are saved to `figures/`

---

## Competition Details

- **Task:** Generate valid SVG code from 1,000 text prompts
- **Scoring:** Geometric mean of Visual Fidelity (85%, SSIM + Edge-F1), Structural Similarity (12%, tree edit distance), Compactness (3%)
- **Constraints:** Valid XML, max 16,000 chars, max 256 paths, 256×256 canvas, allowed tags whitelist
- **Dataset:** 50,000 training pairs (prompt, SVG), 1,000 test prompts

## Training Configuration (V7 — Best SFT)

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3.5-2B (via Unsloth) |
| Quantization | 4-bit NF4, double quant |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Target modules | q/k/v/o_proj + gate/up/down_proj |
| Trainable params | 21.8M / 2.2B (0.98%) |
| Learning rate | 1e-5 (cosine schedule) |
| Epochs | 3 |
| Batch size | 32 |
| Training samples | 48,375 (compressed) |
| Training time | 950 min on RTX PRO 6000 |
| Final loss | 0.4441 |

---

## AI Tooling Disclosure

- **Claude (Anthropic):** Coding assistance, debugging, report drafting
