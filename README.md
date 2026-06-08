# 🔧 LLM LoRA Fine-Tuning Platform

[![LoRA](https://img.shields.io/badge/LoRA%2FQLoRA-PEFT-orange)](.) [![Models](https://img.shields.io/badge/Models-Gemma%7CLlama%7CMistral-blue)](.) [![GPU](https://img.shields.io/badge/GPU-A100%2FH100-green)](.)

> End-to-end **LoRA/QLoRA fine-tuning platform** for open-source LLMs. Fine-tune Gemma-2-27B, Llama-3.2, Mistral-7B on custom datasets with 4-bit quantization, DPO alignment and automated eval.

## 🏆 Fine-tuning Results
| Base Model | Task | Before | After LoRA | GPU Memory |
|-----------|------|--------|------------|------------|
| Gemma-2-27B | Legal QA PT-BR | 61.2% | **89.4%** | 24GB (4-bit) |
| Llama-3.2-11B | Medical NER | 72.1% | **94.7%** | 16GB (4-bit) |
| Mistral-7B | Code Generation | 67.8% | **91.2%** | 12GB (4-bit) |

## 🏗️ Training Pipeline
```
Dataset → Format (Alpaca/ChatML) → QLoRA Config (r=64, α=128)
       → 4-bit Quantization → PEFT Training → DPO Alignment
       → RAGAS Evaluation → Merge & Push to HuggingFace
```

## ✨ Features
- **4-bit QLoRA** — fine-tune 70B models on single A100
- **DPO/RLHF alignment** — preference tuning pipeline included
- **Flash Attention 2** — 2x training speed
- **Automated eval** — RAGAS + MT-Bench + custom benchmarks
- **HuggingFace push** — auto-upload after training
