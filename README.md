# Training LLMs to Reason in Parallel with Global Forking Tokens (SSFT)

<!-- ===== Top badges / quick links (like ParaThinker) ===== -->
<p align="center">
  <!-- arXiv -->
  <a href="https://arxiv.org/abs/2510.05132" title="Read the SSFT paper on arXiv">
    <img src="https://img.shields.io/badge/arXiv-2510.05132-b31b1b?style=flat" alt="arXiv: 2510.05132" />
  </a>

  <!-- HF model: SSFT-32B-N6 -->
  <a href="https://huggingface.co/shengjia-toronto/ssft-32B-N6" title="Hugging Face: SSFT-32B-N6">
    <!-- In path-form: use -- for literal hyphens inside label/message -->
    <img src="https://img.shields.io/badge/Hugging%20Face-SSFT--32B--N6-yellow?logo=huggingface&logoColor=black&style=flat" alt="Hugging Face — SSFT-32B-N6" />
  </a>

  <!-- HF model: GRPO Test (SSFT-32B) -->
  <a href="https://huggingface.co/shengjia-toronto/grpo-test-ssft-32B" title="Hugging Face: GRPO Test (SSFT-32B)">
    <img src="https://img.shields.io/badge/Hugging%20Face-GRPO%20Test%20(SSFT--32B)-blue?logo=huggingface&logoColor=white&style=flat" alt="Hugging Face — GRPO Test (SSFT-32B)" />
  </a>

  <!-- HF model: GRPO (SSFT-32B) 10 extra steps -->
  <a href="https://huggingface.co/shengjia-toronto/ssft32b_grpo_bs256_step10" title="Hugging Face: GRPO (SSFT-32B)">
    <img src="https://img.shields.io/badge/Hugging%20Face-GRPO%20(SSFT--32B)-blue?logo=huggingface&logoColor=white&style=flat" alt="Hugging Face — GRPO (SSFT-32B) 10 steps" />
  </a>
</p>


<!-- ===== Front figure (SVG) ===== -->
<p align="center">
  <img src="assets/frontfigure_rl.svg" alt="SSFT Front Figure" width="85%">
</p>

<!-- ===== Mini navigation (clickable anchors) ===== -->
<p align="center">
  <a href="#updates">💡 Updates</a> |
  <a href="#set-supervised-fine-tuning">🧠 Set Supervised Fine-Tuning</a> |
  <a href="#open-source-list">🧾 Release List</a> |
  <a href="#instructions">⚙️ Instructions</a> |
</p>

---

## Updates

- **[Contact]** If you have questions or are interested in collaborating, feel free to reach out to me at <sheng.jia@mail.utoronto.ca>. I’ll be actively updating this repository with code, documentation, and new checkpoints.

- **[2025-12-08]** Released checkpoints  
  <a href="https://huggingface.co/shengjia-toronto/ssft32b_grpo_bs256_step10" title="Hugging Face: (Stage 2, only 10 RL steps on global forking tokens) (GRPO-SSFT-32B-10steps)">
    <img src="https://img.shields.io/badge/Hugging%20Face-GRPO%20(SSFT--32B)-blue?logo=huggingface&logoColor=white&style=flat" alt="Hugging Face — GRPO Test (SSFT-32B)" />
  </a>

- **[2025-11-05]** Released checkpoints  
  <a href="https://huggingface.co/shengjia-toronto/ssft-32B-N6" title="Hugging Face: (Stage 1) SSFT-32B-N6">
    <img src="https://img.shields.io/badge/Hugging%20Face-SSFT--32B--N6-yellow?logo=huggingface&logoColor=black&style=flat" alt="Hugging Face — SSFT-32B-N6" />
  </a>
  <a href="https://huggingface.co/shengjia-toronto/grpo-test-ssft-32B" title="Hugging Face: (Stage 2 early experiment) (GRPO-TEST-SSFT-32B)">
    <img src="https://img.shields.io/badge/Hugging%20Face-GRPO%20Test%20(SSFT--32B)-blue?logo=huggingface&logoColor=white&style=flat" alt="Hugging Face — GRPO Test (SSFT-32B)" />
  </a>

- **[2025-11-05]** Released our **evaluation code** built on **LightEval**  
  → [`lighteval`](./lighteval) · [`ssft_eval.sh`](./ssft_eval.sh) · [`grpo_ssft_eval.sh`](./grpo_ssft_eval.sh)

- **[2025-10-01]** arXiv preprint released: **“Training LLMs to Reason in Parallel with Global Forking Tokens”** → https://arxiv.org/abs/2510.05132

---

## Set Supervised Fine-Tuning

<p align="center">
  <img src="assets/ssft_alg.svg" alt="SSFT Overview" width="90%">
</p>

---

## Results
> **Note:** When evaluating **SSFT-32B** ([🤗 HF link](https://huggingface.co/shengjia-toronto/ssft-32B-N6)), use `<think1>` for **Pass@1**, and use the set `<think1>...<think6>` (parallel generations) for **Cons@k**. Our custom **LightEval** code inserts these tags automatically. If you’re using other frameworks and don’t want to manage `<think i>` prompting, try our **GRPO fine-tuned** model ([🤗 HF link](https://huggingface.co/shengjia-toronto/ssft32b_grpo_bs256_step10)), which uses RL to only optimize global forking tokens for selecting the optimal tag per question (very efficient with 1k data from DAPO-17k).  SSFT-GRPO models can sample the optimal think tag, so you can just run them directly for your questions.


### Pass@1: Average performance of individual generations (\<think1\> prompted)
| Model         | AIME 2024 | AIME 2025 | MATH-500 | GPQA-D | Average |
|--------------|:---------:|:---------:|:--------:|:------:|:-------:|
| **SSFT-32B** | **64.06** | **58.13** | **90.02** | 60.39  | **68.15** |

### Average of Native Cons@6: Average performance of majority voting with 6 parallel generations
| Model         | AIME 2024 | AIME 2025 | MATH-500 | GPQA-D | Average |
|--------------|:---------:|:---------:|:--------:|:------:|:-------:|
| **SSFT-32B** | **75.45** | **73.94** | **96.47** | **63.05** | **77.23** |

### Cons@32: Majority voting performance with large number of parallel generations
| Model         | AIME 2024 | AIME 2025 | MATH-500 | GPQA-D | Average |
|--------------|:---------:|:---------:|:--------:|:------:|:-------:|
| **SSFT-32B** | **83.33** | **86.67** | **96.80** | 61.62 | **82.11** |


---

## Release List
- [x] **Checkpoint: ssft-32B HF repo** 
- [x] **Code for evaluating ssft-32B**
- [ ] **Evaluation script with tensor parallel so >100GB VRAM per worker is not required**
- [ ] **Code for training ssft-32B**
- [ ] **Code for additional RFT ssft-32B -> grpo-ssft-32B**
- [ ] **More detailed instructions on both training and evaluation**
- [x] **Checkpoint: grpo-test-ssft-32B**
- [x] **Checkpoint: gfpo-ssft-32B-bs256-step10**
- [ ] **Checkpoint: grpo-ssft-32B**


---

## Instructions
> **Note (Compute & runtime):** We used a single AWS EC2 instance **p6-b200.48xlarge** (8× **B200** GPUs) to conduct both **SSFT-32B** training and evaluation. Training took ~**6 hours** end-to-end, and evaluation for each task took roughly **1.5–2 hours**.
### Environment
```bash
git clone https://github.com/Sheng-J/SSFT.git
cd SSFT
uv venv ssft_env --python 3.11 && source ssft_env/bin/activate && uv pip install --upgrade pip
uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
uv pip install -e ./transformers
uv pip install -e ./lighteval
```

### Evaluating SSFT-32B on AIME25/24, MATH-500, GPQA-D (Cons@6 Cons@32 Pass@1)
```bash
. ssft_eval.sh
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{jia2025training,
  title={Training Large Language Models To Reason In Parallel With Global Forking Tokens},
  author={Jia, Sheng and Wang, Xiao and Kasiviswanathan, Shiva Prasad},
  journal={arXiv preprint arXiv:2510.05132},
  year={2025}
}

