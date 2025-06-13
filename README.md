

# 🦙💉 Llama-3 × ChatDoctor - Clinical-Tuned 8-Billion-Parameter Model

Fine-tuning **Meta Llama-3-8B-Instruct** on medical Q&A data à la 🔬 **ChatDoctor**  
with a *from-scratch* 1-GPU recipe, GGUF export for `llama.cpp`, and transparent logging via **Weights & Biases**.


## Table of Contents
1. [Project Goals](#project-goals)  
2. [Directory Layout](#directory-layout)  
3. [Quick Start](#quick-start)  
4. [Environment & Dependencies](#environment--dependencies)  
5. [Data Preparation](#data-preparation)  
6. [Training](#training)  
7. [Merging & GGUF Export](#merging--gguf-export)  
8. [Running Inference](#running-inference)  
9. [Experiment Tracking](#experiment-tracking)  
10. [Results](#results)  
11. [Limitations & Disclaimer](#limitations--disclaimer)  
12. [Acknowledgements](#acknowledgements)  



## Project Goals
* **Validate** that Llama-3-8B can be aligned to domain-specific expertise (here: clinical Q&A) with modest hardware.  
* Provide a **reproducible** end-to-end workflow: dataset curation → PEFT-LoRA fine-tune → model merge → quantisation → inference.  
* Serve as a **reference** repo for anyone wanting to tune Llama-3 on their own specialty corpus.

---

## Directory Layout
```

.
├─ adapter/                  # LoRA adapter checkpoints (PEFT)
├─ llama-3-8b/               # Raw Meta Llama-3-8B-Instruct weights (HF format)
├─ llama3-med/               # PEFT-fine-tuned adapter (LoRA)
├─ llama3-merged/            # Fully-merged FP16 model (HF .safetensors)
├─ llama.cpp/                # llama.cpp submodule fork (for GGUF + quant)
├─ wandb/                    # W\&B run logs & artifacts  ← *tracked*
├─ Llama\_finetune\_8B\_Chatdoctor.ipynb        # walk-through notebook  ← *tracked*
├─ .ipynb\_checkpoints/       # Jupyter autosaves         ← *tracked*
├─ llama3-chat-doctor-f16.gguf       # GGUF FP16 export
├─ llama3-chat-doctor-q4\_K\_M.gguf    # 4-bit quantised model
└─ .gitignore

````
> **Git policy:** Everything except **`wandb/`**, `*.ipynb`, and `.ipynb_checkpoints/` is ignored by default.

---

## Quick Start
```bash
# 1. clone repo & submodules
git clone --recurse-submodules https://github.com/<your-user>/Llama-Finetuning.git
cd Llama-Finetuning

# 2. create env
conda env create -f environment.yml   # or use pip installs from README
conda activate llama3-chatdoctor

# 3. download base weights (EULA!)  
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir llama-3-8b

# 4. run the notebook end-to-end *OR* use CLI:
python finetune.py --config configs/chatdoctor.yaml
````

---

## Environment & Dependencies

* Python 3.10+
* PyTorch >= 2.2 (CUDA 11.8)
* `transformers==4.41.0` / `accelerate==0.29.3`
* `peft==0.10.0` (LoRA)
* `bitsandbytes` (8-bit & 4-bit loaders)
* `sentencepiece`, `datasets`, `wandb`
* For GGUF + quantisation: compile **`llama.cpp`** with `make LLAMA_CUBLAS=1`.

<details>
<summary>Conda one-liner</summary>

```bash
conda create -n llama3-chatdoctor python=3.10 \
  pytorch=2.2 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

</details>

---

## Data Preparation

* Source corpus: **ChatDoctor** (English clinical dialogues), plus 10 k freshly-scraped MedQA pairs.
* Text is normalised, deduplicated, and split 90/5/5 into train/val/test.
* Prompt template (ChatML-style):

```text
<|im_start|>system
You are ChatDoctor, a large language model with medical expertise.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

---

## Training

### PEFT LoRA

```bash
accelerate launch finetune.py \
  --model_name_or_path llama-3-8b \
  --data_path data/med_qa.jsonl \
  --output_dir llama3-med \
  --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --bf16
```

* 1× RTX 4090 (24 GB) ≈ 11 hours.
* Peak GPU RAM \~22 GB thanks to 8-bit loaders + gradient checkpointing.

### Merge & Save

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("llama-3-8b", torch_dtype="float16")
model = PeftModel.from_pretrained(base, "llama3-med")
model = model.merge_and_unload()
model.save_pretrained("llama3-merged", safe_serialization=True)
```

---

## Merging & GGUF Export

1. **Convert HF → GGUF**

   ```bash
   python llama.cpp/convert.py llama3-merged llama3-chat-doctor-f16.gguf
   ```
2. **Quantise** (example: q4\_K\_M)

   ```bash
   ./llama.cpp/quantize \
     llama3-chat-doctor-f16.gguf llama3-chat-doctor-q4_K_M.gguf q4_K_M
   ```

---

## Running Inference

### With llama.cpp

```bash
./llama.cpp/main -m llama3-chat-doctor-q4_K_M.gguf \
  -p "<|im_start|>user\nWhat are the symptoms of anaemia?<|im_end|>\n<|im_start|>assistant\n"
```

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("llama3-merged")
model = AutoModelForCausalLM.from_pretrained("llama3-merged", torch_dtype="float16").cuda()

prompt = "What are first-line treatments for type 2 diabetes?"
out = model.generate(**tok(prompt, return_tensors="pt").to(model.device), max_new_tokens=256)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## Experiment Tracking

* Every run auto-logs to **Weights & Biases** (`wandb/` directory is committed).
* Compare loss curves & eval metrics on your project dashboard:
  `https://wandb.ai/<your-user>/llama3-chatdoctor`

  ![image](https://github.com/user-attachments/assets/45cdfeb3-a63d-41d4-af25-1029d91a2295)


---

## Results

| Metric     | Value (test set) | Notes           |
| ---------- | ---------------- | --------------- |
| Perplexity | **3.7**          | After 3 epochs  |
| BLEU-4     | **34.2**         | vs. 27 baseline |
| MedQA EM   | **61 %**         | base = 44 %     |

*(See `results.md` for full tables.)*

---

## Limitations & Disclaimer

* **Not a certified medical device.** Model outputs **must** be reviewed by qualified professionals.
* Domain data is English-only; performance on other languages is unverified.
* Training corpus may contain outdated or region-specific guidelines.
* Fine-tune recipe uses LoRA; for production you may prefer a full-parameter SFT.

---

## Acknowledgements

* Meta AI for releasing **Llama 3**.
* **ChatDoctor** authors ↔ [https://github.com/Kent0n-Li/ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor).
* Georgi Gerganov et al. for **llama.cpp**.
* Hugging Face ecosystem & the open-source ML community.


