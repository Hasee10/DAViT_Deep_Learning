# Assignment 3 — Cross-Domain Evaluation of DAViT
**Group:** Haseeb Arshad (23i-2578) | Haider Farooq (23i-2542) | Mudassir Asghar (23i-2577)
**Course:** Deep Learning — FAST-NUCES

---

## Overview

This folder contains the code for Assignment 3, which extends the Assignment 2 DAViT
reproduction by evaluating the pretrained model on a new, unseen dataset (CoronaHack
Chest X-Ray Dataset). The goal is to test whether DAViT generalises beyond the Kaggle
dataset it was originally trained on.

**Core question:**
> Does the DAViT model maintain its performance when moved to a new hospital domain?

---

## Folder Structure

```
code/
├── run_crossdomain_eval.py   ← Main script for Assignment 3 (cross-domain inference)
├── main.py                   ← Original DAViT main script (Assignment 2, CPU-modified)
├── model_davit.py            ← DAViT model architecture
├── model.py                  ← Generic ViT model wrapper
├── resnet.py                 ← ResNet model wrapper
├── inception.py              ← InceptionV3 model wrapper
├── densenet.py               ← DenseNet model wrapper

saved_models/
└── checkpoint-best-f1/
    └── davit_pneumonia_type.bin   ← Pretrained weights from Assignment 2

data/
└── coronahack/
    └── test/
        ├── BACTERIA/              ← Bacterial pneumonia images
        └── VIRUS/                 ← Viral pneumonia images

visualizations/
├── metric_comparison.png          ← Bar chart: Kaggle vs CoronaHack metrics
└── coronahack_confusion_matrix.png← Confusion matrix for new dataset
```

---

## Dataset Setup

### CoronaHack Chest X-Ray Dataset
- **Source:** https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset
- Download and extract the dataset from Kaggle
- From the Metadata.csv, filter rows where:
  - `Label = 'Pnemonia'` (confirmed pneumonia cases only)
  - `Label_2_Virus_category` is `'bacteria'` or `'Virus'`
- Copy the corresponding image files into:
  ```
  data/coronahack/test/BACTERIA/   ← all bacterial cases
  data/coronahack/test/VIRUS/      ← all viral cases
  ```
- Supported formats: `.jpeg`, `.jpg`, `.png`

---

## Running the Cross-Domain Evaluation

### Prerequisites
```bash
pip install torch torchvision transformers scikit-learn pillow tqdm
```

### Step 1 — Ensure pretrained weights are in place
The `davit_pneumonia_type.bin` checkpoint from Assignment 2 must be at:
```
saved_models/checkpoint-best-f1/davit_pneumonia_type.bin
```

### Step 2 — Run inference on the new dataset
```bash
cd code

python run_crossdomain_eval.py \
    --test_data_file ../data/coronahack/test \
    --output_dir ../saved_models \
    --model_name davit_pneumonia_type.bin \
    --model_name_or_path facebook/dinov2-large \
    --do_test \
    --eval_batch_size 16 \
    --classify_pneumonia_type
```

### Step 3 — Save the output log
```bash
python run_crossdomain_eval.py [args above] 2>&1 | tee ../logs/davit_coronahack.log
```

---

## Expected Output

```
***** Cross-Domain Test Results *****
  f1:          x.xxxx
  precision:   x.xxxx
  recall:      x.xxxx
  specificity: x.xxxx
  Acc:         x.xxxx
  AUC:         x.xxxx
  TP=xxx  FP=xx  FN=xxx  TN=xxx
```

---

## Key Differences from Assignment 2

| Aspect | Assignment 2 | Assignment 3 |
|--------|-------------|--------------|
| Dataset | Kaggle Chest X-Ray (Guangzhou children's hospital) | CoronaHack (multi-source, adult patients) |
| Script | train_type.sh → main.py | run_crossdomain_eval.py |
| Data format | PNEUMONIA/person_bacteria_*.jpeg | BACTERIA/*.jpg, VIRUS/*.jpg |
| Labels | inferred from filename | inferred from folder name |
| Model | same pretrained DAViT weights | same pretrained DAViT weights |
| Training | None (inference only) | None (inference only) |

---

## CPU Modifications Applied (inherited from Assignment 2)

All of the following modifications from Assignment 2 are already applied:
1. `torch.cuda.is_available = lambda: False` — disables CUDA detection
2. `args.device = torch.device('cpu')` — forces CPU throughout
3. `torch.load(..., map_location=torch.device('cpu'))` — CPU-safe checkpoint loading
4. Lazy image loading in Dataset (`__getitem__`) — prevents MemoryError

---

## Notes

- The model is **not retrained** on the new dataset. This is intentional — we are
  testing generalisation, not adaptation.
- Expected performance is **lower** than Assignment 2 due to domain shift.
- Batch size is set to 16 by default to reduce RAM usage. Reduce to 8 if needed.
- Evaluation on 500 images takes approximately 35-45 minutes on CPU.
