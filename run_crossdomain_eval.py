"""
Assignment 3 - Cross-Domain Evaluation Script
DAViT: Cross-Domain Pneumonia Type Classification
Dataset: CoronaHack Chest X-Ray Dataset (new domain)
Model: Pretrained DAViT weights from Assignment 2

Usage:
    python run_crossdomain_eval.py \
        --test_data_file ../data/coronahack/test \
        --output_dir ./saved_models \
        --model_name davit_pneumonia_type.bin \
        --model_name_or_path facebook/dinov2-large \
        --do_test \
        --eval_batch_size 16 \
        --classify_pneumonia_type

Folder structure expected:
    data/coronahack/test/
        BACTERIA/   <- bacterial pneumonia images (.jpeg or .jpg)
        VIRUS/      <- viral pneumonia images (.jpeg or .jpg)

The positive class (label=0) is BACTERIA.
The negative class (label=1) is VIRUS.
"""

from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
from model_davit import Model
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score, confusion_matrix
)
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


# ── Force CPU ──────────────────────────────────────────────────────────────────
torch.cuda.is_available = lambda: False
DEVICE = torch.device("cpu")


class CrossDomainDataset(Dataset):
    """
    Lazy-loading dataset for cross-domain evaluation.
    Expects:
        root/BACTERIA/*.jpeg  (or .jpg)
        root/VIRUS/*.jpeg     (or .jpg)

    Labels: BACTERIA=0, VIRUS=1  (matches DAViT convention from Assignment 2)
    """

    def __init__(self, root_dir):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.samples = []   # list of (path, label)

        bacteria_dir = os.path.join(root_dir, "BACTERIA")
        virus_dir    = os.path.join(root_dir, "VIRUS")

        for d, label, name in [(bacteria_dir, 0, "BACTERIA"), (virus_dir, 1, "VIRUS")]:
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f"Expected folder '{d}' not found. "
                    f"Please organise the dataset as described in the README."
                )
            count = 0
            for fname in os.listdir(d):
                if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                    self.samples.append((os.path.join(d, fname), label))
                    count += 1
            logger.info(f"  {name}: {count} images loaded")

        random.shuffle(self.samples)
        logger.info(f"Total cross-domain test samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        pixel_values = self.transform(image)
        return pixel_values, torch.tensor(label, dtype=torch.long)


def run_test(args, model, dataset):
    sampler    = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler,
                            batch_size=args.eval_batch_size, num_workers=0)

    logger.info("***** Running cross-domain evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    y_preds, y_trues = [], []

    for batch in tqdm(dataloader, total=len(dataloader)):
        pixel_values, labels = batch
        pixel_values = pixel_values.to(DEVICE)
        labels       = labels.to(DEVICE)
        with torch.no_grad():
            probs = model(pixel_values=pixel_values)
            preds = torch.argmax(probs, dim=1)
            y_trues += labels.cpu().tolist()
            y_preds += preds.cpu().tolist()

    acc         = accuracy_score(y_trues, y_preds)
    f1          = f1_score(y_trues, y_preds, zero_division=0)
    recall      = recall_score(y_trues, y_preds, zero_division=0)
    precision   = precision_score(y_trues, y_preds, zero_division=0)
    auc         = roc_auc_score(y_trues, y_preds)
    tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    logger.info("***** Cross-Domain Test Results *****")
    logger.info(f"  f1:          {f1:.4f}")
    logger.info(f"  precision:   {precision:.4f}")
    logger.info(f"  recall:      {recall:.4f}")
    logger.info(f"  specificity: {specificity:.4f}")
    logger.info(f"  Acc:         {acc:.4f}")
    logger.info(f"  AUC:         {auc:.4f}")
    logger.info(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    return {
        "f1": f1, "precision": precision, "recall": recall,
        "specificity": specificity, "accuracy": acc, "auc": auc,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_total": len(dataset)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-domain evaluation of DAViT on a new chest X-ray dataset"
    )
    parser.add_argument("--test_data_file", required=True,
                        help="Path to test folder (must contain BACTERIA/ and VIRUS/ subfolders)")
    parser.add_argument("--output_dir", default="./saved_models")
    parser.add_argument("--model_name", default="davit_pneumonia_type.bin")
    parser.add_argument("--model_name_or_path", default="facebook/dinov2-large")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--classify_pneumonia_type", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Force CPU
    args.n_gpu  = 0
    args.device = DEVICE

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    logger.warning("device: cpu, n_gpu: 0")

    # ── Load model ─────────────────────────────────────────────────────────────
    logger.info("Loading DINOv2-large processor and backbone...")
    feature_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)
    vit               = AutoModel.from_pretrained(args.model_name_or_path)
    model             = Model(vit, feature_processor, args)

    checkpoint_path = os.path.join(
        args.output_dir, f"checkpoint-best-f1/{args.model_name}"
    )
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"), mmap=True, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    logger.info("Checkpoint loaded successfully.")

    # ── Load dataset ───────────────────────────────────────────────────────────
    logger.info(f"Loading cross-domain test data from: {args.test_data_file}")
    dataset = CrossDomainDataset(args.test_data_file)

    # ── Run evaluation ─────────────────────────────────────────────────────────
    if args.do_test:
        results = run_test(args, model, dataset)
        print("\n" + "="*50)
        print("CROSS-DOMAIN EVALUATION COMPLETE")
        print("="*50)
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k:15s}: {v:.4f}")
            else:
                print(f"  {k:15s}: {v}")


if __name__ == "__main__":
    main()
