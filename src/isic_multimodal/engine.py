from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm

from .losses import contrastive_alignment
from .metrics import calculate_metrics


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def run_epoch(model, loader, criterion, device, optimizer=None, alignment_weight=0.0,
              alignment_temperature=0.1, amp=True):
    training = optimizer is not None
    model.train(training)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    losses, all_y, all_p = [], [], []
    for batch in tqdm(loader, leave=False, desc="train" if training else "eval"):
        image = batch["image"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), _autocast(device, amp):
            output = model(image, metadata)
            loss = criterion(output["logits"], target)
            if alignment_weight > 0:
                if "image_embedding" not in output or "metadata_embedding" not in output:
                    raise ValueError("Alignment loss requires a multimodal model.")
                align = contrastive_alignment(output["image_embedding"], output["metadata_embedding"],
                                              alignment_temperature)
                loss = loss + alignment_weight * align
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses.append(float(loss.detach().cpu()))
        all_y.append(target.detach().cpu().numpy())
        all_p.append(output["logits"].detach().float().softmax(1).cpu().numpy())
    metrics = calculate_metrics(np.concatenate(all_y), np.concatenate(all_p))
    metrics["loss"] = float(np.mean(losses))
    return metrics


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ids, labels, probabilities = [], [], []
    for batch in tqdm(loader, leave=False, desc="predict"):
        output = model(batch["image"].to(device), batch["metadata"].to(device))
        ids.extend(batch["image_id"])
        labels.extend(batch["label"].numpy().tolist())
        probabilities.extend(output["logits"].softmax(1).cpu().numpy().tolist())
    return ids, np.asarray(labels), np.asarray(probabilities)

