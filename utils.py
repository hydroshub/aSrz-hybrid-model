import torch
import random
import pickle
import warnings
import numpy as np
from pathlib import Path
import config

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def masked_r2(pred, target, eps=1e-6):
    valid = torch.isfinite(target)  # more robust than ~torch.isnan
    pred = pred[valid]
    target = target[valid]

    if pred.numel() == 0:
        warnings.warn("masked_r2: No valid elements! Returning 0.0")
        return torch.tensor(0.0, device=target.device)

    # If target is constant, R2 is defined as 0.0
    if torch.allclose(target, target.mean()):
        warnings.warn("masked_r2: Target is constant! Returning 0.0")
        return torch.tensor(0.0, device=target.device)

    ss_res = torch.sum((pred - target) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)

    r2 = 1.0 - ss_res / (ss_tot + eps)

    return 1 - r2

def masked_r2_per_basin(pred, target, min_valid=30, min_var=1):

    B = pred.shape[0]
    scores = []

    for b in range(B):
        p = pred[b, :, 0]
        t = target[b, :, 0]
        valid = ~torch.isnan(t)

        p_valid = p[valid]
        t_valid = t[valid]

        if p_valid.numel() >= min_valid:
            var = torch.var(t_valid)
            if var.item() > min_var:
                mse = torch.mean((p_valid - t_valid) ** 2)
                nse = 1.0 - mse / (var + 1e-6)
                scores.append(nse)

    if not scores:
        return torch.tensor(float("nan"), device=pred.device)
    
    return 1.0 - torch.stack(scores).mean()


def masked_corr(pred, target):
    valid = ~torch.isnan(target)
    pred = pred[valid]
    target = target[valid]

    if pred.numel() == 0:
        return torch.tensor(0.0, device=target.device)

    vx = pred - pred.mean()
    vy = target - target.mean()
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-6)
    return 1.0 - corr

def spearman_corr_penalty(x, y, eps=1e-6):
    """
    Penalize lack of monotonic relationship between x and y.
    """
    x = x.squeeze()
    y = y.squeeze()

    # Rank transform
    x_rank = torch.argsort(torch.argsort(x))
    y_rank = torch.argsort(torch.argsort(y))

    # Centered ranks
    x_rank = x_rank.float() - x_rank.float().mean()
    y_rank = y_rank.float() - y_rank.float().mean()

    # Spearman = Pearson(rank(x), rank(y))
    numerator = torch.sum(x_rank * y_rank)
    denominator = torch.sqrt(torch.sum(x_rank ** 2) * torch.sum(y_rank ** 2)) + eps
    spearman_r = numerator / denominator

    # Return penalty (want correlation → 1, so penalty → 0)
    return 1.0 - spearman_r

def compute_losses(pred, y, z):
    return {
        "q": masked_r2(pred["q"][:, :, 0], y[:, :, 0]),
        "et": masked_r2(pred["et"][:, :, 0], y[:, :, 1]),
        "swe": masked_r2(pred["swe"][:, :, 0], y[:, :, 2]),
        "twsa": masked_r2(pred["twsa_anomaly"][:, :, 0], z[:, :, 0]),
    }


def train_model(
    model, train_loader, val_loader, device, model_path, log_path,
    num_epochs=1000, seed=42, early_stop_patience=10,
    loss_weights=[1.0, 1.0, 1.0, 1.0], grad_clip=10.0, date_seq=None, show_progress=False
):
    # Set random seed
    set_seed(seed)
    model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    no_improve_epochs = 0

    weights = torch.tensor(loss_weights, device=device)

    for epoch in range(1, num_epochs + 1):
        # -----
        # Training phase
        # -----
        model.train()
        train_sums = {k: 0.0 for k in ["q", "et", "swe", "twsa"]}
        total_train_loss = 0.0

        for xb, yb, zb in train_loader:
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)

            optimizer.zero_grad()
            states = model.init_states(batch_size=xb.shape[0], device=device)
            pred = model(xb, date_seq=date_seq, mode="train", show_progress=show_progress, states=states)

            # -----
            # Loss calculation
            # -----
            loss_dict = compute_losses(pred, yb, zb)

            loss = (
                weights[0] * loss_dict["q"] +
                weights[1] * loss_dict["et"] +
                weights[2] * loss_dict["swe"] +
                weights[3] * loss_dict["twsa"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()

            total_train_loss += loss.item()
            for k in train_sums:
                train_sums[k] += loss_dict[k].item()

        # -----
        # Validation phase
        # -----
        model.eval()
        val_sums = {k: 0.0 for k in train_sums}
        total_val_loss = 0.0

        with torch.inference_mode():
            for xb, yb, zb in val_loader:
                xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)

                states = model.init_states(batch_size=xb.shape[0], device=device)
                pred = model(xb, date_seq=date_seq, mode="train", show_progress=show_progress, states=states)

                # Validation loss - no spinup masking needed
                loss_dict = compute_losses(pred, yb, zb)

                loss = (
                    weights[0] * loss_dict["q"] +
                    weights[1] * loss_dict["et"] +
                    weights[2] * loss_dict["swe"] +
                    weights[3] * loss_dict["twsa"]
                )
                total_val_loss += loss.item()
                for k in val_sums:
                    val_sums[k] += loss_dict[k].item()

        # -----
        # Logging and scheduler step
        # -----
        n_train, n_val = len(train_loader), len(val_loader)
        train_avg = {k: v / n_train for k, v in train_sums.items()}
        val_avg = {k: v / n_val for k, v in val_sums.items()}
        train_loss = total_train_loss / n_train
        val_loss = total_val_loss / n_val

        print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        print("  -> " + " | ".join([f"{k}: {train_avg[k]:.4f}/{val_avg[k]:.4f}" for k in train_avg]))

        scheduler.step(val_loss)

        # -----
        # Write log to file
        # -----
        log_line = (
            f"Epoch {epoch:03d} | Train: loss={train_loss:.4f}, " +
            ", ".join([f"{k}={train_avg[k]:.4f}" for k in train_avg]) +
            f" | Val: loss={val_loss:.4f}, " +
            ", ".join([f"{k}={val_avg[k]:.4f}" for k in val_avg])
        )
        write_mode = "w" if epoch == 1 else "a"
        with open(log_path, write_mode) as f:
            f.write(log_line + "\n")

        # -----
        # Early stopping
        # -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_path)
            print("  -> Best model saved.")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break
