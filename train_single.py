import torch 
import torch.nn as nn
import argparse
import csv
import time
from pathlib import Path
from model import GPT

# GPT2 configuration
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
BLOCK_SIZE = 256
VOCAB_SIZE = 50257

# Adamw optimizer parameters
LEARNING_RATE = 6e-4
MAX_ITERS = 600000
WEIGHT_DECAY = 1e-1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0

# learning rate decay settings
LR_STEP_DIVISOR = 5
LR_STEP_GAMMA = 0.5

BATCH_SIZE = 4
EVAL_INTERVAL = 1500 

def get_batch(x: torch.Tensor, y: torch.Tensor, batch_size: int, iter: int):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = x[ iter * batch_size : (iter+1) * batch_size ]
    y = y[ iter * batch_size : (iter+1) * batch_size ]

    x, y = x.to(device), y.to(device)

    return x, y

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--model_path", type=str, default="model.pt")
    parser.add_argument("--dataset_path", type=str, default="wikitext_tensors.pt")
    parser.add_argument("--metrics_path", type=str, default="training_metrics.csv")

    return parser.parse_args()


def load_dataset(dataset_path: str) -> dict[str, torch.Tensor]:
    dataset = torch.load(dataset_path, map_location="cpu")
    print(f"Loaded dataset from {dataset_path}")
    return dataset


def load_model_weights(model: nn.Module, model_path: str, device: str) -> None:
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {model_path}")


def count_model_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return sum(parameter.numel() for parameter in model.parameters())


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    best_valid_loss: float,
    train_losses: list[float] | None = None,
    valid_losses: list[float] | None = None,
) -> None:
    checkpoint = {
        "best_valid_loss": best_valid_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses if train_losses is not None else [],
        "valid_losses": valid_losses if valid_losses is not None else [],
        "model_config": {
            "vocab_size": VOCAB_SIZE,
            "block_size": BLOCK_SIZE,
            "n_layer": N_LAYER,
            "n_head": N_HEAD,
            "n_embd": N_EMBD,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def get_saved_best_valid_loss(checkpoint_path: str) -> float:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        return float("inf")

    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "best_valid_loss" in checkpoint:
            return checkpoint["best_valid_loss"]
        valid_losses = checkpoint.get("valid_losses")
        if valid_losses:
            return min(valid_losses)
    return float("inf")


def initialize_metrics_file(metrics_path: str) -> None:
    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    with metrics_file.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "total_iterations", "train_loss", "valid_loss", "batch_time_seconds", "learning_rate"])


def append_metrics_row(
    metrics_path: str,
    iteration: int,
    total_iterations: int,
    train_loss: float,
    valid_loss: float | None,
    batch_time_seconds: float,
    learning_rate: float,
) -> None:
    with Path(metrics_path).open("a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([iteration, total_iterations, train_loss, valid_loss if valid_loss is not None else "", batch_time_seconds, learning_rate])
        file.flush()

def main():

    args = get_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pt_dtype = None
    if device == "cuda":
        pt_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        pt_dtype = torch.float32

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = GPT(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
    )

    dataset = load_dataset(args.dataset_path)
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    x_test = dataset["x_test"]
    y_test = dataset["y_test"]
    x_valid = dataset["x_valid"]
    y_valid = dataset["y_valid"]

    if args.load_model:
        load_model_weights(model, args.model_path, device)

    model.to(device=device, dtype=pt_dtype)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    total_params = count_model_parameters(model)
    trainable_params = count_model_parameters(model, trainable_only=True)

    print(model)
    print(f"Model moved to {device} with parameters of type: {next(model.parameters()).dtype}")
    print(f"Optimizer initialized with learning rate {LEARNING_RATE}")
    print(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")
    print(f"Train set shape: x={x_train.shape}, y={y_train.shape}")
    print(f"Test set shape: x={x_test.shape}, y={y_test.shape}")
    print(f"Validation set shape: x={x_valid.shape}, y={y_valid.shape}")

    iterations = int(x_train.shape[0] / BATCH_SIZE)
    lr_step_size = max(iterations // LR_STEP_DIVISOR, 1)

    train_losses = []
    valid_losses = []
    best_valid_loss = get_saved_best_valid_loss(args.model_path)
    initialize_metrics_file(args.metrics_path)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_step_size,
        gamma=LR_STEP_GAMMA,
    )

    model.train()

    print("Training GPT2 over the dataset once")
    print(
        f"Using StepLR with step_size={lr_step_size} iterations "
        f"(computed as total_iterations // LR_STEP_DIVISOR = {iterations} // {LR_STEP_DIVISOR}) "
        f"and gamma={LR_STEP_GAMMA}"
    )

    for i in range(iterations):
        batch_start_time = time.perf_counter()

        x_train_batch, y_train_batch = get_batch(x_train, y_train, BATCH_SIZE, i)

        logits, loss = model(x_train_batch, y_train_batch)
        train_losses.append(loss.item())

        optimizer.zero_grad() 
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        batch_time_seconds = time.perf_counter() - batch_start_time
        current_learning_rate = optimizer.param_groups[0]["lr"]

        valid_loss = None

        if(i % EVAL_INTERVAL == 0):
            train_loss = loss.item()

            model.eval()
            x_valid_batch, y_valid_batch = get_batch(x_valid, y_valid, BATCH_SIZE * 8, 0)
            logits, loss = model(x_valid_batch, y_valid_batch)
            valid_loss = loss.item()
            valid_losses.append(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    checkpoint_path=args.model_path,
                    best_valid_loss=best_valid_loss,
                    train_losses=train_losses,
                    valid_losses=valid_losses,
                )

            model.train()

        append_metrics_row(
            metrics_path=args.metrics_path,
            iteration=i + 1,
            total_iterations=iterations,
            train_loss=train_losses[-1],
            valid_loss=valid_loss,
            batch_time_seconds=batch_time_seconds,
            learning_rate=current_learning_rate,
        )



if __name__ == "__main__":
    main()
