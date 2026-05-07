import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from model import GPT

# GPT2 configuration
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
BLOCK_SIZE = 256
VOCAB_SIZE = 50257

# AdamW optimizer parameters
LEARNING_RATE = 6e-4
MAX_ITERS = 600000
WEIGHT_DECAY = 1e-1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0

# Learning rate decay settings
LR_STEP_DIVISOR = 5
LR_STEP_GAMMA = 0.5

BATCH_SIZE = 32
VALID_BATCH_SIZE = BATCH_SIZE * 8
EVAL_INTERVAL = 1500

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_path: str,
    best_valid_loss: float,
    completed_steps: int,
    train_losses: list[float] | None = None,
    valid_losses: list[float] | None = None,
) -> None:
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "completed_steps": completed_steps,
        "best_valid_loss": best_valid_loss,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "model_config": {
            "vocab_size": VOCAB_SIZE,
            "block_size": BLOCK_SIZE,
            "n_layer": N_LAYER,
            "n_head": N_HEAD,
            "n_embd": N_EMBD,
        },
    }
    torch.save(checkpoint, checkpoint_file)
    print(f"Saved checkpoint to {checkpoint_path}")


def initialize_metrics_file(metrics_path: str, append: bool) -> None:
    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = metrics_file.exists() and metrics_file.stat().st_size > 0
    write_header = not (append and file_exists)
    mode = "a" if append and file_exists else "w"

    with metrics_file.open(mode, newline="") as file:
        if write_header:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "iteration",
                    "total_iterations",
                    "train_loss",
                    "valid_loss",
                    "batch_time_seconds",
                    "learning_rate",
                ]
            )
            file.flush()


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
        writer.writerow(
            [
                iteration,
                total_iterations,
                train_loss,
                valid_loss,
                batch_time_seconds,
                learning_rate,
            ]
        )
        file.flush()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        checkpoint_path: str,
        metrics_path: str,
        total_iterations: int,
        lr_step_size: int,
        pt_dtype: torch.dtype,
        load_model: bool,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_main_process = self.global_rank == 0
        self.device = torch.device("cuda", self.local_rank)

        self.model = model.to(device=self.device, dtype=pt_dtype)
        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        self.train_data = train_data
        self.valid_data = valid_data
        self.checkpoint_path = checkpoint_path
        self.metrics_path = metrics_path
        self.total_iterations = total_iterations

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            betas=(BETA1, BETA2),
            weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_step_size,
            gamma=LR_STEP_GAMMA,
        )

        self.train_losses: list[float] = []
        self.valid_losses: list[float] = []
        self.best_valid_loss = float("inf")
        self.completed_steps = 0

        if load_model:
            self._load_checkpoint()

        if self.is_main_process:
            initialize_metrics_file(self.metrics_path, append=self.completed_steps > 0)
        dist.barrier()

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.model.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.valid_losses = checkpoint["valid_losses"]
        self.completed_steps = checkpoint["completed_steps"]
        self.best_valid_loss = checkpoint["best_valid_loss"]

        if self.is_main_process:
            print(
                "Loaded checkpoint "
                f"{self.checkpoint_path} at completed_steps={self.completed_steps}"
            )
    
    def _save_training_state(self) -> None:
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_path=self.checkpoint_path,
            best_valid_loss=self.best_valid_loss,
            completed_steps=self.completed_steps,
            train_losses=self.train_losses,
            valid_losses=self.valid_losses,
        )

    def _reduce_mean(self, value: torch.Tensor) -> float:
        reduced_value = value.detach().clone()
        dist.all_reduce(reduced_value, op=dist.ReduceOp.SUM)
        reduced_value /= self.world_size
        return reduced_value.item()

    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, float]:
        batch_start_time = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)
        _, loss = self.model(source, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        self.optimizer.step()
        self.scheduler.step()

        batch_time_seconds = time.perf_counter() - batch_start_time
        train_loss = self._reduce_mean(loss)
        current_learning_rate = self.optimizer.param_groups[0]["lr"]

        return train_loss, batch_time_seconds, current_learning_rate

    @torch.no_grad()
    def _run_validation(self) -> float:
        self.model.eval()

        loss_totals = torch.zeros(2, device=self.device)
        for source, targets in self.valid_data:
            source = source.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            _, loss = self.model(source, targets)
            loss_totals[0] += loss.detach()
            loss_totals[1] += 1

        dist.all_reduce(loss_totals, op=dist.ReduceOp.SUM)
        self.model.train()

        if loss_totals[1].item() == 0:
            return float("inf")
        return (loss_totals[0] / loss_totals[1]).item()

    def train(self) -> None:
        if self.completed_steps >= self.total_iterations:
            if self.is_main_process:
                print("Checkpoint already contains all training steps for this dataset pass.")
            return

        self.model.train()
        train_sampler = self.train_data.sampler
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(0)

        if self.is_main_process:
            print("Training GPT2 over the dataset once with DDP")

        resume_step = self.completed_steps

        for step, (source, targets) in enumerate(self.train_data, start=1):
            if step <= resume_step:
                continue

            source = source.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            train_loss, batch_time_seconds, current_learning_rate = self._run_batch(source, targets)
            self.completed_steps = step

            valid_loss = None

            if step % EVAL_INTERVAL == 0 or step == self.total_iterations:
                
                valid_loss = self._run_validation()

                if self.is_main_process:
                    self.train_losses.append(train_loss)

                    if valid_loss is not None:
                        self.valid_losses.append(valid_loss)
                        if valid_loss < self.best_valid_loss:
                            self.best_valid_loss = valid_loss
                        self._save_training_state()

                    append_metrics_row(
                        metrics_path=self.metrics_path,
                        iteration=step,
                        total_iterations=self.total_iterations,
                        train_loss=train_loss,
                        valid_loss=valid_loss,
                        batch_time_seconds=batch_time_seconds,
                        learning_rate=current_learning_rate,
                    )

        dist.barrier()

def count_model_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return sum(parameter.numel() for parameter in model.parameters())

def prepare_dataloader(dataset: TensorDataset, batch_size: int, shuffle: bool) -> DataLoader:
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=sampler,
    )

def load_dataset(dataset_path: str) -> dict[str, torch.Tensor]:
    dataset = torch.load(dataset_path, map_location="cpu")
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Loaded dataset from {dataset_path}")
    return dataset

def ddp_setup() -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--model_path", type=str, default="model.pt")
    parser.add_argument("--dataset_path", type=str, default="wikitext_tensors.pt")
    parser.add_argument("--metrics_path", type=str, default="training_metrics.csv")
    return parser.parse_args()

def main():
    ddp_setup()
    args = get_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    pt_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = load_dataset(args.dataset_path)
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    x_valid = dataset["x_valid"]
    y_valid = dataset["y_valid"]

    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_valid, y_valid)

    train_loader = prepare_dataloader(train_dataset, BATCH_SIZE, shuffle=True)
    valid_loader = prepare_dataloader(valid_dataset, VALID_BATCH_SIZE, shuffle=False)

    model = GPT(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
    )

    total_params = count_model_parameters(model)
    trainable_params = count_model_parameters(model, trainable_only=True)
    iterations = len(train_loader)
    lr_step_size = max(iterations // LR_STEP_DIVISOR, 1)

    trainer = Trainer(
        model=model,
        train_data=train_loader,
        valid_data=valid_loader,
        checkpoint_path=args.model_path,
        metrics_path=args.metrics_path,
        total_iterations=iterations,
        lr_step_size=lr_step_size,
        pt_dtype=pt_dtype,
        load_model=args.load_model,
    )

    if trainer.is_main_process:
        print(args)
        print(model)
        print(f"Model moved to {device} with parameters of type: {next(unwrap_model(trainer.model).parameters()).dtype}")
        print(f"Optimizer initialized with learning rate {LEARNING_RATE}")
        print(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")
        print(f"Train set shape: x={x_train.shape}, y={y_train.shape}")
        print(f"Validation set shape: x={x_valid.shape}, y={y_valid.shape}")
        print(
            f"Using StepLR with step_size={lr_step_size} iterations "
            f"(computed as total_iterations // LR_STEP_DIVISOR = {iterations} // {LR_STEP_DIVISOR}) "
            f"and gamma={LR_STEP_GAMMA}"
        )

    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    main()
