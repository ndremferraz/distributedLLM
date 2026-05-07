import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model import GPT


DEFAULT_MODEL_CONFIG = {
    "vocab_size": 50257,
    "block_size": 256,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Input text prompt for generation.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoint/model.pt",
        help="Path to a saved model checkpoint.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Number of tokens to generate after the prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling cutoff. Omit for no top-k filtering.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Sample from the distribution instead of greedy decoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on, for example 'cuda', 'cuda:0', or 'cpu'.",
    )
    return parser.parse_args()


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device is not None:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_checkpoint(model_path: str) -> dict:
    checkpoint_file = Path(model_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    return torch.load(checkpoint_file, map_location="cpu")


def build_model_from_checkpoint(checkpoint: dict, device: torch.device, dtype: torch.dtype) -> GPT:
    model_config = checkpoint.get("model_config", DEFAULT_MODEL_CONFIG)
    model = GPT(
        vocab_size=model_config["vocab_size"],
        block_size=model_config["block_size"],
        n_layer=model_config["n_layer"],
        n_head=model_config["n_head"],
        n_embd=model_config["n_embd"],
    )

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def main():
    args = get_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint = load_checkpoint(args.model_path)
    model = build_model_from_checkpoint(checkpoint, device=device, dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Prompt tokenized to an empty sequence. Provide non-empty input text.")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_k=args.top_k,
        )

    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(generated_text)


if __name__ == "__main__":
    main()
