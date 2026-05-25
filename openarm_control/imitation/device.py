"""Torch device selection + reporting for the learning harness (extension phase F3).

Centralises CUDA detection so the trainers (BC now; ACT/Diffusion in I2) run on the
GPU when one is available and fall back to CPU otherwise.
"""
import torch


def get_device(prefer_cuda=True):
    """Return ``'cuda'`` if a CUDA GPU is available (and preferred), else ``'cpu'``."""
    return "cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu"


def device_report():
    """One-line human-readable summary of the available training device."""
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return (f"CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)} | "
                f"{p.total_memory / 1e9:.1f} GB VRAM | torch {torch.__version__}")
    return f"CPU only (torch {torch.__version__}; CUDA not available)"


def main(argv=None):
    print("Training device:", device_report())


if __name__ == "__main__":
    main()
