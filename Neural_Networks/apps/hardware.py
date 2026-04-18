"""
Neural_Networks.config.hardware
================================
Auto-detects GPU / CPU hardware and returns a training profile dict.

The profile controls hardware-appropriate defaults for batch size,
number of workers, hidden layer sizes, and max training epochs.  It is consumed by the training wizard (``train_ui``) to seed
hyperparameter prompts with sensible values.

No UI or domain logic lives here — this is pure system introspection.
"""

from __future__ import annotations

import torch


def detect_hardware() -> dict:
    """
    Detect GPU and system specs and return a training profile dict.

    Profiles
    --------
    server  — A100 / H100 / A6000 (VRAM >= 40 GB)
    desktop — RTX 3080 / 4070 etc. (VRAM 8-39 GB)
    laptop  — RTX 3050 / 4050 etc. (VRAM < 8 GB but > 0)
    cpu     — no CUDA GPU found

    Returns
    -------
    dict
        Keys: ``profile``, ``gpu_name``, ``vram_gb``, ``ram_gb``,
        ``batch_size``, ``stride``, ``epochs``, ``workers``,
        ``prefetch``, ``compile``, ``hidden_size``, ``fc_layers``.
    """
    import psutil

    ram_gb = psutil.virtual_memory().total / 1e9

    if torch.cuda.is_available():
        props    = torch.cuda.get_device_properties(0)
        vram_gb  = props.total_memory / 1e9
        gpu_name = props.name
    else:
        vram_gb  = 0.0
        gpu_name = "CPU"

    # Classify hardware tier based on VRAM
    if vram_gb >= 40:
        profile = "server"
    elif vram_gb >= 8:
        profile = "desktop"
    elif vram_gb > 0:
        profile = "laptop"
    else:
        profile = "cpu"

    # Per-tier training defaults — tuned for robot torque PINN training
    hw_profiles: dict[str, dict] = {
        # A100/H100/A6000 — large VRAM, many cores, fast PCIe
        "server": {
            "batch_size": 2048, "stride": 1, "epochs": 1000,
            "workers": 8, "prefetch": 8, "compile": True,
            "hidden_size": 256, "fc_layers": [256, 128],

        },
        # RTX 3080/4070/4090 etc. — ample VRAM
        "desktop": {
            "batch_size": 1024, "stride": 1, "epochs": 700,
            "workers": 6, "prefetch": 6, "compile": True,
            "hidden_size": 192, "fc_layers": [192, 96],

        },
        # RTX 3050/3060/4050 Laptop — limited VRAM (~4-8 GB)
        # stride=1 maximises training windows; batch=512 fits in 4 GB VRAM.
        # 500 epochs: PINN models need ≥200 epochs past the data-fitting
        # plateau to diverge from black-box baselines.
        "laptop": {
            "batch_size": 512, "stride": 1, "epochs": 500,
            "workers": 4, "prefetch": 4, "compile": True,
            "hidden_size": 128, "fc_layers": [128, 64],

        },
        # CPU-only — conservative settings to avoid OOM / excessive wall time
        "cpu": {
            "batch_size": 128, "stride": 5, "epochs": 200,
            "workers": 2, "prefetch": 2, "compile": False,
            "hidden_size": 64, "fc_layers": [64],

        },
    }

    params = hw_profiles[profile]
    params.update({
        "profile":  profile,
        "gpu_name": gpu_name,
        "vram_gb":  round(vram_gb, 1),
        "ram_gb":   round(ram_gb, 1),
    })
    return params
