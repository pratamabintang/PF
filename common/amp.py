r""" AMP / device helpers. Flexible dtype: bf16 target (L4), fp16 fallback, fp32 off/CPU. """
import torch


def get_device(prefer_cuda=True):
    return torch.device("cuda:0" if (prefer_cuda and torch.cuda.is_available()) else "cpu")


def resolve_amp_dtype(device, requested="auto"):
    if device.type != "cuda":
        return None
    major = torch.cuda.get_device_properties(device).major
    if requested == "auto":
        return torch.bfloat16 if major >= 8 else torch.float16
    v = (requested or "").lower()
    if v in ("bf16", "bfloat16"):
        if major < 8:
            raise RuntimeError(f"bf16 requires Ampere+ GPU (CC>=8.0), this GPU CC={major}. Use --amp fp16 or off.")
        return torch.bfloat16
    if v in ("fp16", "float16", "half"):
        return torch.float16
    if v in ("off", "none", "fp32", "float32"):
        return None
    raise ValueError(f"unknown --amp value: {requested} (choices: auto, bf16, fp16, off)")


def amp_context(dtype):
    """autocast context; no-op when dtype is None. bf16 never needs GradScaler."""
    if dtype is None:
        import contextlib
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_scaler(dtype):
    """GradScaler only for fp16; bf16 is self-scaling, fp32/off get None."""
    if dtype == torch.float16:
        return torch.amp.GradScaler("cuda")
    return None
