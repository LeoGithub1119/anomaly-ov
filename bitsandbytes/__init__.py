import torch

class _BNBNotAvailable(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "bitsandbytes is not available in this container (shim). "
            "Do not use 8-bit/4-bit loading or bnb optimizers."
        )

class nn:
    class Linear8bitLt(_BNBNotAvailable): pass
    class Linear4bit(_BNBNotAvailable): pass

class optim:
    class Adam8bit(_BNBNotAvailable): pass
    class AdamW8bit(_BNBNotAvailable): pass

COMPILED_WITH_CUDA = False
__all__ = ["nn", "optim", "COMPILED_WITH_CUDA"]
