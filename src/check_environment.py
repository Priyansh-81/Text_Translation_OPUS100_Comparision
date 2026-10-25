import sys, platform
import torch
from datasets import __version__ as datasets_version
from transformers import __version__ as transformers_version
from src.common import device_name

print("Python:", sys.version)
print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("Transformers:", transformers_version)
print("Datasets:", datasets_version)
print("Device:", device_name())
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
