import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu[0]:", torch.cuda.get_device_name(0))