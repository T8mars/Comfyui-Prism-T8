import torch

# Small test matrices otherwise spend most of their time starting CPU threads.
torch.set_num_threads(4)
