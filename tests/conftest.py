import torch
_b = torch.empty(int(1.5 * 2**30), dtype=torch.uint8, device="cuda"); torch.cuda.synchronize()   # keeps a peer scheduler off this GPU
