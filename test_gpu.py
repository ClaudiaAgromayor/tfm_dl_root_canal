import time
import torch

for d in ['cpu', 'cuda:0']:
    x = torch.randn(4000, 4000, device=d)
    y = torch.randn(4000, 4000, device=d)
    if 'cuda' in d:
        torch.cuda.synchronize()
    t = time.time()
    _ = torch.matmul(x, y)
    if 'cuda' in d:
        torch.cuda.synchronize()
    name = torch.cuda.get_device_name(0) if 'cuda' in d else 'CPU'
    print(f"{d}: {time.time() - t:.4f}s ({name})")