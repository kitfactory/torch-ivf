import torch, torch.nn as nn
m = nn.Conv2d(64, 128, 3, padding=1).cuda().half()
x = torch.randn(32,64,64,64, device="cuda", dtype=torch.float16)
y = m(x); torch.cuda.synchronize()
print(y.shape)
