import torch, time
assert torch.cuda.is_available()
x = torch.randn(4096,4096, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
t0=time.time()
y = x @ x
torch.cuda.synchronize()
print("ok", y.shape, "sec", time.time()-t0)
