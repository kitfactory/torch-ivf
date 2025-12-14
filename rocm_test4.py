import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)  # Windowsはまず0推奨
    p.add_argument("--steps", type=int, default=0, help="0: full epoch, >0: limit steps for quick smoke")
    p.add_argument("--amp", action="store_true", help="use autocast (mixed precision)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    # CIFAR-10 tutorial style normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])  # :contentReference[oaicite:2]{index=2}

    train_ds = datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = SmallCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler(enabled=(args.amp and device == "cuda"))

    # Train
    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=(args.amp and device == "cuda")):
                out = model(x)
                loss = crit(out, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if step % 50 == 0 or step == 1:
                acc = accuracy(out.detach(), y)
                print(f"epoch {epoch+1} step {step} loss {loss.item():.4f} acc {acc*100:.1f}%")

            if args.steps and step >= args.steps:
                break

    if device == "cuda":
        torch.cuda.synchronize()
    print(f"train done: {time.time()-t0:.1f}s")

    # Quick eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for step, (x, y) in enumerate(test_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            if step >= 50:  # 評価も軽め（必要なら外してOK）
                break
    print(f"eval (first 50 batches): acc {100.0*correct/total:.2f}%")

    print("OK")


if __name__ == "__main__":
    main()
