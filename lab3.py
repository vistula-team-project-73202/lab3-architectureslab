
"""Lab 3: Architectures and their costs  (MLP vs CNN vs Attention on Fashion-MNIST)

Run:  python lab3.py
Outputs: results.json, tradeoff.png, report.md
Install: pip install torch torchvision torchinfo thop matplotlib
"""
import copy
import json
import statistics
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
BATCH = 128
EPOCHS = 5
LR = 1e-3
TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAT_DEVICE = torch.device("cpu")  # latency measured on CPU for repeatability

torch.manual_seed(SEED)

# ----------------------------------------------------------------- 1. Data
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
full_train = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
train_set, val_set = random_split(full_train, [55000, 5000],
                                  generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH)
test_loader = DataLoader(test_set, batch_size=BATCH)


# --------------------------------------------------------------- 2. Models
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10))

    def forward(self, x):
        return self.net(x)


class PatchAttention(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.proj = nn.Linear(49, d)
        # learnable positional embedding (not in spec; without it patch order is lost)
        self.pos = nn.Parameter(torch.zeros(1, 16, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=128,
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(d, 10)

    def forward(self, x):
        b = x.shape[0]
        p = x.unfold(2, 7, 7).unfold(3, 7, 7).reshape(b, 16, 49)  # 16 patches of 7x7
        z = self.enc(self.proj(p) + self.pos)
        return self.head(z.mean(dim=1))


MODELS = {"MLP": MLP, "CNN": CNN, "Attention": PatchAttention}

# ---------------------------------------------------- 3a. Hand-calc FLOPs
def hand_flops():
    seq, d, ff = 16, 64, 128
    attn_layer = {
        "QKV projections": 2 * seq * d * 3 * d,
        "scores QK^T": 2 * seq * seq * d,
        "attn x V": 2 * seq * seq * d,
        "output projection": 2 * seq * d * d,
        "FFN linear1": 2 * seq * d * ff,
        "FFN linear2": 2 * seq * ff * d,
    }
    enc = sum(attn_layer.values())
    return {
        "MLP": {"example_layer": "Linear 784->256", "example_flops": 2 * 784 * 256,
                "total": 2 * (784 * 256 + 256 * 128 + 128 * 10)},
        "CNN": {"example_layer": "Conv2d 16->32 @14x14", "example_flops": 2 * 3 * 3 * 16 * 32 * 14 * 14,
                "total": 2 * 9 * 1 * 16 * 28 * 28 + 2 * 9 * 16 * 32 * 14 * 14 + 2 * 1568 * 10},
        "Attention": {"example_layer": "1 encoder layer (seq=16,d=64)", "example_flops": enc,
                      "breakdown": attn_layer,
                      "total": 2 * seq * 49 * d + 2 * enc + 2 * d * 10},
    }


# ------------------------------------------------------------ 3b. Latency
def latency_ms(model, batch, runs, warmup=10):
    model = copy.deepcopy(model).to(LAT_DEVICE).eval()
    x = torch.randn(batch, 1, 28, 28, device=LAT_DEVICE)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        for _ in range(runs):
            t = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


# ---------------------------------------------------------------- 4. Train
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    for x, y in loader:
        x, y = x.to(TRAIN_DEVICE), y.to(TRAIN_DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)


def train(model):
    model.to(TRAIN_DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    hist = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot = 0.0
        for x, y in train_loader:
            x, y = x.to(TRAIN_DEVICE), y.to(TRAIN_DEVICE)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        va = evaluate(model, val_loader)
        hist.append({"epoch": ep, "train_loss": tot / len(train_set), "val_acc": va})
        print(f"  epoch {ep}: loss {hist[-1]['train_loss']:.4f}  val_acc {va:.4f}")
    return hist


# -------------------------------------------------------------------- Main
hand = hand_flops()
results = {}
for name, cls in MODELS.items():
    print(f"\n===== {name} =====")
    torch.manual_seed(SEED)
    model = cls()
    info = summary(model, input_size=(1, 1, 28, 28), verbose=0)
    params = info.total_params
    print(info)

    macs, _ = profile(copy.deepcopy(model).cpu().eval(), inputs=(torch.randn(1, 1, 28, 28),),
                      verbose=False)
    lat1 = latency_ms(model, 1, 100)
    lat128 = latency_ms(model, 128, 20)
    print(f"params={params}  hand FLOPs={hand[name]['total']}  thop MACs={int(macs)} "
          f"(x2 = {int(2 * macs)})  lat b1={lat1:.3f} ms  lat b128={lat128:.3f} ms")

    hist = train(model)
    test_acc = evaluate(model, test_loader)
    print(f"  TEST ACC {test_acc:.4f}")
    results[name] = {"params": params, "hand_total_flops": hand[name]["total"],
                     "thop_macs": int(macs), "thop_flops_x2": int(2 * macs),
                     "lat_b1_ms": lat1, "lat_b128_ms": lat128,
                     "test_acc": test_acc, "history": hist}

json.dump({"results": results, "hand": hand}, open("results.json", "w"), indent=2)

# ---------------------------------------------------------------- 5. Plots
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
panels = [("params", "Parameters"), ("lat_b1_ms", "Latency, batch 1 (ms, CPU)"),
          ("lat_b128_ms", "Latency, batch 128 (ms, CPU)")]
for a, (key, label) in zip(ax, panels):
    for n, r in results.items():
        a.scatter(r[key], r["test_acc"], s=80)
        a.annotate(n, (r[key], r["test_acc"]), textcoords="offset points", xytext=(6, 6))
    a.set_xlabel(label)
    a.set_ylabel("Test accuracy")
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("tradeoff.png", dpi=150)

# --------------------------------------------------------------- 6. Report
R = results
best_acc = max(R, key=lambda n: R[n]["test_acc"])
best_eff1 = max(R, key=lambda n: R[n]["test_acc"] / R[n]["lat_b1_ms"])
best_eff128 = max(R, key=lambda n: R[n]["test_acc"] / R[n]["lat_b128_ms"])
bd = "\n".join(f"| {k} | {v:,} |" for k, v in hand["Attention"]["breakdown"].items())

rows_cost = "\n".join(
    f"| {n} | {r['params']:,} | {r['hand_total_flops']:,} | {r['thop_macs']:,} | "
    f"{r['thop_flops_x2']:,} | {r['thop_flops_x2'] / r['hand_total_flops']:.2f} |"
    for n, r in R.items())
rows_lat = "\n".join(
    f"| {n} | {r['lat_b1_ms']:.3f} | {r['lat_b128_ms']:.3f} | {r['test_acc']:.4f} |"
    for n, r in R.items())
rows_ex = "\n".join(
    f"| {n} | {h['example_layer']} | {h['example_flops']:,} |" for n, h in hand.items())

report = f"""# Lab 3: Architectures and their costs

## 1. Setup
Fashion-MNIST, Normalize((0.2860,), (0.3530,)), 55k/5k/10k train/val/test split (seed {SEED}),
batch {BATCH}. Training: Adam, lr {LR}, {EPOCHS} epochs on `{TRAIN_DEVICE}`.
Latency: CPU, eval mode, `torch.no_grad()`, 10 warm-up runs, median of 100 runs (batch 1) and 20 runs (batch 128).

## 2. Models and parameter counts (torchinfo)
Models follow the lab spec exactly. Note that the specified layer sizes do **not** all land in the
100k-150k target: the MLP is above it and the CNN and Attention models are below it (table below).
The Attention model also has a learnable positional embedding (16x64 = 1,024 params), which the spec does not
mention; I added it because mean-pooling alone would discard patch position.

## 3. Cost: hand-calculated vs measured FLOPs
Hand-calculated FLOPs for one layer of each type:

| Model | Layer | FLOPs |
|---|---|---|
{rows_ex}

Breakdown of one Transformer encoder layer (seq=16, d_model=64, ff=128):

| Term | FLOPs |
|---|---|
{bd}

Whole-model comparison (batch 1):

| Model | Params | Hand FLOPs | thop MACs | thop MACs x2 | thop x2 / hand |
|---|---|---|---|---|---|
{rows_cost}

**Explaining the gap.**
1. `thop` reports multiply-accumulates (MACs), and one MAC = 2 FLOPs, so compare against MACs x2.
2. `thop` only has counting rules for known module types. `nn.MultiheadAttention` is not handled, and its internal
   QK^T and attention x V matmuls are functional ops, so the attention scores are not counted at all.
3. My hand count ignores bias adds, ReLU, softmax, LayerNorm, residual adds and pooling (low-FLOP but memory-bound
   ops). `thop` ignores most of these as well, so any small remaining difference comes from them.

## 4. Latency and accuracy

| Model | Latency b=1 (ms) | Latency b=128 (ms) | Test acc |
|---|---|---|---|
{rows_lat}

## 5. Trade-off plots
![tradeoff](tradeoff.png)

## 6. Which architecture is most cost-effective?
Highest test accuracy: **{best_acc}**. Best accuracy per ms: **{best_eff1}** at batch 1 and **{best_eff128}** at batch 128.

Access-pattern explanation (Ch. 4.6-4.7):
- **MLP** uses dense matrix-vector or matrix-matrix products. Each weight is used once per sample, so at batch 1
  arithmetic intensity is about 2 FLOPs per weight read and the layer is memory-bandwidth bound. Larger batches reuse each
  loaded weight across samples, so latency per sample drops sharply.
- **CNN** shares one small kernel across all spatial positions, giving heavy weight reuse and good locality
  (sliding-window access, small working set that fits in cache). It has few parameters but relatively many FLOPs,
  so it is compute-bound rather than bandwidth-bound.
- **Attention** needs dense projections plus seq^2 score matrices, softmax and LayerNorm, which are element-wise,
  memory-bound ops. It also launches many more small kernels, so it pays fixed overhead that dominates at batch 1.
  Its cost grows quadratically with sequence length, which is harmless at seq=16 but would not scale.

Conclusion: *(edit in your own words after checking the numbers above match this reasoning)*
"""
open("report.md", "w").write(report)
print("\nDone. Wrote results.json, tradeoff.png, report.md")