import sys
sys.path.insert(0, "model")
sys.path.insert(0, ".")
import torch
from model.fewshot import FewShotSeg

torch.manual_seed(0)
B, H = 2, 512
q = torch.randn(B, 3, H, H)

for shot in (1, 5):
    m = FewShotSeg(shot=shot, reduce_dim=64, img_size=512)
    m.eval()
    s = torch.randn(B, shot, 3, H, H)
    sm = (torch.rand(B, shot, H, H) > 0.5).float()
    with torch.no_grad():
        out = m(q, s, sm)
    print(f"{shot}-shot out: {tuple(out.shape)}  range: [{out.min():.3f}, {out.max():.3f}]")

# guard: 473 must be rejected at construction
try:
    FewShotSeg(shot=1, reduce_dim=64, img_size=473)
except ValueError as e:
    print("473 guard OK:", e)
