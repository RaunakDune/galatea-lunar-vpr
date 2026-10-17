"""Model construction for frozen backbones / zero-shot baselines (no network access needed)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os
import sys, torch, torch.nn as nn, torch.nn.functional as F, torchvision

EXT = paths.EXT
W = paths.WEIGHTS


# ---------------------------------------------------------------- DINOv2 + SALAD (GSV-Cities)
class DinoSalad(nn.Module):
    """DINOv2-B/14 backbone fine-tuned by SALAD (last 4 blocks) + SALAD aggregator.
    forward_tokens returns the patch tokens [B,768,h,w] and class token [B,768]
    exactly as fed to the SALAD aggregator."""

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(f'{EXT}/dinov2', 'dinov2_vitb14', source='local', pretrained=False)
        import importlib.util
        spec = importlib.util.spec_from_file_location('salad_agg', f'{EXT}/salad/models/aggregators/salad.py')
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        SALAD = mod.SALAD
        self.agg = SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256)
        sd = torch.load(f'{W}/dino_salad.ckpt', map_location='cpu')
        bb = {k[len('backbone.model.'):]: v for k, v in sd.items() if k.startswith('backbone.model.')}
        ag = {k[len('aggregator.'):]: v for k, v in sd.items() if k.startswith('aggregator.')}
        print('dino load:', self.model.load_state_dict(bb, strict=True))
        print('salad load:', self.agg.load_state_dict(ag, strict=True))
        self.eval()

    @torch.no_grad()
    def forward_tokens(self, x):
        B, _, H, Wd = x.shape
        z = self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks:
            z = blk(z)
        z = self.model.norm(z)
        t = z[:, 0]
        f = z[:, 1:].reshape(B, H // 14, Wd // 14, 768).permute(0, 3, 1, 2)
        return f, t

    @torch.no_grad()
    def forward(self, x):
        f, t = self.forward_tokens(x)
        return self.agg((f, t))


# ---------------------------------------------------------------- CosPlace / EigenPlaces (ResNet-50)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)


class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        return x[:, :, 0, 0]


class GeoLocNet(nn.Module):
    def __init__(self, ckpt, fc_dim):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(r.children())[:-2])
        self.aggregation = nn.Sequential(L2Norm(), GeM(), Flatten(), nn.Linear(2048, fc_dim), L2Norm())
        print(ckpt, self.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True))
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        return self.aggregation(self.backbone(x))


def cosplace():
    return GeoLocNet(f'{W}/cosplace_r50.pth', 512)


def eigenplaces():
    return GeoLocNet(f'{W}/eigenplaces_r50.pth', 2048)


def superpoint(max_kp=512):
    sys.path.insert(0, f'{EXT}/LightGlue')
    from lightglue import SuperPoint
    return SuperPoint(max_num_keypoints=max_kp, detection_threshold=0.0005).eval()


def lightglue():
    sys.path.insert(0, f'{EXT}/LightGlue')
    from lightglue import LightGlue
    return LightGlue(features='superpoint', flash=False).eval()


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_rgb_tensor(gray_uint8_list, size=None):
    x = torch.from_numpy(__import__('numpy').stack(gray_uint8_list)).float().div(255.)[:, None]
    if size is not None:
        x = F.interpolate(x, size=size, mode='bilinear', antialias=True, align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return (x - MEAN) / STD
