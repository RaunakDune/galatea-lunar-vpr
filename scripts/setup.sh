#!/usr/bin/env bash
# Download third-party model code and official pretrained weights (none are redistributed here).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party weights
[ -d third_party/dinov2 ]    || git clone --depth 1 https://github.com/facebookresearch/dinov2 third_party/dinov2
[ -d third_party/salad ]     || git clone --depth 1 https://github.com/serizba/salad third_party/salad
[ -d third_party/LightGlue ] || git clone --depth 1 https://github.com/cvg/LightGlue third_party/LightGlue
get() { [ -f "weights/$1" ] || curl -L --fail -o "weights/$1" "$2"; }
get dino_salad.ckpt     https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt
get cosplace_r50.pth    https://github.com/gmberton/CosPlace/releases/download/v1.0/ResNet50_512_cosplace.pth
get eigenplaces_r50.pth https://github.com/gmberton/EigenPlaces/releases/download/v1.0/ResNet50_2048_eigenplaces.pth
# SuperPoint / LightGlue weights are fetched automatically by torch.hub on first use.
echo "Setup complete."
