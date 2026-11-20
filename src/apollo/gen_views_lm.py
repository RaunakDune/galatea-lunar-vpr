"""Low-memory DESPINA generation for a 16 GB GPU shared with the desktop.

Same images as gen_views.py (prompt, LoRA scale 0.9, depth 0.8 / soft edge 0.5, 30 Euler steps,
guidance 7, seed per image), but the prompt is encoded once and both text encoders then move to
the CPU, which frees about 1.6 GB of GPU memory for the UNet and ControlNets.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--views', required=True)
ap.add_argument('--weights', required=True)
ap.add_argument('--despina', required=True)
ap.add_argument('--base', required=True)
ap.add_argument('--seeds', default='0')
ap.add_argument('--steps', type=int, default=30)
ap.add_argument('--size', type=int, default=1024)
ap.add_argument('--category', default='hilly')
a = ap.parse_args()
sys.path.insert(0, a.despina)
from despina import constraints as C
from despina.pipeline import Despina, GenConfig

root = Path(a.views)
views = json.load(open(root / 'views.json'))
W = Path(a.weights)
model = Despina('cuda:0', base=a.base, lora_path=str(W / 'lora'),
                controlnet_paths={'depth': str(W / 'controlnet-depth'), 'softedge': str(W / 'controlnet-softedge')})
cfg = GenConfig(name='full', steps=a.steps, size=a.size)
pipe = model.pipe
pipe.set_adapters(['lunar'], adapter_weights=[cfg.lora_scale])
prompt = C.dem_prompt(a.category)
with torch.no_grad():
    pe, npe, ppe, nppe = pipe.encode_prompt(prompt=prompt, device=torch.device('cuda:0'), num_images_per_prompt=1,
                                            do_classifier_free_guidance=True, negative_prompt=C.NEGATIVE_PROMPT)
pipe.text_encoder.to('cpu'); pipe.text_encoder_2.to('cpu')
torch.cuda.empty_cache()
print(f'GPU memory after loading: {torch.cuda.memory_allocated() / 2**30:.1f} GB', flush=True)
seeds = [int(s) for s in a.seeds.split(',')]
todo = [(v, s) for v in views for s in seeds if not (root / v['name'] / f'image_seed{s}.png').exists()]
print(f'{len(todo)} images to generate', flush=True)
t0 = time.time()
for i, (v, s) in enumerate(todo):
    d = root / v['name']
    imgs = [Image.open(d / f).convert('RGB').resize((cfg.size, cfg.size), Image.BICUBIC)
            for f in ('dem_depth.png', 'dem_softedge.png')]
    g = torch.Generator(device='cpu').manual_seed(s)
    out = pipe(prompt_embeds=pe, negative_prompt_embeds=npe, pooled_prompt_embeds=ppe,
               negative_pooled_prompt_embeds=nppe, image=imgs, controlnet_conditioning_scale=cfg.control_scales,
               control_guidance_end=cfg.control_end, num_inference_steps=cfg.steps, guidance_scale=cfg.guidance,
               height=cfg.size, width=cfg.size, generator=g).images[0]
    out.save(d / f'image_seed{s}.png')
    el = time.time() - t0
    print(f'[{i + 1}/{len(todo)}] {v["name"]} seed {s}  {el / (i + 1):.1f} s/image, peak '
          f'{torch.cuda.max_memory_allocated() / 2**30:.1f} GB, about {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min left',
          flush=True)
